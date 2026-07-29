"""Streaming data pipeline shared by the pretraining and SFT stages.

The pipeline is built on top of `datasets` iterable datasets so that corpora of
arbitrary size can be consumed without materialising them on disk:

1. **Load**   -- expand the glob patterns from the config and stream the shards.
2. **Normalise** -- map heterogeneous raw records onto a single ``text`` column
   (plus an ``answer`` column in ``instruct`` mode).
3. **Tokenise** -- encode to ``input_ids``, padding or truncating to ``seq_length``.
4. **Pack**   -- optionally sample, split or concatenate documents so that every
   emitted example is exactly ``seq_length`` tokens long.
5. **Label**  -- derive ``labels``, masking padding (and the prompt, in ``instruct``
   mode) with ``-100`` so it does not contribute to the loss.

Adapted from the Open-Llama project (``dataset/dataset.py``).
"""

import copy
import math
import random
from glob import glob

import torch
from datasets import load_dataset

from models.tokenization_vocab_32k_gpt2 import vocab_32k_gpt2Tokenizer

random.seed(42)

# Sentinel used to keep the turns of a conversation together while they travel
# through `datasets.map`, before being split back into independent examples.
MULTITURN_SEP = "[multiturn_sep]"

# Token id used by Hugging Face to mark positions that must be ignored by the
# cross-entropy loss.
IGNORE_INDEX = -100


def pretrain_transform(batch):
    """Normalise a raw pretraining record onto the ``text`` column.

    Corpora such as SkyPile-150B and OpenWebText already expose a ``text``
    field, so this is a pass-through. Add a branch here when onboarding a corpus
    that stores its content under different keys (for example a title/body pair
    that needs to be concatenated).
    """
    if "text" not in batch:
        raise ValueError(
            f"Unrecognised pretraining record: expected a 'text' field, got {sorted(batch.keys())}."
        )
    return batch


def _prompt_no_input(row):
    """Alpaca-style template, kept for reference and easy A/B testing."""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{output}</s>"
    ).format_map(row)


def _prompt_input(row):
    """Alpaca-style template with an extra context field, kept for reference."""
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}</s>"
    ).format_map(row)


def prompt_no_input_no_history(row):
    """Render a single-turn instruction/response pair."""
    return ("### Instruction:\n{instruction}\n\n### System:\n{output}</s>").format_map(row)


def prompt_input(row):
    """Render an instruction/response pair that carries additional input context."""
    return (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### System:\n{output}</s>"
    ).format_map(row)


def instruct_transform(batch):
    """Render an instruction record into a prompt/answer pair.

    Three shapes are supported, in priority order: records with an ``input``
    context, multi-turn records carrying a ``history`` of ``(user, assistant)``
    pairs, and plain single-turn records. Multi-turn conversations are joined
    with :data:`MULTITURN_SEP` and split apart again by :func:`split_multiturn`.
    """
    answer = []
    if batch["input"] != [""]:
        text = prompt_input(
            {
                "instruction": batch["instruction"][0],
                "input": batch["input"][0],
                "output": batch["output"][0],
            }
        )
        answer.append(batch["output"][0] + "</s>")

    elif batch["history"][0]:
        chats = []
        for user_turn, assistant_turn in batch["history"][0]:
            chats.append(
                prompt_no_input_no_history(
                    {"instruction": user_turn, "output": assistant_turn}
                )
            )
            answer.append(assistant_turn + "</s>")
        chats.append(
            prompt_no_input_no_history(
                {
                    "instruction": batch["instruction"][0],
                    "output": batch["output"][0],
                }
            )
        )
        answer.append(batch["output"][0] + "</s>")
        text = MULTITURN_SEP.join(chats)

    else:
        text = prompt_no_input_no_history(
            {"instruction": batch["instruction"][0], "output": batch["output"][0]}
        )
        answer.append(batch["output"][0] + "</s>")

    return {"text": [text], "answer": [MULTITURN_SEP.join(answer)]}


def split_multiturn(batch):
    """Expand a joined conversation back into one example per turn."""
    return {
        "text": batch["text"][0].split(MULTITURN_SEP),
        "answer": batch["answer"][0].split(MULTITURN_SEP),
    }


def sample_sequence_gen(seq_length, eos_token_id):
    """Return a mapper that crops a random ``seq_length`` window out of a document.

    The window starts at the beginning of the document one time out of four,
    which keeps document openings represented in the training distribution.
    """

    def sample_sequence(line):
        doc_length = line["input_ids"].shape[0]
        if doc_length <= seq_length:
            start = 0
        else:
            if random.random() < 1 / 4:
                start = 0
            else:
                start = random.randint(0, doc_length - seq_length)
        input_ids = line["input_ids"][start : start + seq_length]
        if input_ids[-1] != eos_token_id:
            input_ids[-1] = eos_token_id
        return {"input_ids": input_ids}

    return sample_sequence


def split_sequence_gen(seq_length):
    """Return a mapper that chops a document into consecutive ``seq_length`` chunks."""

    def split_sequence(batch):
        input_ids = batch["input_ids"][0]
        out = []
        while len(input_ids) >= (1 + len(out)) * seq_length:
            out.append(input_ids[len(out) * seq_length : (1 + len(out)) * seq_length])
        return {"input_ids": out}

    return split_sequence


def concat_multiple_sequence_gen(seq_length, pad_id):
    """Return a mapper that packs several documents into full-length sequences.

    Documents are concatenated, right-padded to a multiple of ``seq_length`` and
    then chunked, which removes almost all padding waste during pretraining.
    """

    def concat_multiple_sequence(batch):
        concat_input_ids = torch.cat(batch["input_ids"], dim=0)
        length = concat_input_ids.shape[0]
        chunks = math.ceil(length / seq_length)
        pad_length = chunks * seq_length - length
        pad = torch.ones(pad_length, dtype=concat_input_ids.dtype) * pad_id
        concat_input_ids = torch.cat([concat_input_ids, pad], dim=0)
        input_ids = torch.chunk(concat_input_ids, chunks)
        return {"input_ids": input_ids}

    return concat_multiple_sequence


def get_labels_gen(pad_id):
    """Return a mapper that builds causal-LM labels, ignoring padding."""

    def get_labels(line):
        input_ids = line["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_id] = IGNORE_INDEX
        return {"labels": labels}

    return get_labels


def get_sft_labels_gen(example, pad_id):
    """Mask padding inside labels that were already prompt-masked by :func:`mask_process`."""
    labels = example["labels"]
    labels[labels == pad_id] = IGNORE_INDEX
    return {"labels": labels}


def mask_process(example, tokenizer, seq_length):
    """Tokenise an instruction example and mask the prompt out of the loss.

    ``answer`` is a suffix of ``text``, so the prompt length is recovered by
    tokenising ``text`` with the answer stripped off and counting non-pad tokens.
    """
    all_text = example["text"]
    target = example["answer"]
    source = example["text"][: len(all_text) - len(target)]

    all_text_tokenized = tokenizer(
        all_text,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
        max_length=seq_length,
        truncation=True,
    )
    input_ids = all_text_tokenized.input_ids[0]

    source_tokenized = tokenizer(
        source,
        return_tensors="pt",
        return_attention_mask=False,
        padding="max_length",
        max_length=seq_length,
        truncation=True,
    )
    source_len = source_tokenized["input_ids"][0].ne(tokenizer.pad_id).sum().item()

    labels = copy.deepcopy(input_ids)
    labels[:source_len] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": all_text_tokenized["attention_mask"][0],
    }


def construct_dataset(dataset_config, tokenizer, return_raw_text=False, world_size=None):
    """Build the streaming training dataset described by ``dataset_config``.

    Args:
        dataset_config: The ``data`` section of a training config.
        tokenizer: Tokenizer used to encode the ``text`` column.
        return_raw_text: Reserved for debugging; the raw-text short circuit is
            currently disabled so that callers always receive tokenised data.
        world_size: When set, the shard list is truncated to a multiple of the
            world size so that ``split_dataset_by_node`` can partition by shard.

    Returns:
        An iterable dataset yielding ``input_ids``/``labels`` (plus
        ``attention_mask`` in ``instruct`` mode).
    """
    is_instruct = dataset_config["mode"] == "instruct"

    all_data_files = []
    for _, pattern in dataset_config["data"].items():
        data_files = glob(pattern)
        assert data_files, f"No files matched the data pattern: {pattern}"
        all_data_files.extend(data_files)
    random.shuffle(all_data_files)

    # When the shard count divides the world size, `split_dataset_by_node` splits
    # by shard; otherwise every rank reads everything and discards most of it.
    # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.distributed.split_dataset_by_node
    if world_size is not None:
        num_shards = len(all_data_files)
        all_data_files = all_data_files[: num_shards // world_size * world_size]

    dataset = load_dataset(
        "json", data_files=all_data_files, split="train", streaming=True
    )
    dataset = dataset.shuffle(seed=42)

    # Normalise heterogeneous records onto a single schema.
    if dataset_config["mode"] == "pretrain":
        dataset = dataset.map(pretrain_transform, batched=True, batch_size=1)
    elif is_instruct:
        dataset = dataset.map(instruct_transform, batched=True, batch_size=1)
        dataset = dataset.select_columns(["text", "answer"])
        dataset = dataset.map(split_multiturn, batched=True, batch_size=1)
    else:
        raise Exception("Dataset mode: {} not found.".format(dataset_config["mode"]))

    full_dataset = dataset

    seq_length = dataset_config["seq_length"]
    pad_to_max = dataset_config.get("pad_to_max", True)
    sequence_sample_mode = dataset_config.get("sequence_sample_mode", "truncation")
    truncation = sequence_sample_mode == "truncation"
    concat_multiple_sequence = dataset_config.get("concat_multiple_sequence", False)

    # Tokenise.
    if pad_to_max:
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                return_attention_mask=False,
                padding="max_length",
                max_length=seq_length,
                truncation=truncation,
            )
        )
    else:
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x["text"],
                return_tensors="pt",
                return_attention_mask=False,
                truncation=truncation,
            )
        )

    # Reduce to the tensors consumed by the model.
    if is_instruct:
        full_dataset = full_dataset.map(
            lambda example: mask_process(example, tokenizer, seq_length), batched=False
        )
        full_dataset = full_dataset.select_columns(
            ["input_ids", "labels", "attention_mask"]
        )
    else:
        full_dataset = full_dataset.map(lambda x: {"input_ids": x["input_ids"][0]})
        full_dataset = full_dataset.select_columns("input_ids")

    # Turn variable-length documents into fixed-length training sequences.
    if sequence_sample_mode in ("truncation", "none"):
        pass
    elif sequence_sample_mode == "sample":
        assert pad_to_max or concat_multiple_sequence
        full_dataset = full_dataset.map(
            sample_sequence_gen(seq_length, tokenizer.eos_token_id)
        )
    elif sequence_sample_mode == "split":
        assert not concat_multiple_sequence
        full_dataset = full_dataset.map(
            split_sequence_gen(seq_length), batched=True, batch_size=1
        )
    else:
        raise Exception(
            f"Unknown sequence_sample mode: {sequence_sample_mode}."
        )

    if concat_multiple_sequence:
        num_sequences = dataset_config["num_sequences"]
        full_dataset = full_dataset.map(
            concat_multiple_sequence_gen(seq_length, tokenizer.pad_id),
            batched=True,
            batch_size=num_sequences,
            drop_last_batch=True,
        )

    # Attach labels.
    if is_instruct:
        full_dataset = full_dataset.map(
            lambda example: get_sft_labels_gen(example, tokenizer.pad_id)
        )
    else:
        full_dataset = full_dataset.map(get_labels_gen(tokenizer.pad_id))

    return full_dataset.shuffle(seed=42)


if __name__ == "__main__":
    # Smoke test: verify tokenizer round-trips and that batches have the right shape.
    import time

    from torch.utils.data import DataLoader

    data_config = {
        "mode": "instruct",
        "data": {"mixed": "test.jsonl"},
        "pad_to_max": True,
        "sequence_sample_mode": "truncation",
        "concat_multiple_sequence": False,
        "num_sequences": 10,
        "seq_length": 1000,
        "tokenizer_model_path": "configs/tokenizer_models/vocab_32k_gpt2.model",
    }
    tokenizer = vocab_32k_gpt2Tokenizer(
        vocab_file=data_config["tokenizer_model_path"], legacy=False
    )

    pretrain_dataset = construct_dataset(data_config, tokenizer, True)
    start = time.time()
    for i, line in enumerate(pretrain_dataset):
        raw_text = line["text"]
        input_ids = tokenizer(
            line["text"], return_tensors="pt", return_attention_mask=False
        )["input_ids"][0]
        decode_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        if raw_text != decode_text and "▁" not in raw_text:
            print(raw_text, "\n", decode_text)
        if i == 10:
            break
    print(f"all checked in {time.time() - start:.2f} seconds.")

    pretrain_dataset = construct_dataset(data_config, tokenizer)
    print(pretrain_dataset.n_shards)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=2, num_workers=16)
    for batch in pretrain_loader:
        for k, v in batch.items():
            print(k, v.shape, "\n", v)
        break
