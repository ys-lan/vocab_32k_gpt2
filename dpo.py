"""Direct Preference Optimization (DPO) on top of an SFT checkpoint.

The script expects a JSONL preference dataset where every row contains a
``question`` together with a preferred (``response_j``) and a rejected
(``response_k``) completion. Rows are rendered into the same instruction
template used during SFT so that the policy stays in-distribution.

Example:
    accelerate launch --config_file configs/accelerate_configs/ds_stage2.yaml dpo.py
"""

import os
import random
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import DPOTrainer

from models.configuration_vocab_32k_gpt2 import vocab_32k_gpt2Config
from models.modeling_vocab_32k_gpt2 import vocab_32k_GPT2LMHeadModel
from models.tokenization_vocab_32k_gpt2 import vocab_32k_gpt2Tokenizer


@dataclass
class ScriptArguments:
    """Command line arguments for the DPO training script."""

    # Loss
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )

    # Model / tokenizer
    model_name_or_path: Optional[str] = field(
        default="ckpt/vocab_32k_gpt2_sft4dpo/checkpoint_epoch6",
        metadata={"help": "the location of the SFT model name or path"},
    )
    model_config_path: Optional[str] = field(
        default="configs/model_configs/vocab_32k_gpt2.json",
        metadata={"help": "path to the model config JSON"},
    )
    tokenizer_model_path: Optional[str] = field(
        default="configs/tokenizer_models/vocab_32k_gpt2.model",
        metadata={"help": "path to the SentencePiece tokenizer model"},
    )
    model_dtype: Optional[str] = field(
        default="float",
        metadata={"help": "model_dtype[float16, bfloat16, float] for loading."},
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "whether to load the model in 4bit"}
    )

    # Data
    train_data_pattern: Optional[str] = field(
        default="data/DPO/mix_dpo_data.jsonl",
        metadata={"help": "glob pattern of the preference JSONL shards"},
    )
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"}
    )
    num_proc: Optional[int] = field(
        default=24, metadata={"help": "processes used to preprocess the raw dataset"}
    )
    dataset_num_proc: Optional[int] = field(
        default=32, metadata={"help": "processes used by DPOTrainer for tokenization"}
    )

    # Optimization
    learning_rate: Optional[float] = field(
        default=5e-4, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(
        default=500, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=5, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use reentrant for gradient checkpointing"},
    )
    max_steps: Optional[int] = field(
        default=50000, metadata={"help": "max number of training steps"}
    )

    # LoRA (unused unless a peft_config is passed to the trainer)
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    # Checkpointing / logging
    output_dir: Optional[str] = field(
        default="./ckpt/vocab_32k_gpt2_dpo/", metadata={"help": "the output directory"}
    )
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(
        default=10000, metadata={"help": "the evaluation frequency"}
    )
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    # Instrumentation
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )


def return_prompt_and_responses(samples) -> dict[str, str]:
    """Render raw preference rows into the prompt/chosen/rejected schema TRL expects."""
    return {
        "prompt": [
            "### Instruction: " + question + "\n\n### System: \n"
            for question in samples["question"]
        ],
        "chosen": samples["response_j"],
        "rejected": samples["response_k"],
    }


def get_dataset_paired(
    data_patterns,
    sanity_check: bool = False,
    num_proc: int = 24,
) -> Dataset:
    """Load a paired-preference dataset from JSONL shards.

    Each raw row must have the following structure::

        {
            "question": str,
            "response_j": str,   # preferred over response_k
            "response_k": str,
        }

    Prompts are rendered as ``"### Instruction: " + question + "\\n\\n### System: \\n"``.
    """
    all_data_files = []
    for _, pattern in data_patterns.items():
        data_files = glob(pattern)
        assert data_files, f"No files matched the data pattern: {pattern}"
        all_data_files.extend(data_files)
    random.shuffle(all_data_files)

    dataset = load_dataset(
        "json", data_files=all_data_files, split="train", streaming=False
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }.get(name, torch.float)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. Tokenizer and policy model, initialised from the SFT checkpoint.
    tokenizer = vocab_32k_gpt2Tokenizer(
        vocab_file=script_args.tokenizer_model_path, legacy=False
    )
    # TRL pads preference pairs with the EOS token, matching the Hub convention
    # for models whose tokenizer ships without a dedicated pad token.
    tokenizer.pad_token = tokenizer.eos_token

    model_config = vocab_32k_gpt2Config.from_pretrained(script_args.model_config_path)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_id

    model = vocab_32k_GPT2LMHeadModel.from_pretrained(
        script_args.model_name_or_path,
        config=model_config,
        low_cpu_mem_usage=True,
        torch_dtype=resolve_dtype(script_args.model_dtype),
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # Boolean buffers break DDP's gradient bucketing; exclude them explicitly.
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # 2. Preference dataset. New sources must first be converted to the
    #    question/response_j/response_k schema documented above.
    train_dataset = get_dataset_paired(
        data_patterns={"mix_dpo_dataset": script_args.train_data_pattern},
        sanity_check=script_args.sanity_check,
        num_proc=script_args.num_proc,
    )

    # 3. Training arguments.
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="no",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="vocab_32k_gpt2_dpo",
        gradient_checkpointing_kwargs=dict(
            use_reentrant=script_args.gradient_checkpointing_use_reentrant
        ),
        seed=script_args.seed,
    )

    # 4. DPO trainer. ref_model=None makes TRL use the frozen initial policy as
    #    the reference, which halves memory compared to a second model copy.
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        dataset_num_proc=script_args.dataset_num_proc,
    )

    # 5. Train and export.
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)
    dpo_trainer.model.save_pretrained(
        os.path.join(script_args.output_dir, "final_checkpoint")
    )


if __name__ == "__main__":
    main()
