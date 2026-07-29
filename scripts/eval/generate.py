"""Sample completions from a checkpoint against the fixed validation prompts.

Two checkpoint layouts are supported:

* a Hugging Face directory saved by ``save_pretrained`` (the default), and
* a raw DeepSpeed ZeRO state directory written by ``accelerator.save_state``,
  via ``--from_zero_checkpoint``.

Examples:
    # Base model, prompts from dataset.validation.val_set_pretrain
    python scripts/eval/generate.py --checkpoint ckpt/vocab_32k_gpt2

    # Instruction-tuned model, prompts pre-rendered with the SFT template
    python scripts/eval/generate.py \\
        --checkpoint ckpt/vocab_32k_gpt2_instruction/checkpoint_epoch4 \\
        --prompt_set sft

    # Resume directly from a DeepSpeed ZeRO checkpoint
    python scripts/eval/generate.py \\
        --checkpoint ckpt/vocab_32k_gpt2 --from_zero_checkpoint
"""

import argparse
import logging
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.validation import val_set_pretrain, val_set_sft  # noqa: E402
from models.configuration_vocab_32k_gpt2 import vocab_32k_gpt2Config  # noqa: E402
from models.modeling_vocab_32k_gpt2 import vocab_32k_GPT2LMHeadModel  # noqa: E402
from models.tokenization_vocab_32k_gpt2 import vocab_32k_gpt2Tokenizer  # noqa: E402

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PROMPT_SETS = {"pretrain": val_set_pretrain, "sft": val_set_sft}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="ckpt/vocab_32k_gpt2",
        help="Checkpoint directory to load.",
    )
    parser.add_argument(
        "--prompt_set",
        default="pretrain",
        choices=sorted(PROMPT_SETS),
        help="Which prompt set from dataset/validation.py to run.",
    )
    parser.add_argument(
        "--model_config",
        default="configs/model_configs/vocab_32k_gpt2.json",
        help="Path to the model config JSON.",
    )
    parser.add_argument(
        "--tokenizer_model",
        default="configs/tokenizer_models/vocab_32k_gpt2.model",
        help="Path to the SentencePiece tokenizer model.",
    )
    parser.add_argument(
        "--from_zero_checkpoint",
        action="store_true",
        help="Read a DeepSpeed ZeRO state directory instead of a save_pretrained directory.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--greedy", action="store_true", help="Disable sampling and decode greedily."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run generation on.",
    )
    return parser.parse_args()


def load_model(args):
    model_config = vocab_32k_gpt2Config.from_pretrained(args.model_config)

    if args.from_zero_checkpoint:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        model = vocab_32k_GPT2LMHeadModel(config=model_config)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint)
        model.load_state_dict(state_dict)
    else:
        model = vocab_32k_GPT2LMHeadModel.from_pretrained(
            args.checkpoint, config=model_config
        )

    model.eval()
    # fp16 halves memory and is sufficient for qualitative inspection.
    if args.device.startswith("cuda"):
        model = model.half()
    model = model.to(args.device)
    logger.info("Loaded %s, ready on %s.", args.checkpoint, args.device)
    return model


def main():
    args = parse_args()
    tokenizer = vocab_32k_gpt2Tokenizer(args.tokenizer_model, legacy=False)
    model = load_model(args)

    for prompt in PROMPT_SETS[args.prompt_set]:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            pred = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        completion = tokenizer.decode(
            pred[0, input_length:].cpu(), skip_special_tokens=True
        )
        print(prompt, "\n", completion, "\n")


if __name__ == "__main__":
    main()
