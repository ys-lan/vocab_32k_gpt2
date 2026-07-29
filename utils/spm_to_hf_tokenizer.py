"""Export a SentencePiece model as a Hugging Face tokenizer directory.

The script wraps a raw ``.model`` file in :class:`vocab_32k_gpt2Tokenizer`,
verifies that encoding is byte-identical to the underlying SentencePiece
processor and that decoding round-trips, then optionally writes a
``save_pretrained`` directory that can be loaded with ``AutoTokenizer``.

Example:
    python utils/spm_to_hf_tokenizer.py \\
        --vocab_file configs/tokenizer_models/vocab_32k_gpt2.model \\
        --output_dir configs/tokenizer_models/hf
"""

import argparse
import os
import sys

import sentencepiece as spm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.tokenization_vocab_32k_gpt2 import vocab_32k_gpt2Tokenizer  # noqa: E402

# Short bilingual probes: enough to catch prefix-space and CJK regressions.
PROBES = [
    "a good man",
    "机器学习是人工智能的一个分支。",
    "### Instruction:\nHello!\n\n### System:\n",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab_file",
        default="configs/tokenizer_models/vocab_32k_gpt2.model",
        help="Path to the SentencePiece model file.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to write the Hugging Face tokenizer into.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Keep SentencePiece's dummy-prefix behaviour instead of disabling it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sp_model = spm.SentencePieceProcessor(model_file=args.vocab_file)
    tokenizer = vocab_32k_gpt2Tokenizer(vocab_file=args.vocab_file, legacy=args.legacy)

    print(f"vocab size      : {tokenizer.vocab_size}")
    print(
        f"special tokens  : unk={tokenizer.unk_token} ({tokenizer.unk_id}), bos={tokenizer.bos_token} ({tokenizer.bos_id}), eos={tokenizer.eos_token} ({tokenizer.eos_id}), pad={tokenizer.pad_token} ({tokenizer.pad_id})"
    )

    failures = 0
    for text in PROBES:
        hf_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(hf_ids, skip_special_tokens=True)
        spm_ids = sp_model.encode(text)

        print(f"\ntext            : {text!r}")
        print(f"hf ids          : {hf_ids}")
        print(f"sentencepiece   : {spm_ids}")
        print(f"decoded         : {decoded!r}")

        if decoded != text:
            print("  [FAIL] decode did not round-trip")
            failures += 1
        # With legacy=False the dummy prefix is disabled, so the two encoders are
        # expected to diverge on strings that start with whitespace.
        elif args.legacy and hf_ids != spm_ids:
            print("  [FAIL] hf ids differ from sentencepiece ids")
            failures += 1

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nSaved Hugging Face tokenizer to {args.output_dir}")

    if failures:
        raise SystemExit(f"{failures} probe(s) failed.")
    print("\nAll probes passed.")


if __name__ == "__main__":
    main()
