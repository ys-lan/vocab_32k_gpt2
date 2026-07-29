"""Unified training entrypoint for pretraining and supervised fine-tuning (SFT).

The training stage is selected by ``data.mode`` inside the training config:

* ``pretrain``  -- causal language modelling on raw text corpora.
* ``instruct``  -- supervised fine-tuning on instruction/response pairs.

Example:
    accelerate launch --config_file configs/accelerate_configs/ds_stage2.yaml \\
        train.py \\
        --train_config configs/pretrain_config.yaml \\
        --model_config configs/model_configs/vocab_32k_gpt2.json

Adapted from the Open-Llama project (``train_lm.py``).
"""

import logging

import yaml
from absl import app, flags
from accelerate import Accelerator
from datasets.distributed import split_dataset_by_node
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader

from dataset.dataset import construct_dataset
from models.configuration_vocab_32k_gpt2 import vocab_32k_gpt2Config
from models.modeling_vocab_32k_gpt2 import vocab_32k_GPT2LMHeadModel
from models.tokenization_vocab_32k_gpt2 import vocab_32k_gpt2Tokenizer
from trainer import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string("train_config", None, "Path to the training config (YAML).")
flags.DEFINE_string("model_config", None, "Path to the model config (JSON).")
flags.mark_flags_as_required(["train_config", "model_config"])

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_tokenizer(data_config):
    return vocab_32k_gpt2Tokenizer(
        vocab_file=data_config["tokenizer_model_path"], legacy=False
    )


def build_dataloader(config, tokenizer, accelerator):
    """Build a sharded, streaming dataloader for the current process."""
    data_config = config["data"]
    train_config = config["train"]

    if data_config.get("split_by_shard", False):
        train_dataset = construct_dataset(
            data_config, tokenizer, world_size=accelerator.num_processes
        )
    else:
        train_dataset = construct_dataset(data_config, tokenizer)

    train_dataset = split_dataset_by_node(
        train_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    return DataLoader(
        train_dataset,
        batch_size=train_config["train_batch_size"],
        num_workers=train_config["train_num_workers_4_dataloader"],
        prefetch_factor=train_config.get("prefetch_factor", 2),
        pin_memory=True,
    )


def build_model(config, tokenizer):
    """Instantiate the model, optionally resuming from a Hugging Face checkpoint.

    Note:
        Under ZeRO-3, parameter partitioning only takes effect for models built
        inside ``deepspeed.zero.Init()``. The ``Auto*`` classes enter that context
        for you; the concrete class used here does not, so every rank may
        materialise the full weights. Register this architecture with
        ``AutoModelForCausalLM`` before relying on ZeRO-3 for memory savings.
        See huggingface/accelerate#932.
    """
    model_config = vocab_32k_gpt2Config.from_pretrained(FLAGS.model_config)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_id

    ckpt = config["train"].get("ckpt")
    if ckpt is not None:
        model = vocab_32k_GPT2LMHeadModel.from_pretrained(ckpt, config=model_config)
        logger.info("Loaded checkpoint from: %s", ckpt)
    else:
        model = vocab_32k_GPT2LMHeadModel(config=model_config)
        logger.info("Initialised model from scratch.")

    logger.info(
        "Model parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6
    )
    return model


def apply_lora(model):
    """Wrap the model with LoRA adapters.

    ``enable_input_require_grads`` keeps gradient checkpointing compatible with
    frozen embeddings, see huggingface/transformers#23170.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, module_input, module_output):
            module_output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=1,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main(argv):
    del argv  # Handled by absl.

    with open(FLAGS.train_config, encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"].get(
            "gradient_accumulation_steps", 1
        )
    )

    tokenizer = build_tokenizer(config["data"])
    train_loader = build_dataloader(config, tokenizer, accelerator)
    raw_model = build_model(config, tokenizer)

    if config["train"].get("use_lora", False):
        raw_model = apply_lora(raw_model)
    if config["train"].get("gradient_checkpointing_enable", False):
        raw_model.gradient_checkpointing_enable()

    trainer = Trainer(config, raw_model, train_loader, tokenizer, accelerator)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
