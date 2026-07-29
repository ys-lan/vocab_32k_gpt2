"""Training loop with distributed checkpointing, logging and periodic sampling."""

import logging
import os
import time

import torch
import wandb
from deepspeed.ops.adam import FusedAdam
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup

from dataset.validation import val_set_pretrain

logger = logging.getLogger(__name__)

# Default to offline logging so that training never blocks on network access.
# Export WANDB_MODE=online (and WANDB_API_KEY) to stream metrics to the cloud.
os.environ.setdefault("WANDB_MODE", "offline")


class Trainer:
    """Drives a single training run on top of an :class:`~accelerate.Accelerator`.

    Args:
        config: Parsed training config (see ``configs/*.yaml``).
        raw_model: Unwrapped model, prior to ``accelerator.prepare``.
        train_loader: Dataloader yielding ``input_ids``/``labels``/``attention_mask``.
        tokenizer: Tokenizer used for decoding validation samples.
        accelerator: Configured accelerator handling distribution and precision.
    """

    def __init__(self, config, raw_model, train_loader, tokenizer, accelerator):
        self.config = config
        self.raw_model = raw_model
        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.train_and_eval = config["train"].get("train_and_eval", False)
        self.gradient_accumulation_steps = config["train"].get(
            "gradient_accumulation_steps", 1
        )
        # Schedulers are stepped once per process per micro-batch, so warmup and
        # total step counts have to be rescaled to stay in "global step" units.
        self.lr_scheduler_factor = (
            accelerator.num_processes / accelerator.gradient_accumulation_steps
        )
        # Intervals in the config are expressed in optimizer steps; convert them
        # to micro-batch counts, which is what the training loop counts.
        self.log_interval = (
            self.config["log_interval"] * accelerator.gradient_accumulation_steps
        )
        self.eval_interval = (
            self.config["eval_interval"] * accelerator.gradient_accumulation_steps
        )
        self.save_interval = (
            self.config["save_interval"] * accelerator.gradient_accumulation_steps
        )
        self.work_dir = self.config["work_dir"]

        if accelerator.is_main_process:
            wandb.init(project=self.config["project_name"])

    def get_model_info(self):
        """Print a layer-by-layer summary of the model (debugging helper)."""
        with torch.no_grad():
            summary(
                self.raw_model.cuda(),
                input_data=torch.ones(1, 64, dtype=torch.int64).cuda(),
            )

    def get_optimizer(self):
        """Build a FusedAdam optimizer, excluding biases and norms from weight decay."""
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        if self.config["train"].get("use_lora", False):
            optimizer_grouped_parameters = self.raw_model.parameters()
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.config["train"]["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in self.raw_model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        self.optim = FusedAdam(
            optimizer_grouped_parameters,
            lr=self.config["train"]["lr"],
            betas=(0.9, 0.95),
        )

    def get_lr_scheduler(self):
        """Build a cosine schedule with linear warmup."""
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optim,
            num_warmup_steps=self.config["train"]["num_warmup_steps"]
            * self.lr_scheduler_factor,
            num_training_steps=self.config["train"]["num_training_steps"]
            * self.lr_scheduler_factor,
        )

    def prepare(self):
        """Wrap objects for distributed training and resume from ``work_dir`` if possible."""
        (
            _,
            self.model,
            self.optim,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_loader, self.raw_model, self.optim, self.scheduler
        )
        self.optim.zero_grad()
        self.global_step = 0
        try:
            self.accelerator.load_state(self.work_dir)
            self.global_step = self.scheduler.scheduler._step_count - 1
            self.global_step = self.global_step // self.accelerator.num_processes
            logger.info("Restored training state from %s", self.work_dir)
        except Exception:  # noqa: BLE001 - absence of a checkpoint is not fatal.
            logger.info("No checkpoint found in %s, starting from scratch.", self.work_dir)
        if self.global_step > 0:
            skip_steps = self.global_step * self.gradient_accumulation_steps
            logger.info("Fast-forwarding the dataloader by %d batches.", skip_steps)
            self.train_loader_skiped = self.accelerator.skip_first_batches(
                self.train_loader, num_batches=skip_steps
            )
        else:
            self.train_loader_skiped = self.train_loader
        self.accelerator.wait_for_everyone()

    def train_step(self, batch):
        out = self.model(**batch)
        total_loss = out.loss
        losses = {"total_loss": total_loss}
        self.accelerator.backward(total_loss)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad()
        return losses

    def train(self):
        self.get_optimizer()
        self.get_lr_scheduler()
        self.prepare()
        self.start_time = time.time()
        self.epoch = 0
        self.data_step = 0

        while True:
            if self.data_step >= self.config["train"]["num_training_steps"]:
                break
            # Only the resumed epoch needs to skip already-consumed batches.
            if self.epoch == 0:
                train_loader = self.train_loader_skiped
            else:
                train_loader = self.train_loader

            for batch in train_loader:
                if self.data_step >= self.config["train"]["num_training_steps"]:
                    break
                for k, v in batch.items():
                    batch[k] = v.to(self.accelerator.device, non_blocking=True)
                self.model.train()
                with self.accelerator.accumulate(self.model):
                    losses = self.train_step(batch)
                    if self.accelerator.sync_gradients:
                        self.global_step += 1

                if (
                    self.data_step % self.log_interval == 0
                    and self.data_step > 0
                    and self.accelerator.is_main_process
                ):
                    self.log(losses)
                if (
                    self.data_step % self.eval_interval == 0
                    and self.accelerator.is_main_process
                    and self.train_and_eval
                ):
                    self.eval()

                # All ranks must reach the barrier before a collective save.
                self.accelerator.wait_for_everyone()
                if self.data_step % self.save_interval == 0 and self.data_step > 0:
                    self.accelerator.save_state(
                        os.path.join(self.work_dir, f"checkpoint_epoch{self.epoch}")
                    )
                self.data_step += 1
            self.epoch += 1
        wandb.finish()

    def log(self, losses):
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        tokens = (
            self.config["train"]["train_batch_size"]
            * self.log_interval
            * self.config["data"]["seq_length"]
        )
        wandb.log({"Training/Token per second per gpu": tokens / cost_time})
        for k, v in losses.items():
            wandb.log({f"Losses/{k}": v})
        current_lr = self.optim.param_groups[0]["lr"]
        wandb.log({"Training/LR": current_lr})
        if self.optim.scaler is not None:
            wandb.log({"Training/Loss Scale": self.optim.scaler.get_scale()})
        wandb.log({"Training/Data Step": self.data_step})
        wandb.log({"Training/Global Step": self.global_step})
        wandb.log({"Training/Epoch": self.epoch})
        self.accelerator.print(
            f"Epoch: {self.epoch}, Global Step: {self.global_step}, "
            f"Data Step: {self.data_step}, LR: {current_lr:.3e}, "
            f"Loss: {losses['total_loss']:.4f}, "
            f"Tokens/s/GPU: {tokens / cost_time:.0f}"
        )

    def eval(self):
        """Sample completions for the validation prompts and log them to W&B."""
        text_table = wandb.Table(columns=["question", "pred"])
        self.model.eval()
        with torch.no_grad():
            for raw_inputs in val_set_pretrain:
                inputs = self.tokenizer(
                    raw_inputs,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                input_length = inputs["input_ids"].shape[1]
                for k, v in inputs.items():
                    inputs[k] = v.to(self.accelerator.device)
                self.accelerator.wait_for_everyone()
                pred = self.model.generate(
                    **inputs, max_new_tokens=256, do_sample=True, repetition_penalty=2.0
                )
                self.accelerator.wait_for_everyone()
                pred = pred[0, input_length:]
                pred = self.tokenizer.decode(pred.cpu(), skip_special_tokens=True)
                text_table.add_data(raw_inputs, pred)
                print(raw_inputs, "\n", pred, "\n")
        wandb.log({f"Predictions on {self.global_step}": text_table})
