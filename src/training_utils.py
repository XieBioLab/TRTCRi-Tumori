"""Shared training configuration; save all effective settings with every run."""

import inspect
from functools import partial

import torch
from transformers import Trainer, TrainingArguments

from common import compute_binary_metrics, save_metrics_json
from modeling_tcr import initialize_tcr_model


class DifferentialLRTrainer(Trainer):
    def __init__(self, *args, head_learning_rate, **kwargs):
        self.head_learning_rate = head_learning_rate
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            groups = {}
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                encoder = name.startswith("esm.")
                decay = parameter.ndim > 1  # Exclude biases and LayerNorm scales.
                key = (encoder, decay)
                if key not in groups:
                    groups[key] = {"params": [],
                                   "lr": self.args.learning_rate if encoder else self.head_learning_rate,
                                   "weight_decay": self.args.weight_decay if decay else 0.0}
                groups[key]["params"].append(parameter)
            self.optimizer = torch.optim.AdamW(
                list(groups.values()), betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon)
        return self.optimizer


def add_head_arguments(parser):
    parser.add_argument("--head_learning_rate", type=float, default=None,
                        help="Head LR; default is 10 times --learning_rate (encoder LR).")
    parser.add_argument("--dropout_rates", type=float, nargs=3, default=None,
                        help="Three branch probabilities; default repeats --classifier_dropout.")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Retained trainer checkpoints per fold; best checkpoint is preserved.")


def validate_training_arguments(args):
    args.head_learning_rate = args.head_learning_rate if args.head_learning_rate is not None else 10 * args.learning_rate
    args.dropout_rates = args.dropout_rates if args.dropout_rates is not None else [args.classifier_dropout] * 3
    if min(args.learning_rate, args.head_learning_rate) <= 0:
        raise ValueError("Encoder and head learning rates must be positive.")
    if any(not 0 <= p < 1 for p in args.dropout_rates) or not 0 <= args.label_smoothing < 1:
        raise ValueError("Dropout and label smoothing must lie in [0, 1).")
    if not 0 < args.threshold < 1:
        raise ValueError("The classification threshold must lie in (0, 1).")
    if getattr(args, "save_total_limit", 2) < 1:
        raise ValueError("At least one trainer checkpoint must be retained.")


def build_training_args(output_dir, args):
    kwargs = {
        "output_dir": str(output_dir / "trainer_checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": "epoch", "logging_steps": 100,
        "save_total_limit": getattr(args, "save_total_limit", 2),
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "load_best_model_at_end": True, "metric_for_best_model": "eval_f1",
        "greater_is_better": True, "report_to": "none", "lr_scheduler_type": "cosine",
        "fp16": torch.cuda.is_available() and args.fp16,
        "seed": args.seed, "data_seed": args.seed,
        # Smoothing is computed in the model after averaging branch logits.
        "label_smoothing_factor": 0.0,
    }
    keyword = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    kwargs[keyword] = "epoch"
    return TrainingArguments(**kwargs)


def compute_trainer_metrics(pred, threshold=0.5):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()[:, 1]
    metrics = compute_binary_metrics(pred.label_ids, probabilities, threshold=threshold)
    return {name: metrics[name] for name in ("accuracy", "f1", "roc_auc", "average_precision")}


def make_trainer(model, output_dir, train_dataset, val_dataset, args):
    return DifferentialLRTrainer(
        model=model, args=build_training_args(output_dir, args),
        head_learning_rate=args.head_learning_rate,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=partial(compute_trainer_metrics, threshold=args.threshold))


def new_model(args):
    return initialize_tcr_model(args.base_model, args.dropout_rates, args.label_smoothing)


def save_run_config(args, output_dir, labels, split_method):
    import transformers
    values = dict(vars(args))
    values.update({"architecture": "CLS + masked mean + masked max -> Linear/GELU/LayerNorm -> 3 shared-classifier dropout branches",
                   "checkpoint_selection": "highest validation F1",
                   "initialization": "original pretrained ESM-2; independent head per run/fold",
                   "split_method": split_method, "n_records": len(labels),
                   "n_positive": sum(labels), "n_negative": len(labels) - sum(labels),
                   "torch_version": torch.__version__, "transformers_version": transformers.__version__})
    save_metrics_json(values, output_dir / "run_config.json")
