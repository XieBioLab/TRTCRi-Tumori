"""ESM-2 with the pooled, projected, multi-sample dropout manuscript head."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, EsmConfig, EsmModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import EsmPreTrainedModel


HEAD_TYPE = "cls_mean_max_multisample_v1"


def pool_sequence(hidden, input_ids, attention_mask, config):
    """Concatenate CLS, amino-acid mean and max; exclude padding and boundaries."""
    mask = attention_mask.bool()
    cls_id = getattr(config, "cls_token_id", None)
    eos_id = getattr(config, "eos_token_id", None)
    for token_id in (config.pad_token_id, 0 if cls_id is None else cls_id,
                     2 if eos_id is None else eos_id):
        if token_id is not None:
            mask = mask & input_ids.ne(token_id)
    weights = mask.unsqueeze(-1)
    # Accumulate the mean in float32 to avoid reduced-precision overflow.
    mean = (hidden.float() * weights).sum(dim=1)
    mean = (mean / weights.sum(dim=1).clamp_min(1)).to(hidden.dtype)
    maximum = hidden.masked_fill(~weights, torch.finfo(hidden.dtype).min).max(dim=1).values
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((hidden[:, 0], mean, maximum), dim=-1)


class TCRForSequenceClassification(EsmPreTrainedModel):
    """Three independent dropout masks share ONE classifier; average their logits."""

    config_class = EsmConfig

    def __init__(self, config):
        config.tcr_head_type = HEAD_TYPE
        config.tcr_dropout_rates = list(getattr(config, "tcr_dropout_rates", [0.2] * 3))
        config.tcr_label_smoothing = float(getattr(config, "tcr_label_smoothing", 0.1))
        if len(config.tcr_dropout_rates) != 3 or any(not 0 <= p < 1 for p in config.tcr_dropout_rates):
            raise ValueError("Exactly three dropout rates in [0, 1) are required.")
        if not 0 <= config.tcr_label_smoothing < 1:
            raise ValueError("Label smoothing must lie in [0, 1).")
        if config.num_labels != 2:
            raise ValueError("TRTCRi/Tumori require two labels.")
        super().__init__(config)
        self.esm = EsmModel(config, add_pooling_layer=False)
        self.projection = nn.Sequential(
            nn.Linear(3 * config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.dropouts = nn.ModuleList(nn.Dropout(p) for p in config.tcr_dropout_rates)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        if input_ids is None:
            raise ValueError("input_ids are required for special-token masking.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()
        use_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states, return_dict=True)
        pooled = pool_sequence(outputs.last_hidden_state, input_ids, attention_mask, self.config)
        feature = self.projection(pooled)
        logits = torch.stack([self.classifier(dropout(feature)) for dropout in self.dropouts]).mean(dim=0)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels.long(),
                                   label_smoothing=self.config.tcr_label_smoothing)
        if not use_dict:
            result = (logits,) + tuple(x for x in (outputs.hidden_states, outputs.attentions) if x is not None)
            return (loss,) + result if loss is not None else result
        return SequenceClassifierOutput(loss=loss, logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


def initialize_tcr_model(base_model, dropout_rates, label_smoothing):
    """Always initialize a new head from the ORIGINAL ESM checkpoint, per run/fold."""
    config = EsmConfig.from_pretrained(base_model)
    if getattr(config, "tcr_head_type", None) or any(
            "SequenceClassification" in name for name in (config.architectures or [])):
        raise ValueError("--base_model must be an original pretrained ESM-2 encoder/MLM, not a TRTCRi/Tumori checkpoint.")
    config.num_labels = 2
    config.problem_type = "single_label_classification"
    config.tcr_dropout_rates = list(dropout_rates)
    config.tcr_label_smoothing = label_smoothing
    config.tcr_base_model = str(base_model)
    model, info = TCRForSequenceClassification.from_pretrained(
        base_model, config=config, output_loading_info=True)
    missing_encoder = [key for key in info["missing_keys"] if key.startswith("esm.")]
    if missing_encoder:
        raise ValueError(f"Pretrained encoder weights are missing: {missing_encoder[:5]}")
    return model


def load_prediction_model(model_dir):
    """Load our full head, or retain support for legacy default-ESM classifiers."""
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    head_type = getattr(config, "tcr_head_type", None)
    if head_type is None:
        return AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    if head_type != HEAD_TYPE:
        raise ValueError(f"Unsupported TCR classification head: {head_type}")
    model, info = TCRForSequenceClassification.from_pretrained(
        model_dir, config=config, local_files_only=True, output_loading_info=True)
    if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete or incompatible TCR checkpoint: {info}")
    return model
