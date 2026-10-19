"""Offline CPU tests: tiny ESM-2 configs, no downloaded 650M weights or data."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import EsmConfig, EsmForMaskedLM, EsmForSequenceClassification, EsmTokenizer

from common import TCRDataset, require_label_column
from modeling_tcr import TCRForSequenceClassification, initialize_tcr_model, load_prediction_model, pool_sequence
from predict_tumori import add_sample_score
from training_utils import compute_trainer_metrics, make_trainer, validate_training_arguments


def tiny_config():
    return EsmConfig(vocab_size=33, hidden_size=12, num_hidden_layers=1,
                     num_attention_heads=3, intermediate_size=24,
                     max_position_embeddings=64, pad_token_id=1, mask_token_id=32,
                     token_dropout=False, hidden_dropout_prob=0.0,
                     attention_probs_dropout_prob=0.0, num_labels=2)


def settings():
    return SimpleNamespace(epochs=1, batch_size=2, gradient_accumulation_steps=2,
                           learning_rate=1e-4, head_learning_rate=1e-3,
                           weight_decay=0.01, fp16=False, seed=42, threshold=0.5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_pooling_excludes_special_and_padding_tokens(self):
        hidden = torch.tensor([[[99., 99.], [2., 4.], [4., 8.], [88., 88.], [999., 999.]]])
        ids = torch.tensor([[0, 5, 6, 2, 1]])
        pooled = pool_sequence(hidden, ids, torch.tensor([[1, 1, 1, 1, 0]]), tiny_config())
        torch.testing.assert_close(pooled, torch.tensor([[99., 99., 3., 6., 4., 8.]]))
        empty = pool_sequence(hidden, torch.tensor([[0, 2, 1, 1, 1]]),
                              torch.tensor([[1, 1, 0, 0, 0]]), tiny_config())
        self.assertTrue(torch.isfinite(empty).all())
        torch.testing.assert_close(empty[:, 2:], torch.zeros(1, 4))

    def test_shared_classifier_three_branches_and_smoothed_loss(self):
        config = tiny_config();config.tcr_dropout_rates = [0.1, 0.2, 0.3]
        model = TCRForSequenceClassification(config).train()
        calls = []
        handle = model.classifier.register_forward_hook(lambda module, inputs, output: calls.append(output))
        labels = torch.tensor([0, 1])
        result = model(input_ids=torch.tensor([[0, 5, 6, 2], [0, 7, 8, 2]]), labels=labels)
        handle.remove()
        self.assertEqual(len(calls), 3)
        torch.testing.assert_close(result.logits, torch.stack(calls).mean(0))
        torch.testing.assert_close(result.loss, F.cross_entropy(result.logits, labels, label_smoothing=0.1))
        result.loss.backward()
        self.assertIsNotNone(model.projection[0].weight.grad)
        self.assertIsNotNone(model.esm.embeddings.word_embeddings.weight.grad)

    def test_checkpoint_roundtrip_and_deterministic_eval(self):
        model = TCRForSequenceClassification(tiny_config()).eval()
        ids = torch.tensor([[0, 5, 6, 2]])
        with torch.no_grad():expected = model(input_ids=ids).logits
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored = load_prediction_model(folder).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(input_ids=ids).logits, expected)
                torch.testing.assert_close(restored(input_ids=ids).logits, expected)

    def test_pretrained_encoder_is_loaded_and_finetuned_base_rejected(self):
        base = EsmForMaskedLM(tiny_config())
        with tempfile.TemporaryDirectory() as folder:
            base.save_pretrained(folder)
            first = initialize_tcr_model(folder, [0.2] * 3, 0.1)
            second = initialize_tcr_model(folder, [0.2] * 3, 0.1)
            torch.testing.assert_close(first.esm.embeddings.word_embeddings.weight,
                                       base.esm.embeddings.word_embeddings.weight)
            self.assertFalse(torch.equal(first.classifier.weight, second.classifier.weight))
            fine = Path(folder) / "fine";first.save_pretrained(fine)
            with self.assertRaisesRegex(ValueError, "original pretrained"):
                initialize_tcr_model(fine, [0.2] * 3, 0.1)

    def test_legacy_checkpoint_still_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            EsmForSequenceClassification(tiny_config()).save_pretrained(folder)
            self.assertIsInstance(load_prediction_model(folder), EsmForSequenceClassification)

    def test_optimizer_parameter_coverage_and_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            vocab = Path(folder) / "vocab.txt"
            vocab.write_text("\n".join(['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']), encoding="utf-8")
            tokenizer = EsmTokenizer(vocab_file=str(vocab))
            dataset = TCRDataset(["AG", "VC", "AS", "GE"], tokenizer, [0, 1, 0, 1], max_length=12)
            model = TCRForSequenceClassification(tiny_config())
            trainer = make_trainer(model, Path(folder), dataset, dataset, settings())
            trainer.args.use_cpu = True
            optimizer = trainer.create_optimizer()
            ids = [id(p) for g in optimizer.param_groups for p in g['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    encoder = any(parameter is p for p in model.esm.parameters())
                    self.assertEqual(group['lr'], 1e-4 if encoder else 1e-3)
            trainer.train()
            self.assertIsNotNone(trainer.state.best_model_checkpoint)
            self.assertIsNotNone(trainer.state.best_metric)
            out = Path(folder) / "final";trainer.save_model(out)
            self.assertIsInstance(load_prediction_model(out), TCRForSequenceClassification)

    def test_top_five_percent_scale_and_stable_ties(self):
        df = pd.DataFrame({'reads_A': [1] * 21, 'reads_B': [0] * 21,
                           'p': [0.9, 0.7] + [0.1] * 19})
        scored, result = add_sample_score(df, 'p')
        self.assertEqual(result['top_k'], 2)
        expected = (0.9 + 0.7) / 21
        self.assertAlmostEqual(result['weighted_score'], expected)
        self.assertAlmostEqual(result['normalized_score'], 1 - np.exp(-31.92 * expected))
        self.assertAlmostEqual(scored['Frequency'].sum(), 1)

    def test_selection_uses_requested_threshold(self):
        logits = np.log(np.array([[0.4, 0.6], [0.1, 0.9]]))
        pred = SimpleNamespace(predictions=logits, label_ids=np.array([0, 1]))
        self.assertEqual(compute_trainer_metrics(pred, threshold=0.8)['f1'], 1.0)
        with self.assertRaises(ValueError):require_label_column(pd.DataFrame({'Label': [0, 0.5]}))


if __name__ == '__main__':
    unittest.main()
