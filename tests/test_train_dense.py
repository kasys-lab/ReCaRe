"""Tests for dense retriever contrastive fine-tuning helpers."""

from __future__ import annotations

import json

import pytest
import torch

from recare_baselines import dense as dense_mod
from recare_baselines import train_dense as td


def test_contrastive_logits_shape_and_labels():
    batch_size = 3
    dim = 5
    query = torch.randn(batch_size, dim)
    positive = torch.randn(batch_size, dim)
    hard_negative = torch.randn(batch_size, dim)

    logits, labels = td.contrastive_logits(
        query,
        positive,
        hard_negative,
        temperature=0.05,
    )

    assert logits.shape == (batch_size, 2 * batch_size)
    assert labels.tolist() == [0, 1, 2]


def test_contrastive_labels_point_to_positive_block():
    query = torch.eye(2)
    positive = torch.eye(2)
    hard_negative = torch.flip(torch.eye(2), dims=[0])

    logits, labels = td.contrastive_logits(
        query,
        positive,
        hard_negative,
        temperature=1.0,
    )

    assert labels.tolist() == [0, 1]
    assert logits.argmax(dim=1).tolist() == [0, 1]


def test_qrels_false_negative_mask_removes_other_known_positives():
    mask = td.qrels_false_negative_mask(
        qrel_positive_doc_ids=[("p1", "p2"), ("p1", "p2")],
        positive_doc_ids=["p1", "p2"],
        hard_negative_doc_ids=["n1", "n2"],
    )

    assert mask.tolist() == [
        [False, True, False, False],
        [True, False, False, False],
    ]


def test_contrastive_logits_apply_false_negative_mask_but_keep_label():
    query = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    hard_negative = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])

    logits, labels = td.contrastive_logits(
        query,
        positive,
        hard_negative,
        temperature=1.0,
        false_negative_mask=mask,
    )

    assert labels.tolist() == [0]
    assert logits[0, 0].item() == 1.0
    assert logits[0, 1].item() < -1e30


def test_early_stopping_tracks_bad_epochs_and_reset():
    state = td.EarlyStoppingState(patience=3)

    assert state.update(1.0) == (True, False)
    assert state.bad_epochs == 0
    assert state.update(1.1) == (False, False)
    assert state.bad_epochs == 1
    assert state.update(0.9) == (True, False)
    assert state.bad_epochs == 0
    assert state.update(0.95) == (False, False)
    assert state.update(0.96) == (False, False)
    assert state.update(0.97) == (False, True)
    assert state.bad_epochs == 3


def test_append_jsonl_appends_training_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"

    td.append_jsonl(metrics_path, {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6})
    td.append_jsonl(metrics_path, {"epoch": 2, "train_loss": 0.4, "val_loss": 0.55})

    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.55},
    ]


def test_jina_task_is_resolved_from_model_spec():
    spec = dense_mod.MODELS["jina-v3"]

    assert td._jina_task(spec, is_query=True) == "retrieval.query"
    assert td._jina_task(spec, is_query=False) == "retrieval.passage"
    assert td._jina_task(dense_mod.MODELS["bge-m3"], is_query=True) is None


def test_jina_trainable_pooling_uses_mean_pooling():
    hidden = torch.tensor(
        [
            [
                [1.0, 1.0],
                [3.0, 5.0],
                [100.0, 100.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0]])
    out = type("Output", (), {"last_hidden_state": hidden})()

    emb = td._pool_model_output(out, attention_mask, "jina_encode")

    assert emb.tolist() == [[2.0, 3.0]]


def test_jina_forward_uses_adapter_mask_for_task_lora():
    class TaskAwareModel(torch.nn.Module):
        _adaptation_map = {"retrieval.query": 3}

        def __init__(self):
            super().__init__()
            self.seen_adapter_mask = None

        def forward(self, input_ids, attention_mask, adapter_mask):
            self.seen_adapter_mask = adapter_mask
            return type(
                "Output",
                (),
                {
                    "last_hidden_state": torch.zeros(
                        input_ids.shape[0], input_ids.shape[1], 2
                    )
                },
            )()

    model = TaskAwareModel()
    inputs = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }

    td._forward_model(model, inputs, task="retrieval.query")

    assert model.seen_adapter_mask is not None
    assert model.seen_adapter_mask.dtype == torch.int32
    assert model.seen_adapter_mask.tolist() == [3, 3]


def test_jina_task_instruction_reads_remote_model_metadata():
    class JinaLikeModel(torch.nn.Module):
        _task_instructions = {
            "retrieval.query": "Represent the query for retrieving evidence documents: "
        }

    assert td._jina_task_instruction(JinaLikeModel(), "retrieval.query") == (
        "Represent the query for retrieving evidence documents: "
    )
    assert td._jina_task_instruction(JinaLikeModel(), None) == ""


def test_train_load_kwargs_use_flash_attention_for_jina_only():
    assert td.trainable_model_load_kwargs(
        dense_mod.MODELS["bge-m3"], tuning_method="lora"
    ) == {}
    assert td.trainable_model_load_kwargs(
        dense_mod.MODELS["jina-v3"], tuning_method="lora"
    ) == {
        "use_flash_attn": True
    }

    assert td.trainable_model_load_kwargs(
        dense_mod.MODELS["jina-v3"], tuning_method="full"
    ) == {
        "use_flash_attn": True,
        "lora_main_params_trainable": True,
    }


def test_disable_gradient_checkpointing_clears_remote_flags():
    class CheckpointedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = True

    class CheckpointedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = CheckpointedModule()
            self.disabled = False

        def gradient_checkpointing_disable(self):
            self.disabled = True

    model = CheckpointedModel()

    td._disable_gradient_checkpointing(model)

    assert model.disabled is True
    assert model.child.gradient_checkpointing is False


def test_gradient_checkpointing_default_is_model_specific():
    assert (
        td._resolve_gradient_checkpointing(
            dense_mod.MODELS["jina-v3"],
            tuning_method="lora",
            requested=None,
        )
        is False
    )
    assert (
        td._resolve_gradient_checkpointing(
            dense_mod.MODELS["bge-m3"],
            tuning_method="lora",
            requested=None,
        )
        is True
    )
    assert (
        td._resolve_gradient_checkpointing(
            dense_mod.MODELS["jina-v3"],
            tuning_method="lora",
            requested=True,
        )
        is True
    )


def test_lora_target_detection_prefers_transformer_linear_layers():
    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Linear(2, 2)
            self.key = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 2)
            self.dense = torch.nn.Linear(2, 2)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TinyAttention()
            self.classifier = torch.nn.Linear(2, 1)

    targets = td._default_lora_target_modules(TinyModel())

    assert targets == ["dense", "key", "query", "value"]


def test_jina_forward_requires_task_aware_model():
    class NoTaskModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return type(
                "Output",
                (),
                {
                    "last_hidden_state": torch.zeros(
                        input_ids.shape[0], input_ids.shape[1], 2
                    )
                },
            )()

    inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    with pytest.raises(RuntimeError, match="query/passage adapter distinction"):
        td._forward_model(NoTaskModel(), inputs, task="retrieval.query")


def test_peft_gradient_checkpointing_is_skipped_without_input_embeddings():
    class NoInputEmbeddingsModel(torch.nn.Module):
        def gradient_checkpointing_enable(self):
            raise AssertionError("should not be called by the support probe")

        def get_input_embeddings(self):
            raise NotImplementedError("remote model does not expose embeddings")

    assert td._supports_peft_gradient_checkpointing(NoInputEmbeddingsModel()) is False


def test_peft_gradient_checkpointing_is_supported_with_input_embeddings():
    class HasInputEmbeddingsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(2, 3)

        def gradient_checkpointing_enable(self):
            pass

        def get_input_embeddings(self):
            return self.emb

    assert td._supports_peft_gradient_checkpointing(HasInputEmbeddingsModel()) is True
