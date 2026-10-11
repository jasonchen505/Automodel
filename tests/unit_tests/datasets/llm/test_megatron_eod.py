# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Megatron document-boundary features (upstream issue #4157).

Covers ``_get_ltor_masks_and_position_ids`` with a real EOD token id and the
config threading of ``reset_position_ids`` / ``reset_attention_mask`` /
``eod_mask_loss`` through ``MegatronPretrainingConfig`` and
``MegatronPretraining``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_automodel.components.datasets.llm import megatron_dataset
from nemo_automodel.components.datasets.llm.megatron.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    _get_ltor_masks_and_position_ids,
)
from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretraining, MegatronPretrainingConfig

EOD = 99


def _tokens():
    # Two documents: [5, EOD] and [7, 8, EOD], then a partial third: [9].
    return torch.tensor([5, EOD, 7, 8, EOD, 9])


class TestGetLtorMasksAndPositionIds:
    """Behavior of document-boundary features with a real EOD token id."""

    def test_eod_mask_loss_zeros_loss_at_eod_positions(self):
        _, loss_mask, _ = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=True,
            create_attention_mask=False,
        )
        assert loss_mask.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    def test_reset_position_ids_restarts_after_each_eod(self):
        _, _, position_ids = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=True,
            reset_attention_mask=False,
            eod_mask_loss=False,
            create_attention_mask=False,
        )
        assert position_ids.tolist() == [0, 1, 0, 1, 2, 0]

    def test_reset_attention_mask_blocks_cross_document_attention(self):
        attention_mask, _, _ = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=False,
            reset_attention_mask=True,
            eod_mask_loss=False,
            create_attention_mask=True,
        )
        # Bool mask, True = masked. Tokens after the first EOD (rows 2+)
        # must not attend to the first document (cols 0-1).
        assert attention_mask.shape == (1, 6, 6)
        assert attention_mask[0, 2, 0].item() is True
        assert attention_mask[0, 2, 1].item() is True
        # Within-document attention is untouched.
        assert attention_mask[0, 1, 0].item() is False
        assert attention_mask[0, 3, 2].item() is False

    def test_all_flags_off_matches_legacy_behavior(self):
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            create_attention_mask=True,
        )
        assert loss_mask.tolist() == [1.0] * 6
        assert position_ids.tolist() == [0, 1, 2, 3, 4, 5]
        # Plain causal mask: lower triangle (incl. diagonal) unmasked.
        assert attention_mask[0, 3, 2].item() is False
        assert attention_mask[0, 2, 3].item() is True

    def test_unmatched_eod_token_is_noop(self):
        # The historical -10000 sentinel never occurs: features stay no-ops.
        _, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            _tokens(),
            -10000,
            reset_position_ids=True,
            reset_attention_mask=False,
            eod_mask_loss=True,
            create_attention_mask=False,
        )
        assert loss_mask.tolist() == [1.0] * 6
        assert position_ids.tolist() == [0, 1, 2, 3, 4, 5]


class TestEodConfigThreading:
    """The three flags travel from MegatronPretrainingConfig to GPTDatasetConfig."""

    def test_config_defaults_are_backward_compatible(self):
        config = MegatronPretrainingConfig(paths="dummy")
        assert config.reset_position_ids is False
        assert config.reset_attention_mask is False
        assert config.eod_mask_loss is False

    def test_init_stores_eod_flags(self, monkeypatch, tmp_path):
        monkeypatch.setattr(megatron_dataset, "compile_helper", lambda: None)
        monkeypatch.setattr(megatron_dataset, "get_blend_from_list", lambda p: (["dummy"], None))
        monkeypatch.setattr(megatron_dataset, "validate_dataset_asset_accessibility", lambda *a, **k: None)
        mp = MegatronPretraining(
            paths=[str(tmp_path)],
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
        )
        assert mp.reset_position_ids is True
        assert mp.reset_attention_mask is True
        assert mp.eod_mask_loss is True

    def test_gpt_dataset_config_forwards_eod_flags(self, monkeypatch):
        captured = {}

        class FakeGPTDatasetConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(megatron_dataset, "GPTDatasetConfig", FakeGPTDatasetConfig)
        mp = MegatronPretraining.__new__(MegatronPretraining)
        mp.seed = 1234
        mp.seq_length = 2048
        mp.tokenizer = SimpleNamespace(name_or_path="dummy")
        mp.index_mapping_dir = None
        mp.create_attention_mask = False
        mp.reset_position_ids = True
        mp.reset_attention_mask = True
        mp.eod_mask_loss = True
        mp.num_dataset_builder_threads = 1
        mp.object_storage_config = None
        mp.build_kwargs = {}

        mp.gpt_dataset_config

        assert captured["reset_position_ids"] is True
        assert captured["reset_attention_mask"] is True
        assert captured["eod_mask_loss"] is True


class TestGetItemDocumentBoundaryOutput:
    """__getitem__ must surface the newly enabled features in the sample dict."""

    def _dataset(self, **flags):
        ds = GPTDataset.__new__(GPTDataset)
        text = np.array([5, EOD, 7, 8, EOD, 9], dtype=np.int64)
        ds._query_document_sample_shuffle_indices = lambda idx: (text, None)  # noqa: E731
        ds.config = SimpleNamespace(
            add_extra_token_to_sequence=True,
            reset_position_ids=flags.get("reset_position_ids", False),
            reset_attention_mask=flags.get("reset_attention_mask", False),
            eod_mask_loss=flags.get("eod_mask_loss", False),
            create_attention_mask=flags.get("create_attention_mask", False),
        )
        ds._pad_token_id = 0
        ds._eos_token_id = EOD
        ds._eod_token_id = EOD
        ds.masks_and_position_ids_are_cacheable = False
        ds.masks_and_position_ids_are_cached = False
        return ds

    def test_position_ids_returned_when_reset_enabled(self):
        sample = self._dataset(reset_position_ids=True)[0]
        assert "position_ids" in sample
        # tokens are [5, EOD, 7, 8, EOD]: positions restart after each EOD.
        assert sample["position_ids"].tolist() == [0, 1, 0, 1, 2]

    def test_position_ids_absent_when_reset_disabled(self):
        sample = self._dataset()[0]
        assert "position_ids" not in sample

    def test_eod_mask_loss_encodes_ignore_index_in_labels(self):
        sample = self._dataset(eod_mask_loss=True)[0]
        assert sample["loss_mask"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]
        # The helper masks loss_mask at input-EOD positions (1, 4), matching
        # Megatron semantics. Those label positions must not contribute to the
        # CE loss or to _count_label_tokens normalization, which both key off
        # -100.
        assert sample["labels"].tolist() == [EOD, -100, 8, EOD, -100]

    def test_labels_untouched_when_eod_mask_loss_disabled(self):
        sample = self._dataset()[0]
        assert sample["labels"].tolist() == [EOD, 7, 8, EOD, 9]

    def test_reset_attention_mask_requires_create_attention_mask(self):
        with pytest.raises(ValueError, match="create_attention_mask"):
            GPTDatasetConfig(
                random_seed=0,
                sequence_length=8,
                tokenizer=SimpleNamespace(eos_token_id=EOD, pad_token_id=0),
                reset_position_ids=False,
                reset_attention_mask=True,
                eod_mask_loss=False,
                create_attention_mask=False,
            )
