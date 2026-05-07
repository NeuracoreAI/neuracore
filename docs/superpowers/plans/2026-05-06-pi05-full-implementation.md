<!-- cspell:disable -->

# pi05_full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement subtask generation and Knowledge Insulation training for Pi0.5 in Neuracore as `pi05_full`, a sibling algorithm to `pi05`.

**Architecture:** Three-loss training (flow MSE + subtask CE + FAST CE) with attention-level Knowledge Insulation; two-stage curriculum reachable via existing config knobs; new `DataType.SUBTASK_LANGUAGE` registered against the existing `LanguageData` / `BatchedLanguageData` classes.

**Tech Stack:** Python 3.10+, PyTorch, transformers (PaliGemma + Gemma), `physical-intelligence/fast` HuggingFace tokenizer, pytest.

**Spec:** `docs/superpowers/specs/2026-05-06-pi05-full-design.md`

**Branch:** `feat/add_pi05_full` (already created)

---

## Plan-wide notes

### Naming convention

- Folder: `neuracore/ml/algorithms/pi05_full/`
- Top-level model class: **`Pi05Full`** (in `pi05.py`) — renamed from `Pi05` to disambiguate from the sibling algorithm
- Inner nn.Module: **`PI05FullPolicy`** (in `modules.py`)
- Config dataclass: **`PI05FullConfig`** (in `utils.py`)

### Spec deviation: no new neuracore_types classes

The spec's file map listed `subtask_language_data.py` and `batched_subtask_language_data.py` as new files. Implementation will instead reuse the existing `LanguageData` and `BatchedLanguageData` classes by registering `DataType.SUBTASK_LANGUAGE` in the existing dispatch dicts. This matches the pattern used by other multi-mapped DataTypes (`JointData` serves five DataTypes). Net change: no new files in `neuracore_types/nc_data/`, only edits to `__init__.py`.

### Commit message convention

This branch's first commit was `feat: first draft`. Subsequent commits use conventional commit prefixes: `feat:`, `test:`, `refactor:`, `docs:`, `fix:`. The repo memory says **no Co-Authored-By Claude lines** — commits use the user's signature only.

### Pre-commit hooks

Each commit fires pyupgrade, isort, black, ruff, pydocstyle, mypy, cspell. Code must conform. Add new technical terms to `neuracore-dictionary.txt` if cspell rejects them.

### Testing baseline

- Tests live under `tests/unit/ml/algorithms/pi05_full/`.
- Running a single test: `pytest tests/unit/ml/algorithms/pi05_full/test_<name>.py -v`
- Running the full pi05_full suite: `pytest tests/unit/ml/algorithms/pi05_full/ -v`
- The `tiny_pi05_full_config` fixture (Task 16) keeps tests fast by using the smallest gemma variant.

---

## Phase A — `DataType.SUBTASK_LANGUAGE` (1 task)

### Task 1: Register DataType.SUBTASK_LANGUAGE in neuracore_types

**Files:**
- Modify: `/home/kewang/Documents/neuracore_types/neuracore_types/nc_data/__init__.py`
- Test: `/home/kewang/Documents/neuracore_types/tests/unit/nc_data/test_subtask_language_datatype.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/nc_data/test_subtask_language_datatype.py
"""Tests for the SUBTASK_LANGUAGE data type registration."""

from neuracore_types import DataType
from neuracore_types.nc_data import (
    DATA_TYPE_CONTENT_MAPPING,
    DATA_TYPE_TO_NC_DATA_CLASS,
    DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS,
)
from neuracore_types.nc_data.language_data import (
    LanguageData,
    LanguageDataImportConfig,
)


def test_subtask_language_datatype_exists():
    assert DataType.SUBTASK_LANGUAGE.value == "SUBTASK_LANGUAGE"


def test_subtask_language_maps_to_language_data_class():
    assert DATA_TYPE_TO_NC_DATA_CLASS[DataType.SUBTASK_LANGUAGE] is LanguageData


def test_subtask_language_maps_to_language_import_config():
    assert (
        DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS[DataType.SUBTASK_LANGUAGE]
        is LanguageDataImportConfig
    )


def test_subtask_language_content_mapping_is_json():
    assert DATA_TYPE_CONTENT_MAPPING[DataType.SUBTASK_LANGUAGE] == "JSON"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/kewang/Documents/neuracore_types
pytest tests/unit/nc_data/test_subtask_language_datatype.py -v
```

Expected: FAIL with `AttributeError: SUBTASK_LANGUAGE`.

- [ ] **Step 3: Add the enum value and registry entries**

Edit `neuracore_types/nc_data/__init__.py`:

In the `DataType` enum (currently ends at `CUSTOM_1D = "CUSTOM_1D"`), add right after `LANGUAGE`:

```python
    LANGUAGE = "LANGUAGE"
    SUBTASK_LANGUAGE = "SUBTASK_LANGUAGE"
    CUSTOM_1D = "CUSTOM_1D"
```

In `DATA_TYPE_TO_NC_DATA_CLASS`, add an entry right after `DataType.LANGUAGE: LanguageData,`:

```python
    DataType.LANGUAGE: LanguageData,
    DataType.SUBTASK_LANGUAGE: LanguageData,
    DataType.CUSTOM_1D: Custom1DData,
```

In `DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS`, the same way:

```python
    DataType.LANGUAGE: LanguageDataImportConfig,
    DataType.SUBTASK_LANGUAGE: LanguageDataImportConfig,
    DataType.CUSTOM_1D: Custom1DDataImportConfig,
```

In `DATA_TYPE_CONTENT_MAPPING`, after `DataType.LANGUAGE: ...`:

```python
    DataType.LANGUAGE: "JSON",
    DataType.SUBTASK_LANGUAGE: "JSON",
```

(Verify the existing `LANGUAGE` content mapping value first; if it differs from "JSON", match it. As of writing it is "JSON".)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/nc_data/test_subtask_language_datatype.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run the broader neuracore_types test suite to confirm nothing else broke**

```bash
pytest tests/unit/ -v
```

Expected: previously-passing tests still pass.

- [ ] **Step 6: Commit (in neuracore_types repo)**

```bash
cd /home/kewang/Documents/neuracore_types
git add neuracore_types/nc_data/__init__.py tests/unit/nc_data/test_subtask_language_datatype.py
git commit -m "feat: add SUBTASK_LANGUAGE data type

Registers DataType.SUBTASK_LANGUAGE for the pi05_full algorithm,
reusing the existing LanguageData / LanguageDataImportConfig plumbing."
```

---

## Phase B — pi05_full config and FAST tokenizer (2 tasks)

### Task 2: Extend `PI05FullConfig` with new fields

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/utils.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_config.py`

- [ ] **Step 1: Create the test directory and write a failing test**

```bash
mkdir -p tests/unit/ml/algorithms/pi05_full
touch tests/unit/ml/algorithms/pi05_full/__init__.py
```

```python
# tests/unit/ml/algorithms/pi05_full/test_config.py
"""Tests for PI05FullConfig new fields."""

import pytest

from neuracore.ml.algorithms.pi05_full.utils import PI05FullConfig


def test_default_loss_weights_match_jax_paper():
    cfg = PI05FullConfig()
    assert cfg.subtask_loss_weight == 10.0
    assert cfg.fast_token_loss_weight == 1.0
    assert cfg.flow_matching_loss_weight == 1.0


def test_knowledge_insulation_default_true():
    cfg = PI05FullConfig()
    assert cfg.knowledge_insulation is True


def test_subtask_token_lengths_have_defaults():
    cfg = PI05FullConfig()
    assert cfg.max_subtask_tokens == 64
    assert cfg.max_fast_tokens == 128


def test_fast_tokenizer_defaults():
    cfg = PI05FullConfig()
    assert cfg.fast_tokenizer_name == "physical-intelligence/fast"
    assert cfg.fast_skip_tokens == 2048


def test_subtask_generation_defaults():
    cfg = PI05FullConfig()
    assert cfg.max_decoding_steps == 200
    assert cfg.subtask_temperature == 0.0


def test_negative_loss_weights_rejected():
    cfg = PI05FullConfig(subtask_loss_weight=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        cfg.validate_features()


def test_all_zero_loss_weights_rejected():
    cfg = PI05FullConfig(
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=0.0,
    )
    with pytest.raises(ValueError, match="At least one loss weight"):
        cfg.validate_features()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_config.py -v
```

Expected: ImportError or AttributeError on `PI05FullConfig`.

- [ ] **Step 3: Rename `PI05Config` to `PI05FullConfig` and add new fields**

In `neuracore/ml/algorithms/pi05_full/utils.py`:

Find `class PI05Config:` and rename to `class PI05FullConfig:`. Append new fields after the existing ones (before `validate_features`):

```python
    # --- pi05_full additions ---
    # Loss weights (defaults match the JAX paper)
    subtask_loss_weight: float = 10.0
    fast_token_loss_weight: float = 1.0
    flow_matching_loss_weight: float = 1.0

    # Knowledge insulation (training only)
    knowledge_insulation: bool = True

    # Subtask + FAST tokenization
    max_subtask_tokens: int = 64
    max_fast_tokens: int = 128
    fast_tokenizer_name: str = "physical-intelligence/fast"
    fast_skip_tokens: int = 2048

    # Subtask generation at inference
    max_decoding_steps: int = 200
    subtask_temperature: float = 0.0
```

Extend `validate_features` (still in the same dataclass):

```python
    def validate_features(self) -> None:
        # ... existing validation ...

        for name, value in [
            ("subtask_loss_weight", self.subtask_loss_weight),
            ("fast_token_loss_weight", self.fast_token_loss_weight),
            ("flow_matching_loss_weight", self.flow_matching_loss_weight),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if (
            self.subtask_loss_weight == 0.0
            and self.fast_token_loss_weight == 0.0
            and self.flow_matching_loss_weight == 0.0
        ):
            raise ValueError(
                "At least one loss weight must be > 0. "
                "All zero would yield an untrainable model."
            )

        if self.max_subtask_tokens <= 0:
            raise ValueError(
                f"max_subtask_tokens must be > 0, got {self.max_subtask_tokens}"
            )
        if self.max_fast_tokens <= 0:
            raise ValueError(
                f"max_fast_tokens must be > 0, got {self.max_fast_tokens}"
            )
        if self.subtask_temperature < 0:
            raise ValueError(
                f"subtask_temperature must be >= 0, got {self.subtask_temperature}"
            )
```

- [ ] **Step 4: Update modules.py and pi05.py imports for the rename**

In `modules.py`, change:
```python
from neuracore.ml.algorithms.pi05.utils import (
    OPENPI_ATTENTION_MASK_VALUE,
    PI05Config,
    ...
)
```
to:
```python
from .utils import (
    OPENPI_ATTENTION_MASK_VALUE,
    PI05FullConfig,
    _align_mask_length,
    _create_sinusoidal_pos_embedding,
    _make_att_2d_masks,
    _sample_beta,
)
```

(Also rename `from neuracore.ml.algorithms.pi05.gemma_pytorch import ...` to `from .gemma_pytorch import ...`.)

Replace every occurrence of `PI05Config` with `PI05FullConfig` in `modules.py`. Same in `pi05.py` — also import from `.utils` and `.modules` rather than from the `pi05` package.

Replace `class PI05Policy` in `modules.py` with `class PI05FullPolicy`.

In `pi05.py`, replace the line `class Pi05(NeuracoreModel):` with `class Pi05Full(NeuracoreModel):`. Update `from .modules import PI05Policy` to `from .modules import PI05FullPolicy`. Replace internal references (`PI05Policy(...)`, `PI05Policy.from_pretrained`) with `PI05FullPolicy`.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_config.py -v
```

Expected: 7 PASSED.

- [ ] **Step 6: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/utils.py \
        neuracore/ml/algorithms/pi05_full/modules.py \
        neuracore/ml/algorithms/pi05_full/pi05.py \
        tests/unit/ml/algorithms/pi05_full/__init__.py \
        tests/unit/ml/algorithms/pi05_full/test_config.py
git commit -m "refactor: rename pi05 internals to PI05Full inside pi05_full

Renames Pi05 -> Pi05Full, PI05Policy -> PI05FullPolicy, PI05Config ->
PI05FullConfig inside neuracore/ml/algorithms/pi05_full/. Local imports
replace cross-package imports from pi05/ so pi05_full is self-contained.
Adds new config fields for loss weights, knowledge insulation, FAST and
subtask tokenization, and subtask generation. Validates the new fields."
```

### Task 3: Add FAST tokenizer loader and subtask prompt helpers

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/utils.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_fast_tokenization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_fast_tokenization.py
"""Tests for FAST tokenizer loading and action tokenization."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from neuracore.ml.algorithms.pi05_full.utils import (
    PI05FullConfig,
    fast_tokenize_actions,
    load_fast_tokenizer,
)


@pytest.fixture(scope="module")
def fast_tokenizer():
    return load_fast_tokenizer("physical-intelligence/fast")


def test_load_fast_tokenizer_returns_tokenizer(fast_tokenizer):
    # The FAST tokenizer is a HF AutoProcessor; we just need encode() to work.
    assert hasattr(fast_tokenizer, "__call__") or hasattr(fast_tokenizer, "encode")


def test_fast_tokenize_returns_padded_ids_and_mask(fast_tokenizer):
    cfg = PI05FullConfig()
    actions = np.random.randn(2, 10, 7).astype(np.float32)  # (B, T, action_dim)
    token_ids, mask = fast_tokenize_actions(
        actions,
        tokenizer=fast_tokenizer,
        max_tokens=cfg.max_fast_tokens,
        skip_tokens=cfg.fast_skip_tokens,
        vocab_size=257152,  # paligemma vocab size
    )
    assert token_ids.shape == (2, cfg.max_fast_tokens)
    assert mask.shape == (2, cfg.max_fast_tokens)
    assert token_ids.dtype == torch.long
    assert mask.dtype == torch.bool


def test_fast_tokens_land_in_paligemma_tail(fast_tokenizer):
    """FAST tokens must map into the last `fast_skip_tokens` slots of the vocab."""
    cfg = PI05FullConfig()
    actions = np.random.randn(1, 10, 7).astype(np.float32)
    token_ids, mask = fast_tokenize_actions(
        actions,
        tokenizer=fast_tokenizer,
        max_tokens=cfg.max_fast_tokens,
        skip_tokens=cfg.fast_skip_tokens,
        vocab_size=257152,
    )
    valid_ids = token_ids[mask]
    if valid_ids.numel() > 0:
        assert (valid_ids >= 257152 - cfg.fast_skip_tokens).all()
        assert (valid_ids < 257152).all()


def test_fast_tokenize_truncation_when_too_long(fast_tokenizer):
    """If FAST emits more tokens than max_fast_tokens, truncate from the right."""
    actions = np.random.randn(1, 200, 16).astype(np.float32)  # large chunk
    token_ids, mask = fast_tokenize_actions(
        actions,
        tokenizer=fast_tokenizer,
        max_tokens=8,  # tiny cap forces truncation
        skip_tokens=128,
        vocab_size=257152,
    )
    assert token_ids.shape == (1, 8)
    # mask should be all True (8 tokens of valid output, none padding)
    assert mask.all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_fast_tokenization.py -v
```

Expected: ImportError on `load_fast_tokenizer` / `fast_tokenize_actions`.

- [ ] **Step 3: Implement the helpers**

Add at the bottom of `neuracore/ml/algorithms/pi05_full/utils.py`:

```python
import numpy as np


def load_fast_tokenizer(name_or_path: str):
    """Load the FAST action tokenizer.

    The FAST tokenizer is a HuggingFace AutoProcessor that converts
    continuous action chunks (T x action_dim) into discrete token sequences.
    Loaded with trust_remote_code=True because FAST ships custom tokenization
    logic outside the standard HF tokenizer machinery.

    Args:
        name_or_path: HF repo id (default `physical-intelligence/fast`).

    Returns:
        The loaded tokenizer object with an `__call__` interface.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)


def fast_tokenize_actions(
    actions: np.ndarray | torch.Tensor,
    tokenizer,
    max_tokens: int,
    skip_tokens: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch of action chunks into discrete IDs in the vocab tail.

    Maps the FAST integer token IDs (which start at 0) into the last
    `skip_tokens` slots of the PaliGemma vocab so they share the LM head.

    Args:
        actions: (B, T, action_dim) float array or tensor.
        tokenizer: A FAST tokenizer (`physical-intelligence/fast`).
        max_tokens: Pad/truncate the per-example token sequence to this length.
        skip_tokens: Number of vocab slots reserved at the tail for FAST.
        vocab_size: Total PaliGemma vocab size (used to compute the offset).

    Returns:
        Tuple of:
        - token_ids: (B, max_tokens) int64 tensor with right-padding (pad value 0).
        - mask: (B, max_tokens) bool tensor; True where token_ids are real tokens.
    """
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
    else:
        actions_np = np.asarray(actions)

    if actions_np.ndim != 3:
        raise ValueError(
            f"actions must have shape (B, T, action_dim); got {actions_np.shape}"
        )

    batch_size = actions_np.shape[0]
    offset = vocab_size - skip_tokens

    out_ids = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    out_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)

    for b in range(batch_size):
        encoded = tokenizer(actions_np[b])
        if isinstance(encoded, dict):
            ids = list(encoded.get("input_ids", []))
        else:
            ids = list(encoded)
        if len(ids) == 0:
            continue
        ids = ids[:max_tokens]
        ids_t = torch.tensor(ids, dtype=torch.long) + offset
        out_ids[b, : len(ids)] = ids_t
        out_mask[b, : len(ids)] = True

    return out_ids, out_mask
```

The exact tokenizer call signature varies; if tests fail on the call shape, inspect with:

```bash
python -c "
from transformers import AutoProcessor
t = AutoProcessor.from_pretrained('physical-intelligence/fast', trust_remote_code=True)
import numpy as np
print(type(t(np.random.randn(10, 7))))
"
```

and adapt the unpacking inside `fast_tokenize_actions`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_fast_tokenization.py -v
```

Expected: 4 PASSED. (First run downloads the FAST tokenizer; subsequent runs use cache.)

- [ ] **Step 5: Add `numpy`, `denoise`, `denoised`, `denoising`, `paligemma`, `cspell`-flagged words to dictionary if cspell complains**

Run a sanity check:

```bash
git add neuracore/ml/algorithms/pi05_full/utils.py tests/unit/ml/algorithms/pi05_full/test_fast_tokenization.py
git commit -m "feat: add FAST tokenizer loader and action-to-token helper"
```

If cspell rejects words, append them to `neuracore-dictionary.txt` (one per line) and re-stage + commit.

---

## Phase C — Knowledge Insulation in attention (3 tasks)

### Task 4: Write the KI gradient-flow test (red)

**Files:**
- Test: `tests/unit/ml/algorithms/pi05_full/test_knowledge_insulation.py`

This test is the central correctness check for KI. Write it before any KI implementation so we have a falsifiable target.

- [ ] **Step 1: Create a tiny config fixture file (or inline it)**

```python
# tests/unit/ml/algorithms/pi05_full/conftest.py
"""Shared fixtures for pi05_full tests."""

import pytest
import torch

from neuracore.ml.algorithms.pi05_full.utils import PI05FullConfig


@pytest.fixture
def tiny_pi05_full_config() -> PI05FullConfig:
    """Smallest possible config for fast tests.

    Uses gemma_300m for both branches and reduces action dimensions and
    chunk sizes so a forward+backward fits comfortably in CPU memory.
    """
    return PI05FullConfig(
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        chunk_size=8,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=2,
        max_subtask_tokens=8,
        max_fast_tokens=8,
        max_decoding_steps=4,
        device="cpu",
        dtype="float32",
        gradient_checkpointing=False,
    )
```

- [ ] **Step 2: Write the KI gradient flow test**

```python
# tests/unit/ml/algorithms/pi05_full/test_knowledge_insulation.py
"""Tests that knowledge insulation severs gradient flow correctly."""

import pytest
import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy


def _vlm_param(policy: PI05FullPolicy) -> torch.nn.Parameter:
    """Pick a VLM K projection weight that sees gradient via attention."""
    return (
        policy.paligemma_with_expert.paligemma.language_model.layers[0]
        .self_attn.k_proj.weight
    )


def _run_training_forward(policy: PI05FullPolicy, knowledge_insulation: bool):
    """Run a tiny training forward and return the loss dict."""
    torch.manual_seed(0)
    bsize = 1
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)
    subtask_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    subtask_masks = torch.ones(bsize, 4, dtype=torch.bool)
    fast_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    fast_masks = torch.ones(bsize, 4, dtype=torch.bool)
    actions = torch.zeros(bsize, policy.config.chunk_size, policy.config.max_action_dim)

    policy.train()
    policy.config.knowledge_insulation = knowledge_insulation
    return policy.forward(
        [img], [img_mask],
        lang_tokens, lang_masks,
        subtask_tokens, subtask_masks,
        fast_tokens, fast_masks,
        actions,
    )


def test_ki_blocks_flow_gradient_into_vlm(tiny_pi05_full_config):
    """With KI on and only flow loss, VLM K-proj.grad must be zero."""
    policy = PI05FullPolicy(tiny_pi05_full_config)
    losses = _run_training_forward(policy, knowledge_insulation=True)
    losses["flow_mse_loss"].backward()
    assert torch.allclose(
        _vlm_param(policy).grad,
        torch.zeros_like(_vlm_param(policy).grad),
        atol=0.0,
    )


def test_ki_off_allows_flow_gradient_into_vlm(tiny_pi05_full_config):
    """With KI off and only flow loss, VLM K-proj.grad must be non-zero."""
    policy = PI05FullPolicy(tiny_pi05_full_config)
    losses = _run_training_forward(policy, knowledge_insulation=False)
    losses["flow_mse_loss"].backward()
    assert _vlm_param(policy).grad.abs().sum() > 0


def test_ki_does_not_block_subtask_gradient_into_vlm(tiny_pi05_full_config):
    """With KI on and only subtask loss, VLM K-proj.grad must be non-zero."""
    policy = PI05FullPolicy(tiny_pi05_full_config)
    losses = _run_training_forward(policy, knowledge_insulation=True)
    losses["subtask_ce_loss"].backward()
    assert _vlm_param(policy).grad.abs().sum() > 0
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_knowledge_insulation.py -v
```

Expected: FAIL — likely `PI05FullPolicy.forward` signature mismatch (we haven't extended it yet) or missing `knowledge_insulation` config support.

This is the target. Tasks 5, 6, 8, 11 will collectively land all the pieces needed to make it pass. Don't commit a passing version here — just confirm the test exists and is currently red.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/unit/ml/algorithms/pi05_full/conftest.py \
        tests/unit/ml/algorithms/pi05_full/test_knowledge_insulation.py
git commit -m "test: add knowledge insulation gradient flow test (red)

Test currently fails because the model code has not been extended yet.
Will pass after the KI attention path and the three-loss forward are
implemented."
```

### Task 5: Implement KI attention path in `compute_shared_attention_layer`

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/gemma_pytorch.py`

- [ ] **Step 1: Read the current `compute_shared_attention_layer` and `PaliGemmaWithExpertModel.forward`**

```bash
sed -n '17,140p' neuracore/ml/algorithms/pi05_full/gemma_pytorch.py
sed -n '362,475p' neuracore/ml/algorithms/pi05_full/gemma_pytorch.py
```

Confirm `inputs_embeds[0]` is the PaliGemma branch and `inputs_embeds[1]` is the action expert. The boundary in the concatenated Q/K/V along `dim=2` is `inputs_embeds[0].shape[1]`.

- [ ] **Step 2: Add a `knowledge_insulation` parameter to `compute_shared_attention_layer`**

In `gemma_pytorch.py`, change the function signature:

```python
def compute_shared_attention_layer(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    adarms_cond: list[torch.Tensor | None],
    paligemma: PaliGemmaForConditionalGeneration,
    gemma_expert: GemmaForCausalLM,
    knowledge_insulation: bool = False,
) -> list[torch.Tensor]:
```

Replace the single `eager_attention_forward` call (currently around line 96) with the KI branch:

```python
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    self_attn = paligemma.language_model.layers[layer_idx].self_attn

    apply_ki = (
        knowledge_insulation
        and inputs_embeds[1] is not None
        and inputs_embeds[0] is not None
        and any(inp.requires_grad for inp in inputs_embeds if inp is not None)
    )

    if apply_ki:
        prefix_len = inputs_embeds[0].shape[1]

        q_vlm = query_states[:, :, :prefix_len, :]
        q_action = query_states[:, :, prefix_len:, :]
        k_vlm = key_states[:, :, :prefix_len, :]
        k_action = key_states[:, :, prefix_len:, :]
        v_vlm = value_states[:, :, :prefix_len, :]
        v_action = value_states[:, :, prefix_len:, :]

        # VLM queries see full K/V (gradients flow normally)
        att_output_vlm, _ = modeling_gemma.eager_attention_forward(
            self_attn,
            q_vlm,
            torch.cat([k_vlm, k_action], dim=2),
            torch.cat([v_vlm, v_action], dim=2),
            attention_mask[:, :, :prefix_len, :],
            scaling,
        )

        # Action queries see VLM K/V detached — gradient cannot reach VLM
        att_output_action, _ = modeling_gemma.eager_attention_forward(
            self_attn,
            q_action,
            torch.cat([k_vlm.detach(), k_action], dim=2),
            torch.cat([v_vlm.detach(), v_action], dim=2),
            attention_mask[:, :, prefix_len:, :],
            scaling,
        )

        att_output = torch.cat([att_output_vlm, att_output_action], dim=2)
    else:
        att_output, _ = modeling_gemma.eager_attention_forward(
            self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
```

The rest of the function (head-dim reshape, per-stream `o_proj` and MLP) stays unchanged.

- [ ] **Step 3: Add `knowledge_insulation` parameter to `PaliGemmaWithExpertModel.forward`**

Find `def forward(` in `PaliGemmaWithExpertModel` (around line 362). Add a new keyword arg:

```python
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
        knowledge_insulation: bool = False,
    ) -> tuple[list[torch.Tensor | None], list[torch.FloatTensor] | None]:
```

In the joint-forward branch (around line 448-471), pass `knowledge_insulation` to `compute_shared_attention_layer`:

```python
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_shared_attention_layer,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        knowledge_insulation=knowledge_insulation,
                    )
                else:
                    inputs_embeds = compute_shared_attention_layer(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        knowledge_insulation=knowledge_insulation,
                    )
```

Suffix-only and prefix-only branches are unaffected (KI only applies to joint forward).

- [ ] **Step 4: Don't run tests yet — they need the `forward()` extension from Task 6**

We'll verify the green state in Task 8 after the forward signature is extended in Task 6.

- [ ] **Step 5: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/gemma_pytorch.py
git commit -m "feat: add knowledge insulation branch in compute_shared_attention_layer

Splits attention into per-stream calls when knowledge_insulation=True so
action queries attend to VLM K/V via .detach(). Forward values unchanged;
backward gradients from action losses cannot reach VLM K/V projections.
Inactive at inference (no requires_grad) and when only one branch is
present."
```

### Task 6: Extend `PI05FullPolicy.forward` to take subtask + FAST tokens and return three losses

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/modules.py`
- Test: existing `test_knowledge_insulation.py` (will start to pass after this)

- [ ] **Step 1: Extend `_embed_prefix` to accept subtask + FAST segments**

Find `def _embed_prefix(` in `modules.py` and replace its body with the four-segment version:

```python
    def _embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        subtask_tokens: Tensor | None = None,
        subtask_masks: Tensor | None = None,
        fast_tokens: Tensor | None = None,
        fast_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, slice]]:
        """Embed image, language, subtask, and FAST tokens into one prefix sequence.

        Returns:
            embs, pad_masks, att_masks_t, segments — where segments maps each
            segment name to the slice it occupies in the prefix sequence.
        """
        embs = []
        pad_masks = []
        att_masks: list[int] = []
        segments: dict[str, slice] = {}

        cursor = 0
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img: Tensor) -> Tensor:
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
            cursor += num_img_embs

        def lang_embed_func(lt: Tensor) -> Tensor:
            le = self.paligemma_with_expert.embed_language_tokens(lt)
            return le * math.sqrt(le.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        cursor += lang_emb.shape[1]

        if subtask_tokens is not None:
            subtask_emb = self._apply_checkpoint(lang_embed_func, subtask_tokens)
            embs.append(subtask_emb)
            assert subtask_masks is not None
            pad_masks.append(subtask_masks)
            seg_len = subtask_emb.shape[1]
            # First subtask token starts a new causal segment (att=1), rest follow.
            att_masks += [1] + [0] * (seg_len - 1)
            segments["subtask"] = slice(cursor, cursor + seg_len)
            cursor += seg_len

        if fast_tokens is not None:
            fast_emb = self._apply_checkpoint(lang_embed_func, fast_tokens)
            embs.append(fast_emb)
            assert fast_masks is not None
            pad_masks.append(fast_masks)
            seg_len = fast_emb.shape[1]
            att_masks += [1] + [0] * (seg_len - 1)
            segments["fast"] = slice(cursor, cursor + seg_len)
            cursor += seg_len

        embs_t = torch.cat(embs, dim=1)
        pad_masks_t = torch.cat(pad_masks, dim=1).to(dtype=torch.bool)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks_t.device)
        att_masks_t = _align_mask_length(att_masks_t, pad_masks_t.shape[1])
        bsize = pad_masks_t.shape[0]
        att_masks_t = att_masks_t[None, :].expand(bsize, att_masks_t.shape[0])
        return embs_t, pad_masks_t, att_masks_t, segments
```

Note: callers that previously expected a 3-tuple now must unpack 4 elements. Update `sample_actions` and `_denoise_step` accordingly (just discard the 4th return with `_`).

- [ ] **Step 2: Replace `forward` signature and body**

Replace the existing `def forward(` of `PI05FullPolicy` with:

```python
    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        subtask_tokens: Tensor,
        subtask_masks: Tensor,
        fast_tokens: Tensor,
        fast_masks: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the three losses for pi05_full training.

        Returns:
            Dict with keys flow_mse_loss, subtask_ce_loss, fast_ce_loss, loss.
            All values are scalar tensors.
        """
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device)
        if time is None:
            time = self._sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, segments = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
            subtask_tokens=subtask_tokens, subtask_masks=subtask_masks,
            fast_tokens=fast_tokens, fast_masks=fast_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self._embed_suffix(x_t, time)
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # Suffix MUST NOT attend to FAST tokens. Zero out the attention from
        # suffix positions to fast-segment positions in the 2D mask.
        att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
        if "fast" in segments:
            fast_slice = segments["fast"]
            suffix_start = pad_masks.shape[1] - suffix_pad_masks.shape[1]
            att_2d_masks[:, suffix_start:, fast_slice] = False

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids,
                         adarms_cond):
            outs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                knowledge_insulation=self.config.knowledge_insulation,
            )
            return outs[0], outs[1]

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)

        # Flow matching MSE
        v_t = self._apply_checkpoint(
            lambda s: self.action_out_proj(s), suffix_out
        )
        flow_mse_loss = F.mse_loss(u_t, v_t, reduction="none").mean()

        # Subtask CE via tied LM head
        lm_head = self.paligemma_with_expert.paligemma.lm_head
        subtask_slice = segments["subtask"]
        subtask_hidden = prefix_out[:, subtask_slice, :].to(dtype=torch.float32)
        subtask_logits = lm_head(subtask_hidden)
        subtask_ce_loss = self._token_ce(
            subtask_logits, subtask_tokens, subtask_masks
        )

        # FAST CE via tied LM head
        fast_slice = segments["fast"]
        fast_hidden = prefix_out[:, fast_slice, :].to(dtype=torch.float32)
        fast_logits = lm_head(fast_hidden)
        fast_ce_loss = self._token_ce(fast_logits, fast_tokens, fast_masks)

        loss = (
            self.config.flow_matching_loss_weight * flow_mse_loss
            + self.config.subtask_loss_weight * subtask_ce_loss
            + self.config.fast_token_loss_weight * fast_ce_loss
        )
        return {
            "flow_mse_loss": flow_mse_loss,
            "subtask_ce_loss": subtask_ce_loss,
            "fast_ce_loss": fast_ce_loss,
            "loss": loss,
        }

    @staticmethod
    def _token_ce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Shifted cross-entropy with right-padding mask, computed in float32.

        Predicts targets[:, 1:] from logits[:, :-1] (causal shift).
        Returns mask-averaged scalar loss.
        """
        if logits.shape[1] < 2:
            return torch.zeros((), device=logits.device, dtype=torch.float32)
        logits_for_pred = logits[:, :-1, :].contiguous()
        targets_for_pred = targets[:, 1:].contiguous()
        mask_for_pred = mask[:, 1:].contiguous().to(dtype=torch.float32)
        per_token = F.cross_entropy(
            logits_for_pred.view(-1, logits_for_pred.shape[-1]),
            targets_for_pred.view(-1),
            reduction="none",
        ).view_as(targets_for_pred).to(dtype=torch.float32)
        weighted = per_token * mask_for_pred
        denom = mask_for_pred.sum().clamp(min=1.0)
        return weighted.sum() / denom
```

- [ ] **Step 3: Update `sample_actions` and `_denoise_step` to accept the new `_embed_prefix` return shape and to take an optional `subtask_tokens`**

In `sample_actions`, replace the prefix embed call:

```python
        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
        )
```

Add `subtask_tokens: Tensor | None = None` and `subtask_masks: Tensor | None = None` to the `sample_actions` signature.

`_denoise_step` is unchanged because it operates on cached prefix.

- [ ] **Step 4: Run the KI test from Task 4**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_knowledge_insulation.py -v
```

Expected: 3 PASSED. If a test fails:
- `test_ki_blocks_flow_gradient_into_vlm`: KI is leaking. Verify the `.detach()` call is on `k_vlm` and `v_vlm` not on `q_vlm`.
- `test_ki_off_allows_flow_gradient_into_vlm`: forward path is broken. Verify that with `apply_ki=False`, `eager_attention_forward` runs once on the concatenated streams.
- `test_ki_does_not_block_subtask_gradient_into_vlm`: subtask CE is reaching the wrong stream. Verify subtask hidden states come from `prefix_out`, not from the action expert output.

- [ ] **Step 5: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/modules.py
git commit -m "feat: extend PI05FullPolicy.forward with subtask + FAST + 3 losses

Adds subtask and FAST token segments to the prefix; computes flow MSE,
subtask CE, and FAST CE losses; returns the four scalars as a dict.
Suffix-to-FAST attention is masked off in the 2D attention mask.
Knowledge insulation flag is plumbed through to the attention layer."
```

---

## Phase D — Inference: subtask generation (2 tasks)

### Task 7: Implement `generate_subtask_tokens` autoregressive decode loop

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/modules.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_subtask_generation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_subtask_generation.py
"""Tests for autoregressive subtask generation at inference."""

import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy


def test_generate_subtask_tokens_returns_correct_shape(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    policy.eval()
    bsize = 2
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)
    bos_id = 2  # PaliGemma BOS

    tokens, masks = policy.generate_subtask_tokens(
        [img], [img_mask], lang_tokens, lang_masks, bos_token_id=bos_id,
    )
    assert tokens.shape[0] == bsize
    assert tokens.shape[1] <= tiny_pi05_full_config.max_decoding_steps + 1
    assert tokens.dtype == torch.long
    assert masks.shape == tokens.shape
    assert masks.dtype == torch.bool


def test_generate_starts_with_bos(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    policy.eval()
    bsize = 1
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)
    bos_id = 2

    tokens, _ = policy.generate_subtask_tokens(
        [img], [img_mask], lang_tokens, lang_masks, bos_token_id=bos_id,
    )
    assert tokens[0, 0].item() == bos_id
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_subtask_generation.py -v
```

Expected: FAIL — `AttributeError: 'PI05FullPolicy' object has no attribute 'generate_subtask_tokens'`.

- [ ] **Step 3: Implement `generate_subtask_tokens`**

In `modules.py`, add to `PI05FullPolicy`:

```python
    @torch.no_grad()
    def generate_subtask_tokens(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        bos_token_id: int,
        eos_token_id: int | None = None,
        loc_token_id: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Autoregressively generate subtask tokens.

        Args:
            images / img_masks / lang_tokens / lang_masks: standard prefix inputs.
            bos_token_id: PaliGemma BOS token to seed generation.
            eos_token_id: optional EOS to halt early per-batch-item.
            loc_token_id: optional first <loc####> id; all token ids >= this are
                masked out before sampling so we never emit visual-grounding tokens.

        Returns:
            Tuple of:
            - generated_tokens: (B, L) int64 — starts with BOS, ends at EOS or cap.
            - masks: (B, L) bool — True for valid (non-pad) tokens.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        max_steps = self.config.max_decoding_steps
        temperature = self.config.subtask_temperature

        # Phase A: prefill cache with [images, language, BOS]
        bos_col = torch.full((bsize, 1), bos_token_id, dtype=torch.long, device=device)
        bos_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
            subtask_tokens=bos_col, subtask_masks=bos_mask,
        )
        att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        outs, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_out = outs[0]

        lm_head = self.paligemma_with_expert.paligemma.lm_head

        def _sample(logits: Tensor) -> Tensor:
            if loc_token_id is not None:
                logits[:, loc_token_id:] = float("-inf")
            if temperature == 0.0:
                return torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

        first_logits = lm_head(prefix_out[:, -1:, :].to(dtype=torch.float32))[:, -1]
        next_tok = _sample(first_logits)

        generated = [bos_col, next_tok[:, None]]
        finished = torch.zeros(bsize, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            finished |= next_tok == eos_token_id

        # Phase B: autoregressive decode
        # Keep a running pad mask so position_ids stay correct.
        running_pad = torch.cat([prefix_pad_masks, bos_mask], dim=1)
        for _ in range(max_steps - 1):
            if finished.all():
                break
            tok_emb = self.paligemma_with_expert.embed_language_tokens(next_tok[:, None])
            tok_emb = tok_emb * math.sqrt(tok_emb.shape[-1])

            running_pad = torch.cat(
                [running_pad, torch.ones(bsize, 1, dtype=torch.bool, device=device)],
                dim=1,
            )
            position_ids = (running_pad.long().cumsum(dim=1) - 1)[:, -1:]
            # Single-token attention mask: attends to entire history.
            step_pad_4d = self._prepare_attention_masks_4d(
                running_pad[:, None, None, :].expand(bsize, 1, 1, running_pad.shape[1])
            )

            outs, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=step_pad_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[tok_emb, None],
                use_cache=True,
            )
            step_out = outs[0]
            step_logits = lm_head(step_out[:, -1:, :].to(dtype=torch.float32))[:, -1]
            next_tok = _sample(step_logits)
            # Force finished sequences to emit pad (id=0)
            next_tok = torch.where(finished, torch.zeros_like(next_tok), next_tok)
            if eos_token_id is not None:
                finished |= next_tok == eos_token_id
            generated.append(next_tok[:, None])

        generated_tokens = torch.cat(generated, dim=1)
        masks = generated_tokens != 0
        masks[:, 0] = True  # always keep BOS valid
        return generated_tokens, masks
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_subtask_generation.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/modules.py \
        tests/unit/ml/algorithms/pi05_full/test_subtask_generation.py
git commit -m "feat: implement autoregressive subtask token generation

Adds PI05FullPolicy.generate_subtask_tokens that runs PaliGemma
autoregressively from a BOS seed, producing a per-batch token sequence.
Greedy by default, temperature-controllable; supports EOS-driven halt and
masking out visual-grounding <loc####> tokens."
```

### Task 8: Wire generated subtask into `sample_actions` for end-to-end inference

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/modules.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_inference_flow.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_inference_flow.py
"""End-to-end inference: subtask gen → flow denoise → actions."""

import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy


def test_sample_actions_with_generated_subtask(tiny_pi05_full_config):
    policy = PI05FullPolicy(tiny_pi05_full_config)
    policy.eval()
    bsize = 1
    img = torch.randn(bsize, 3, 224, 224)
    img_mask = torch.ones(bsize, dtype=torch.bool)
    lang_tokens = torch.zeros(bsize, 4, dtype=torch.long)
    lang_masks = torch.ones(bsize, 4, dtype=torch.bool)

    subtask, subtask_mask = policy.generate_subtask_tokens(
        [img], [img_mask], lang_tokens, lang_masks, bos_token_id=2,
    )
    actions = policy.sample_actions(
        [img], [img_mask], lang_tokens, lang_masks,
        subtask_tokens=subtask, subtask_masks=subtask_mask,
    )
    assert actions.shape == (
        bsize,
        tiny_pi05_full_config.chunk_size,
        tiny_pi05_full_config.max_action_dim,
    )
    assert torch.isfinite(actions).all()
```

- [ ] **Step 2: Run test**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_inference_flow.py -v
```

Expected: FAIL on the `sample_actions` signature (it now takes `subtask_tokens` keyword).

- [ ] **Step 3: Update `sample_actions` to accept subtask tokens**

In `modules.py`, the existing `sample_actions` already calls `_embed_prefix`. Confirm the signature now reads:

```python
    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        subtask_tokens: Tensor | None = None,
        subtask_masks: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
```

And the body's `_embed_prefix` call passes them through (was edited in Task 6):

```python
        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
        )
```

(No FAST tokens at inference — the prefix is `[images, language, subtask]`.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_inference_flow.py -v
```

Expected: 1 PASSED.

- [ ] **Step 5: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/modules.py \
        tests/unit/ml/algorithms/pi05_full/test_inference_flow.py
git commit -m "feat: condition flow-matching denoise on generated subtask tokens

PI05FullPolicy.sample_actions now accepts subtask_tokens / subtask_masks
and embeds them into the prefix before the denoising loop, completing
the inference path: subtask gen -> flow denoise -> actions."
```

---

## Phase E — `Pi05Full` wrapper integration (5 tasks)

### Task 9: Construction validation: require SUBTASK_LANGUAGE in/out, validate weights

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/pi05.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_construction_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_construction_validation.py
"""Construction-time validation for Pi05Full."""

import pytest

from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full

# Helper builds a minimal ModelInitDescription. Reuse pi05's existing fixture if
# present, else inline (this code is heavily project-specific; check existing
# tests in tests/unit/ml/algorithms/pi05/ for the canonical shape).


def _make_model_init_description_without_subtask():
    """Build a description that has LANGUAGE inputs but not SUBTASK_LANGUAGE."""
    # See tests/unit/ml/algorithms/pi05/test_pi05.py for the canonical pattern.
    raise NotImplementedError("Adapt from existing pi05 test fixtures.")


# These tests will be skipped via xfail until the fixture is wired (see Step 2).
@pytest.mark.xfail(reason="Awaiting model_init_description fixture wiring", strict=False)
def test_missing_subtask_input_raises():
    desc = _make_model_init_description_without_subtask()
    with pytest.raises(ValueError, match="SUBTASK_LANGUAGE"):
        Pi05Full(desc)
```

- [ ] **Step 2: Adapt the existing pi05 test fixture for SUBTASK_LANGUAGE**

```bash
cat tests/unit/ml/algorithms/pi05/test_pi05.py | head -60
ls tests/unit/ml/algorithms/pi05/
```

Reuse the helper that builds `ModelInitDescription`. Extend it to include `SUBTASK_LANGUAGE` in inputs and outputs by adding the relevant `dataset_statistics` entry (a `LanguageDataStats(text=DataItemStats())`). Drop the `xfail` once the fixture is in place. The full test should look like:

```python
import pytest
from neuracore_types import DataType
from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full
# ... import the model_init_description builder used by pi05 tests ...


def test_missing_subtask_input_raises(model_init_description_no_subtask):
    with pytest.raises(ValueError, match="SUBTASK_LANGUAGE"):
        Pi05Full(model_init_description_no_subtask)


def test_missing_subtask_output_raises(model_init_description_no_subtask_output):
    with pytest.raises(ValueError, match="SUBTASK_LANGUAGE"):
        Pi05Full(model_init_description_no_subtask_output)


def test_negative_loss_weight_raises(model_init_description_with_subtask):
    with pytest.raises(ValueError, match="non-negative"):
        Pi05Full(model_init_description_with_subtask, subtask_loss_weight=-1.0)


def test_all_zero_loss_weights_raise(model_init_description_with_subtask):
    with pytest.raises(ValueError, match="At least one loss weight"):
        Pi05Full(
            model_init_description_with_subtask,
            subtask_loss_weight=0.0,
            fast_token_loss_weight=0.0,
            flow_matching_loss_weight=0.0,
        )


def test_warns_when_subtask_loss_with_action_expert_only(
    caplog, model_init_description_with_subtask,
):
    Pi05Full(
        model_init_description_with_subtask,
        finetune_action_expert_only=True,
        subtask_loss_weight=10.0,
    )
    assert "subtask CE loss has no effect" in caplog.text
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_construction_validation.py -v
```

- [ ] **Step 4: Implement validation in `Pi05Full.__init__`**

In `pi05.py`, add to `Pi05Full.__init__` after `super().__init__(model_init_description)` and before any other setup:

```python
        from neuracore_types import DataType  # local import to avoid cycles

        if DataType.SUBTASK_LANGUAGE not in self.input_data_types:
            raise ValueError(
                "Pi05Full requires SUBTASK_LANGUAGE in inputs. Use the Pi05 "
                "algorithm if your dataset has no subtask annotations."
            )
        if DataType.SUBTASK_LANGUAGE not in self.output_data_types:
            raise ValueError(
                "Pi05Full requires SUBTASK_LANGUAGE in outputs. Add it to your "
                "configured output data types (it is always produced by the model)."
            )

        # Loss weight validation (mirrors PI05FullConfig.validate_features)
        for name, value in [
            ("subtask_loss_weight", subtask_loss_weight),
            ("fast_token_loss_weight", fast_token_loss_weight),
            ("flow_matching_loss_weight", flow_matching_loss_weight),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if (
            subtask_loss_weight == 0.0
            and fast_token_loss_weight == 0.0
            and flow_matching_loss_weight == 0.0
        ):
            raise ValueError(
                "At least one loss weight must be > 0. All zero would yield an "
                "untrainable model."
            )

        # Misconfiguration warnings
        if finetune_action_expert_only and subtask_loss_weight > 0:
            logger.warning(
                "subtask CE loss has no effect when finetune_action_expert_only "
                "is True (VLM is frozen). Set subtask_loss_weight=0 to save compute."
            )
        if finetune_action_expert_only and fast_token_loss_weight > 0:
            logger.warning(
                "fast_token CE loss has no effect when finetune_action_expert_only "
                "is True. Set fast_token_loss_weight=0 to save compute."
            )
```

Also add the new constructor kwargs (`subtask_loss_weight=10.0`, `fast_token_loss_weight=1.0`, `flow_matching_loss_weight=1.0`, `knowledge_insulation=True`, `max_subtask_tokens=64`, `max_fast_tokens=128`, `max_decoding_steps=200`, `subtask_temperature=0.0`, `fast_tokenizer_name="physical-intelligence/fast"`, `fast_skip_tokens=128`) to `__init__` and store them as instance attributes. Pass them into `PI05FullConfig`.

- [ ] **Step 5: Update `get_supported_input_data_types` and `get_supported_output_data_types`**

```python
    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.RGB_IMAGES,
            DataType.LANGUAGE,
            DataType.SUBTASK_LANGUAGE,
        }

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
            DataType.SUBTASK_LANGUAGE,
        }
```

Add `DataType.SUBTASK_LANGUAGE` to `CANONICAL_OUTPUT_DATA_TYPE_ORDER` if Neuracore's base class enforces an ordering. Check `neuracore/ml/core/neuracore_model.py`:

```python
DEFAULT_OUTPUT_DATA_TYPE_ORDER: tuple[DataType, ...] = (
    DataType.JOINT_TARGET_POSITIONS,
    DataType.JOINT_POSITIONS,
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
)
```

Override on `Pi05Full`:

```python
    CANONICAL_OUTPUT_DATA_TYPE_ORDER = (
        DataType.JOINT_TARGET_POSITIONS,
        DataType.JOINT_POSITIONS,
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        DataType.SUBTASK_LANGUAGE,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_construction_validation.py -v
```

Expected: all PASSED.

- [ ] **Step 7: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/pi05.py \
        tests/unit/ml/algorithms/pi05_full/test_construction_validation.py
git commit -m "feat: validate SUBTASK_LANGUAGE and loss weights at construction

Pi05Full now requires SUBTASK_LANGUAGE in both inputs and outputs, rejects
negative or all-zero loss weights, and warns about misconfiguration with
finetune_action_expert_only. Adds SUBTASK_LANGUAGE to supported in/out
data types and to the canonical output order."
```

### Task 10: Implement `_process_subtask_tokens` helper in `Pi05Full`

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/pi05.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_subtask_token_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_subtask_token_pipeline.py
"""Tests _process_subtask_tokens turns BatchedSubtaskLanguageData into
fixed-length token tensors with masks."""

import torch

from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full


def test_subtask_tokens_are_padded_to_max_length(model_init_description_with_subtask):
    model = Pi05Full(model_init_description_with_subtask, max_subtask_tokens=8)
    batch = _make_batch_with_subtask_text(["pick up the cup", "place down"])
    tokens, masks = model._process_subtask_tokens(batch)
    assert tokens.shape == (2, 8)
    assert masks.shape == (2, 8)
    assert tokens.dtype == torch.long
    # First token should be BOS (not pad)
    assert masks[:, 0].all()
```

`_make_batch_with_subtask_text` is a helper that constructs a `BatchedInferenceInputs` with a `SUBTASK_LANGUAGE` channel containing the given text. Reuse `BatchedLanguageData.from_nc_data_list` for tokenization. Create the helper in `tests/unit/ml/algorithms/pi05_full/conftest.py`.

- [ ] **Step 2: Implement `_process_subtask_tokens`**

In `Pi05Full` (in `pi05.py`):

```python
    def _process_subtask_tokens(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract last-timestep subtask tokens, prepend BOS, pad to max_subtask_tokens.

        Returns:
            tokens: (B, max_subtask_tokens) int64
            masks: (B, max_subtask_tokens) bool
        """
        from neuracore_types.batched_nc_data.batched_language_data import (
            BatchedLanguageData,
        )

        if DataType.SUBTASK_LANGUAGE not in batch.inputs:
            raise ValueError(
                "Subtask channel missing from batch.inputs. Pi05Full training "
                "requires SUBTASK_LANGUAGE per batch."
            )
        items = cast(
            list[BatchedLanguageData], batch.inputs[DataType.SUBTASK_LANGUAGE]
        )
        # Use the last subtask channel and the last timestep, mirroring how
        # LANGUAGE is consumed in the base pi05 pipeline.
        last = items[-1]
        ids = last.input_ids[:, -1, :]  # (B, L)
        attn = last.attention_mask[:, -1, :].to(dtype=torch.bool)

        bsize = ids.shape[0]
        max_len = self.max_subtask_tokens
        bos = self.prompt_tokenizer.bos_token_id
        if bos is None:
            bos = self.prompt_tokenizer.eos_token_id

        out_ids = torch.full((bsize, max_len), 0, dtype=torch.long)
        out_mask = torch.zeros(bsize, max_len, dtype=torch.bool)

        for i in range(bsize):
            valid = ids[i][attn[i]].detach().cpu().tolist()
            seq = [bos] + valid
            seq = seq[:max_len]
            out_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            out_mask[i, : len(seq)] = True

        return out_ids.to(self.device), out_mask.to(self.device)
```

- [ ] **Step 3: Run test to verify it passes**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_subtask_token_pipeline.py -v
```

- [ ] **Step 4: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/pi05.py \
        tests/unit/ml/algorithms/pi05_full/test_subtask_token_pipeline.py \
        tests/unit/ml/algorithms/pi05_full/conftest.py
git commit -m "feat: extract subtask tokens from batch with BOS + right-padding"
```

### Task 11: Implement `_build_action_targets_and_fast_tokens` in `Pi05Full`

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/pi05.py`

- [ ] **Step 1: Implement the helper**

Add to `Pi05Full`:

```python
    def _build_action_targets_and_fast_tokens(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate action targets, normalize, FAST-tokenize.

        Returns:
            target_actions: (B, T, max_action_dim) float
            fast_tokens: (B, max_fast_tokens) int64
            fast_masks: (B, max_fast_tokens) bool
        """
        action_targets = []
        for data_type in self.ordered_output_data_types:
            if data_type == DataType.SUBTASK_LANGUAGE:
                continue
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                joints = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend(j.value for j in joints)
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData],
                    batch.outputs[data_type],
                )
                action_targets.extend(g.open_amount for g in grippers)
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        action_data = torch.cat(action_targets, dim=-1)
        target_actions = self.action_normalizer.normalize(data=action_data)
        target_actions = pad_vector(target_actions, self.max_action_dim).to(self.device)

        # FAST tokenize on CPU
        from .utils import fast_tokenize_actions

        vocab_size = self.prompt_tokenizer.vocab_size
        fast_ids, fast_mask = fast_tokenize_actions(
            target_actions.detach(),
            tokenizer=self.fast_tokenizer,
            max_tokens=self.max_fast_tokens,
            skip_tokens=self.fast_skip_tokens,
            vocab_size=vocab_size,
        )
        return (
            target_actions,
            fast_ids.to(self.device),
            fast_mask.to(self.device),
        )
```

Ensure `self.fast_tokenizer` is loaded in `__init__`:

```python
        from .utils import load_fast_tokenizer
        self.fast_tokenizer = load_fast_tokenizer(self.fast_tokenizer_name)
```

- [ ] **Step 2: Add a smoke test**

```python
# Append to tests/unit/ml/algorithms/pi05_full/test_subtask_token_pipeline.py

def test_action_and_fast_token_shapes(
    model_init_description_with_subtask, synthetic_training_batch,
):
    model = Pi05Full(model_init_description_with_subtask)
    actions, fast_ids, fast_mask = model._build_action_targets_and_fast_tokens(
        synthetic_training_batch,
    )
    bsize = synthetic_training_batch.batch_size
    assert actions.shape == (bsize, model.output_prediction_horizon, model.max_action_dim)
    assert fast_ids.shape == (bsize, model.max_fast_tokens)
    assert fast_mask.shape == (bsize, model.max_fast_tokens)
```

`synthetic_training_batch` is a fixture in `conftest.py` that builds a `BatchedTrainingSamples` with the required channels. Refer to the existing pi05 test patterns for the construction.

- [ ] **Step 3: Run tests**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_subtask_token_pipeline.py -v
```

- [ ] **Step 4: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/pi05.py \
        tests/unit/ml/algorithms/pi05_full/test_subtask_token_pipeline.py \
        tests/unit/ml/algorithms/pi05_full/conftest.py
git commit -m "feat: build action targets + FAST tokens in Pi05Full helper"
```

### Task 12: Wire `training_step` to compute and return four losses

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/pi05.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_training_step.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_training_step.py
"""End-to-end training step on a synthetic batch."""

import torch

from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full


def test_training_step_returns_four_finite_losses(
    model_init_description_with_subtask, synthetic_training_batch,
):
    model = Pi05Full(model_init_description_with_subtask)
    out = model.training_step(synthetic_training_batch)
    expected = {"loss", "flow_mse_loss", "subtask_ce_loss", "fast_ce_loss"}
    assert set(out.losses.keys()) == expected
    assert set(out.metrics.keys()) == expected
    for k in expected:
        assert torch.isfinite(out.losses[k]).all()
    assert out.losses["loss"].requires_grad
```

- [ ] **Step 2: Replace the existing `training_step` body with the three-loss version**

In `pi05.py`:

```python
    def training_step(self, batch):
        inference_sample = BatchedInferenceInputs(
            inputs=batch.inputs, inputs_mask=batch.inputs_mask,
            batch_size=batch.batch_size,
        )
        images, image_masks, lang_tokens, lang_masks = self._build_inputs_from_batch(
            inference_sample
        )
        subtask_tokens, subtask_masks = self._process_subtask_tokens(inference_sample)
        target_actions, fast_tokens, fast_masks = (
            self._build_action_targets_and_fast_tokens(batch)
        )

        loss_dict = self.model.forward(
            images, image_masks,
            lang_tokens, lang_masks,
            subtask_tokens, subtask_masks,
            fast_tokens, fast_masks,
            target_actions,
        )

        return BatchedTrainingOutputs(
            losses={k: v for k, v in loss_dict.items()},
            metrics={k: v.detach() for k, v in loss_dict.items()},
        )
```

- [ ] **Step 3: Run test**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_training_step.py -v
```

Expected: PASSED.

- [ ] **Step 4: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/pi05.py \
        tests/unit/ml/algorithms/pi05_full/test_training_step.py
git commit -m "feat: wire pi05_full training step to three-loss forward"
```

### Task 13: Wire `forward` (inference) to return SUBTASK_LANGUAGE output

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/pi05.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_inference_outputs.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_inference_outputs.py
"""Inference returns joint/gripper actions plus a SUBTASK_LANGUAGE field."""

from neuracore_types import DataType
from neuracore_types.batched_nc_data.batched_language_data import BatchedLanguageData

from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full


def test_forward_returns_subtask_language(
    model_init_description_with_subtask, synthetic_inference_batch,
):
    model = Pi05Full(model_init_description_with_subtask)
    out = model.forward(synthetic_inference_batch)
    assert DataType.SUBTASK_LANGUAGE in out
    items = out[DataType.SUBTASK_LANGUAGE]
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, BatchedLanguageData)
    assert item.input_ids.shape[0] == synthetic_inference_batch.batch_size
```

- [ ] **Step 2: Update `Pi05Full.forward` to generate subtasks and emit them**

In `pi05.py`:

```python
    def forward(self, batch):
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        if self.compile_model:
            self.model.compile_model_enable()

        images, image_masks, lang_tokens, lang_masks = self._build_inputs_from_batch(batch)
        bos_id = self.prompt_tokenizer.bos_token_id
        eos_id = self.prompt_tokenizer.eos_token_id
        loc0_id = self.prompt_tokenizer.convert_tokens_to_ids("<loc0000>")
        if loc0_id == self.prompt_tokenizer.unk_token_id:
            loc0_id = None  # tokenizer doesn't know <loc####>; skip masking

        subtask_tokens, subtask_masks = self.model.generate_subtask_tokens(
            images, image_masks, lang_tokens, lang_masks,
            bos_token_id=bos_id, eos_token_id=eos_id, loc_token_id=loc0_id,
        )

        actions = self.model.sample_actions(
            images, image_masks, lang_tokens, lang_masks,
            subtask_tokens=subtask_tokens, subtask_masks=subtask_masks,
        )
        actions = actions[:, :, : self.action_dim]
        predictions = self.action_normalizer.unnormalize(actions)

        output_tensors: dict[DataType, list] = {}
        for data_type in self.ordered_output_data_types:
            if data_type == DataType.SUBTASK_LANGUAGE:
                # Build BatchedLanguageData from the subtask token IDs
                from neuracore_types.batched_nc_data.batched_language_data import (
                    BatchedLanguageData,
                )
                # Add a singleton T dimension to match (B, T, L) shape
                ids_3d = subtask_tokens.unsqueeze(1)
                mask_3d = subtask_masks.unsqueeze(1).to(dtype=torch.float32)
                output_tensors[data_type] = [
                    BatchedLanguageData(input_ids=ids_3d, attention_mask=mask_3d)
                ]
                continue
            start_idx, end_idx = self.output_dims[data_type]
            output_width = end_idx - start_idx
            dt_preds = predictions[:, :, start_idx:end_idx]

            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                output_tensors[data_type] = [
                    BatchedJointData(value=dt_preds[:, :, i : i + 1])
                    for i in range(output_width)
                ]
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                output_tensors[data_type] = [
                    BatchedParallelGripperOpenAmountData(
                        open_amount=dt_preds[:, :, i : i + 1]
                    )
                    for i in range(output_width)
                ]
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        return output_tensors
```

Note: `self.output_dims` is built only for the joint/gripper outputs in `__init__` (the existing pi05 code skips SUBTASK_LANGUAGE because it's not joint/gripper). Verify this; if SUBTASK_LANGUAGE accidentally enters that loop, skip it explicitly.

- [ ] **Step 3: Run test**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_inference_outputs.py -v
```

- [ ] **Step 4: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/pi05.py \
        tests/unit/ml/algorithms/pi05_full/test_inference_outputs.py
git commit -m "feat: emit predicted subtask as SUBTASK_LANGUAGE inference output"
```

---

## Phase F — Stage 2 freeze tightening (1 task)

### Task 14: Make `finetune_action_expert_only` set `requires_grad=False` on VLM

**Files:**
- Modify: `neuracore/ml/algorithms/pi05_full/pi05.py`
- Test: `tests/unit/ml/algorithms/pi05_full/test_finetune_action_expert_only.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/ml/algorithms/pi05_full/test_finetune_action_expert_only.py
"""Stage 2: VLM params get requires_grad=False, weights unchanged after step."""

import torch

from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full


def test_action_expert_only_freezes_vlm(model_init_description_with_subtask):
    model = Pi05Full(
        model_init_description_with_subtask,
        finetune_action_expert_only=True,
    )
    expert_param_substr = (
        "gemma_expert", "action_in_proj", "action_out_proj",
        "time_mlp_in", "time_mlp_out",
    )
    for name, param in model.model.named_parameters():
        if any(s in name for s in expert_param_substr):
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name


def test_action_expert_only_step_does_not_change_vlm(
    model_init_description_with_subtask, synthetic_training_batch,
):
    model = Pi05Full(
        model_init_description_with_subtask,
        finetune_action_expert_only=True,
    )
    optimizer = model.configure_optimizers()[0]
    vlm_param = (
        model.model.paligemma_with_expert.paligemma.language_model.layers[0]
        .self_attn.k_proj.weight
    )
    before = vlm_param.detach().clone()

    out = model.training_step(synthetic_training_batch)
    out.losses["loss"].backward()
    optimizer.step()

    assert torch.allclose(vlm_param.detach(), before)
```

- [ ] **Step 2: Update `_setup_optimizer_param_groups`**

In `pi05.py`, find `_setup_optimizer_param_groups` and rewrite the action-expert-only branch:

```python
        ACTION_EXPERT_PARAM_NAMES = [
            "gemma_expert", "action_in_proj", "action_out_proj",
            "time_mlp_in", "time_mlp_out",
        ]
        VISION_ENCODER_PARAM_NAMES = ["vision_tower", "multi_modal"]

        if self.finetune_action_expert_only:
            for name, param in self.model.named_parameters():
                if any(p in name for p in ACTION_EXPERT_PARAM_NAMES):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.param_groups = [{"params": params, "lr": self.optimizer_lr}]
        elif self.finetune_vision_encoder_and_action_expert:
            allowed = ACTION_EXPERT_PARAM_NAMES + VISION_ENCODER_PARAM_NAMES
            for name, param in self.model.named_parameters():
                if any(p in name for p in allowed):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.param_groups = [{"params": params, "lr": self.optimizer_lr}]
        else:
            self.param_groups = [{
                "params": list(self.model.parameters()),
                "lr": self.optimizer_lr,
            }]
```

- [ ] **Step 3: Run test**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_finetune_action_expert_only.py -v
```

- [ ] **Step 4: Commit**

```bash
git add neuracore/ml/algorithms/pi05_full/pi05.py \
        tests/unit/ml/algorithms/pi05_full/test_finetune_action_expert_only.py
git commit -m "fix: requires_grad=False on frozen VLM params in pi05_full

When finetune_action_expert_only or finetune_vision_encoder_and_action_expert
is set, set requires_grad=False on the frozen parameters so backward saves
compute and memory rather than computing gradients that get discarded."
```

---

## Phase G — Integration tests and polish (3 tasks)

### Task 15: Stage configurations integration test

**Files:**
- Test: `tests/unit/ml/algorithms/pi05_full/test_stage_configurations.py`

- [ ] **Step 1: Write the test**

```python
# tests/unit/ml/algorithms/pi05_full/test_stage_configurations.py
"""Stage 1, Stage 2, and joint training configurations all run."""

from neuracore.ml.algorithms.pi05_full.pi05 import Pi05Full


def test_stage1_config_runs_training_step(
    model_init_description_with_subtask, synthetic_training_batch,
):
    model = Pi05Full(
        model_init_description_with_subtask,
        flow_matching_loss_weight=0.0,
        subtask_loss_weight=10.0,
        fast_token_loss_weight=1.0,
        finetune_action_expert_only=False,
        knowledge_insulation=False,
    )
    out = model.training_step(synthetic_training_batch)
    assert out.losses["loss"].item() != 0


def test_stage2_config_runs_training_step(
    model_init_description_with_subtask, synthetic_training_batch,
):
    model = Pi05Full(
        model_init_description_with_subtask,
        flow_matching_loss_weight=1.0,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        finetune_action_expert_only=True,
    )
    out = model.training_step(synthetic_training_batch)
    assert out.losses["loss"].item() != 0


def test_joint_config_runs_training_step(
    model_init_description_with_subtask, synthetic_training_batch,
):
    model = Pi05Full(model_init_description_with_subtask)  # defaults = joint
    out = model.training_step(synthetic_training_batch)
    assert out.losses["loss"].item() != 0
```

- [ ] **Step 2: Run**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_stage_configurations.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/unit/ml/algorithms/pi05_full/test_stage_configurations.py
git commit -m "test: stage 1/2/joint configurations all run a training step"
```

### Task 16: Full suite green check + test fixture cleanup

- [ ] **Step 1: Run the full pi05_full suite**

```bash
pytest tests/unit/ml/algorithms/pi05_full/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the broader Neuracore test suite as a regression check**

```bash
pytest tests/unit/ml/ -v
```

Expected: existing tests for pi05 (and other algorithms) still pass — no regression from the rename or `_setup_optimizer_param_groups` change. Note: `pi05` does not use `requires_grad=False` so existing pi05 tests should be unaffected.

- [ ] **Step 3: If anything is red, debug; otherwise commit any test-fixture cleanup**

```bash
git status
git diff
```

If you made fixture changes during the suite run, commit them:

```bash
git add tests/
git commit -m "test: tighten fixtures and shared helpers"
```

### Task 17: Add slow-marked pretrained-load smoke test (optional, gated)

**Files:**
- Test: `tests/unit/ml/algorithms/pi05_full/test_pretrained_load.py`

- [ ] **Step 1: Write the test**

```python
# tests/unit/ml/algorithms/pi05_full/test_pretrained_load.py
"""Slow: load lerobot/pi05_base into Pi05Full and verify state dict."""

import os

import pytest
import torch

from neuracore.ml.algorithms.pi05_full.modules import PI05FullPolicy
from neuracore.ml.algorithms.pi05_full.utils import PI05FullConfig


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run.",
)
def test_load_lerobot_pi05_base():
    cfg = PI05FullConfig(
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
        chunk_size=50, device="cpu", dtype="float32",
    )
    model = PI05FullPolicy.from_pretrained(
        "lerobot/pi05_base", config=cfg,
    )
    expert = model.paligemma_with_expert.gemma_expert
    assert any(p.numel() > 0 for p in expert.parameters())
    # Tied embedding sanity check
    pg = model.paligemma_with_expert.paligemma
    embed_w = pg.model.language_model.embed_tokens.weight
    head_w = pg.lm_head.weight
    assert embed_w.data_ptr() == head_w.data_ptr()
```

- [ ] **Step 2: Verify the gating works**

```bash
pytest tests/unit/ml/algorithms/pi05_full/test_pretrained_load.py -v
# expects: 1 SKIPPED

RUN_SLOW_TESTS=1 pytest tests/unit/ml/algorithms/pi05_full/test_pretrained_load.py -v
# expects: 1 PASSED (after weight download)
```

- [ ] **Step 3: Commit**

```bash
git add tests/unit/ml/algorithms/pi05_full/test_pretrained_load.py
git commit -m "test: gated slow smoke test for lerobot/pi05_base load"
```

---

## Self-review

Coverage map (spec → plan task):

- §1 Summary differences → covered by all later tasks
- §2 Architecture file map → Tasks 1–14 (file edits as listed)
- §2 Class hierarchy renames → Task 2
- §2 Three-loss return contract → Task 6 (forward); Task 12 (training_step)
- §3 Training data flow → Tasks 6, 10, 11, 12
- §3 Inference data flow → Tasks 7, 8, 13
- §3 Subtask caching deferred → noted in plan-wide notes
- §4 KI implementation → Tasks 4, 5, 6
- §5 Two-stage training via configs → Tasks 14, 15
- §6 Construction validation → Task 9
- §6 Training-time validation (truncation, etc.) → covered inside Tasks 3, 10, 11
- §6 Inference-time validation → covered in Task 13's forward
- §7 Testing strategy → all `test_*.py` tasks (4, 7, 9, 10, 12, 13, 14, 15, 17)
- §8 Defaults → Task 2

Placeholder scan: no TBDs remain.

Type consistency: `PI05FullPolicy.forward` consistently takes `(images, img_masks, lang_tokens, lang_masks, subtask_tokens, subtask_masks, fast_tokens, fast_masks, actions, ...)`. `_embed_prefix` returns 4-tuple `(embs, pad_masks, att_masks, segments)` everywhere. `generate_subtask_tokens` returns `(tokens, masks)` consistently across Task 7 and Task 8/13.

Open implementation-time questions (called out in the spec, repeated here for the executing engineer):
- FAST tokenizer pinning to a specific HF revision — left to default-resolved for v1.
- Stage 1 perf shortcut (skip suffix forward when flow weight is 0) — not done; revisit only if Stage 1 is too slow.
- Optimal `max_subtask_tokens` and `max_fast_tokens` defaults — current values match lerobot, may be tuned after first training run.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-06-pi05-full-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
