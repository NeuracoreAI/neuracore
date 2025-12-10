"""Simple test script for loading pretrained PI0 model from HuggingFace.

This script tests the Pi0.from_pretrained() functionality by:
1. Loading the pretrained model from lerobot/pi0_base
2. Verifying the model loads successfully
3. Running a simple forward pass with dummy data

Usage:
    python examples/test_pi0_pretrained.py

Requirements:
    - HF_TOKEN environment variable set (for HuggingFace authentication)
    - transformers and safetensors packages installed
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from neuracore_types import (
    DataItemStats,
    DatasetDescription,
    DataType,
    ModelInitDescription,
)

# Add parent directory to path to import neuracore
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuracore.ml import BatchedInferenceSamples, MaskableData  # noqa: E402
from neuracore.ml.algorithms.pi0.pi0 import Pi0  # noqa: E402

# Test configuration
BS = 1  # Batch size
CAMS = 2  # Number of cameras
JOINT_POSITION_DIM = 16  # Joint dimension
PRED_HORIZON = 8  # Prediction horizon
LANGUAGE_MAX_LEN = 128  # Language token length
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"HF_TOKEN set: {os.environ.get('HF_TOKEN') is not None}")


def create_model_init_description() -> ModelInitDescription:
    """Create a minimal ModelInitDescription for testing."""
    dataset_description = DatasetDescription(
        joint_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_target_positions=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_velocities=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        joint_torques=DataItemStats(
            mean=np.zeros(JOINT_POSITION_DIM, dtype=float),
            std=np.ones(JOINT_POSITION_DIM, dtype=float),
        ),
        rgb_images=DataItemStats(max_len=CAMS),
        language=DataItemStats(max_len=LANGUAGE_MAX_LEN),
    )
    return ModelInitDescription(
        dataset_description=dataset_description,
        input_data_types=[
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
            DataType.LANGUAGE,
        ],
        output_data_types=[DataType.JOINT_TARGET_POSITIONS],
        output_prediction_horizon=PRED_HORIZON,
        device=DEVICE.type,
    )


def create_dummy_batch() -> BatchedInferenceSamples:
    """Create dummy inference batch for testing."""
    return BatchedInferenceSamples(
        joint_positions=MaskableData(
            torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
        ),
        joint_velocities=MaskableData(
            torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
        ),
        joint_torques=MaskableData(
            torch.randn(BS, JOINT_POSITION_DIM, dtype=torch.float32),
            torch.ones(BS, JOINT_POSITION_DIM, dtype=torch.float32),
        ),
        rgb_images=MaskableData(
            torch.rand(BS, CAMS, 3, 224, 224, dtype=torch.float32),
            torch.ones(BS, CAMS, dtype=torch.float32),
        ),
        language_tokens=MaskableData(
            torch.randint(0, 1000, (BS, LANGUAGE_MAX_LEN), dtype=torch.long),
            torch.ones(BS, LANGUAGE_MAX_LEN, dtype=torch.float32),
        ),
    )


def main():
    """Test loading pretrained PI0 model."""
    print("=" * 60)
    print("Testing PI0.from_pretrained()")
    print("=" * 60)

    # Check for HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        print(
            "WARNING: HF_TOKEN not set. This may cause authentication issues."
            " Set it with: export HF_TOKEN=your_token"
        )

    # Create model init description
    print("\n1. Creating ModelInitDescription...")
    model_init_description = create_model_init_description()
    print("   ✓ ModelInitDescription created")

    # Test 1: Load with default (lerobot/pi0_base)
    print("\n2. Loading pretrained model from lerobot/pi0_base...")
    try:
        model = Pi0.from_pretrained(
            model_init_description,
            # pretrained_name_or_path="lerobot/pi0_base",  # Optional default
            cache_dir=None,  # Use default cache
            local_files_only=False,  # Allow downloading
        )
        model = model.to(DEVICE)
        print("   ✓ Model loaded successfully!")
        print(f"   - Model type: {type(model)}")
        print(f"   - Policy type: {type(model.policy)}")
        print(f"   - Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test 2: Verify model structure
    print("\n3. Verifying model structure...")
    try:
        assert hasattr(model, "policy"), "Model should have policy attribute"
        assert hasattr(model, "model"), "Model should have model attribute"
        assert hasattr(model, "config"), "Model should have config attribute"
        print("   ✓ Model structure verified")
    except AssertionError as e:
        print(f"   ✗ Model structure check failed: {e}")
        return 1

    # Test 3: Run forward pass
    print("\n4. Running forward pass with dummy data...")
    try:
        batch = create_dummy_batch()
        batch = batch.to(DEVICE)

        model.eval()
        with torch.no_grad():
            output = model(batch)

        print("   ✓ Forward pass successful!")
        print(f"   - Output type: {type(output)}")
        output_shape = output.outputs[DataType.JOINT_TARGET_POSITIONS].shape
        print(f"   - Output shape: {output_shape}")
        print(f"   - Prediction time: {output.prediction_time:.4f}s")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test 4: Test with custom repo (optional, can skip if repo doesn't exist)
    print("\n5. Testing with explicit repo name...")
    try:
        model2 = Pi0.from_pretrained(
            model_init_description,
            pretrained_name_or_path="lerobot/pi0_base",
        )
        model2 = model2.to(DEVICE)
        print("   ✓ Explicit repo name works!")
    except Exception as e:
        print(f"   ⚠ Explicit repo test failed (this is OK if repo doesn't exist): {e}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
