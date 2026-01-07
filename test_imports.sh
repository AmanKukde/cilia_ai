#!/bin/bash
# Quick test script to verify imports work correctly after fixing SAM2 namespace conflict

echo "Testing cilia_ai package structure..."
echo "======================================"

# Check Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Test import structure
python << 'EOF'
import sys
import os

print("\nChecking directory structure...")
print("================================")

# Check directories exist
dirs = ['cilia_utils', 'models']
for d in dirs:
    if os.path.isdir(d):
        print(f"✓ {d}/ directory exists")
    else:
        print(f"❌ {d}/ directory not found")
        sys.exit(1)

# Check required files
files = [
    'cilia_utils/__init__.py',
    'cilia_utils/utils.py',
    'cilia_utils/plotting_utils.py',
    'models/__init__.py',
    'models/unet.py',
    'models/model_factory.py',
    'train_segmentation.py',
    'finetune_sam2_multichannel.py'
]

print("\nChecking required files...")
print("==========================")
for f in files:
    if os.path.exists(f):
        print(f"✓ {f}")
    else:
        print(f"❌ {f} not found")
        sys.exit(1)

# Check no conflicting sam2 directory
print("\nChecking for conflicts...")
print("=========================")
if os.path.isdir('sam2'):
    print("❌ WARNING: sam2/ directory exists - this will conflict with SAM2 package!")
    print("   Please remove it: rm -rf sam2/")
    sys.exit(1)
else:
    print("✓ No conflicting sam2/ directory")

print("\n✓ All structural checks passed!")
EOF

# Try to import (will fail if dependencies not installed, but that's OK)
echo ""
echo "Testing Python imports..."
echo "========================="

python << 'EOF'
import sys

# Test imports (may fail due to missing dependencies)
imports_ok = True

try:
    from cilia_utils.utils import normalize_channel
    print("✓ cilia_utils.utils imports successfully")
except ImportError as e:
    if "numpy" in str(e) or "torch" in str(e):
        print(f"⚠ cilia_utils.utils needs dependencies: {e}")
    else:
        print(f"❌ cilia_utils.utils import failed: {e}")
        imports_ok = False

try:
    from cilia_utils.plotting_utils import plot_training_metrics
    print("✓ cilia_utils.plotting_utils imports successfully")
except ImportError as e:
    if "numpy" in str(e) or "matplotlib" in str(e):
        print(f"⚠ cilia_utils.plotting_utils needs dependencies: {e}")
    else:
        print(f"❌ cilia_utils.plotting_utils import failed: {e}")
        imports_ok = False

try:
    from models.model_factory import ModelFactory
    print("✓ models.model_factory imports successfully")
except ImportError as e:
    if "torch" in str(e) or "transformers" in str(e):
        print(f"⚠ models.model_factory needs dependencies: {e}")
    else:
        print(f"❌ models.model_factory import failed: {e}")
        imports_ok = False

try:
    from models.unet import UNet
    print("✓ models.unet imports successfully")
except ImportError as e:
    if "torch" in str(e):
        print(f"⚠ models.unet needs dependencies: {e}")
    else:
        print(f"❌ models.unet import failed: {e}")
        imports_ok = False

if not imports_ok:
    print("\n❌ Import structure has errors!")
    sys.exit(1)
else:
    print("\n✓ Import structure is correct!")
    print("  (Some imports may need dependencies installed)")
EOF

echo ""
echo "======================================"
echo "Test complete!"
echo ""
echo "To install dependencies:"
echo "  pip install torch torchvision tifffile numpy matplotlib scikit-image tqdm"
echo ""
echo "For Hugging Face models:"
echo "  pip install transformers"
echo ""
echo "For SAM2:"
echo "  # Follow instructions in FINETUNE_README.md"
