"""
CoBELa: Concept Bottlenecks on Energy Landscapes.

This package contains the from-scratch implementation of CoBELa.
External dependencies (dnnlib/, torch_utils/) live at the project root.
"""

import os
import sys
import warnings

# Ensure project root is on sys.path so dnnlib/torch_utils are importable
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_pkg_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Numpy compat for older StyleGAN2 code
import numpy as np
if not hasattr(np, "float"):
    np.float = np.float64
    np.int = np.int_
    np.complex = np.complex128
    np.object = np.object_
    np.bool = np.bool_

# Writable torch extensions cache
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

# Suppress harmless deprecation warnings from old StyleGAN2 code
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
warnings.filterwarnings("ignore", message=".*is_autocast_enabled.*")


def patch_stylegan2_ops():
    """Force pure-PyTorch fallback for StyleGAN2 custom CUDA ops."""
    try:
        from torch_utils.ops import bias_act, upfirdn2d
        bias_act._init = lambda: False
        upfirdn2d._init = lambda: False
        try:
            from torch_utils.ops import filtered_lrelu
            filtered_lrelu._init = lambda: False
        except ImportError:
            pass
    except ImportError:
        raise ImportError(
            "torch_utils not found. Copy it from the CB-AE repo.\n"
            "See README.md for instructions."
        )
