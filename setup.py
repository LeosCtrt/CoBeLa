#!/usr/bin/env python3
"""
CoBELa Environment Setup
=========================
Installs dependencies, downloads pretrained weights, patches StyleGAN2
for modern PyTorch, and verifies the full pipeline.

Usage:
    python setup.py                  # full setup
    python setup.py --check          # verify only
    python setup.py --skip-cub       # skip CUB weights
    python setup.py --copy-vendor /path/to/cbae_repo   # auto-copy dnnlib + torch_utils
"""

import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ── 1. Install pip dependencies ──────────────────────────────────────

def install_deps():
    deps = ["ninja", "pyspng", "pytorch-fid", "omegaconf", "lmdb", "gdown"]
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], check=False)
    print("[OK] Dependencies installed.")


# ── 2. Copy vendor files from CB-AE repo ─────────────────────────────

def copy_vendor(cbae_dir):
    """Copy dnnlib/ and torch_utils/ from a CB-AE clone."""
    import shutil

    for dirname in ["dnnlib", "torch_utils"]:
        src = os.path.join(cbae_dir, dirname)
        dst = os.path.join(PROJECT_ROOT, dirname)
        if not os.path.isdir(src):
            print(f"[ERROR] {src} not found in CB-AE repo.")
            return False
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        n_files = sum(len(f) for _, _, f in os.walk(dst))
        print(f"[OK] Copied {dirname}/ ({n_files} files)")

    return True


# ── 3. Apply compatibility patches ───────────────────────────────────

def apply_patches():
    """Patch numpy deprecations + StyleGAN2 CUDA ops for modern PyTorch."""
    import numpy as np
    if not hasattr(np, "float"):
        np.float = np.float64
        np.int = np.int_
        np.complex = np.complex128
        np.object = np.object_
        np.bool = np.bool_

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    import warnings
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
    warnings.filterwarnings("ignore", message=".*is_autocast_enabled.*")

    # Force pure-PyTorch fallback for StyleGAN2 custom ops
    try:
        from torch_utils.ops import bias_act, upfirdn2d
        bias_act._init = lambda: False
        upfirdn2d._init = lambda: False
        try:
            from torch_utils.ops import filtered_lrelu
            filtered_lrelu._init = lambda: False
        except ImportError:
            pass
        print("[OK] StyleGAN2 custom ops patched → pure PyTorch fallback.")
    except ImportError:
        print("[WARN] torch_utils not found. Copy vendor files first (see README).")


# ── 4. Download weights ──────────────────────────────────────────────

def download_weights(skip_cub=False):
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    downloads = [
        {
            "name": "StyleGAN2 CelebA-HQ",
            "filename": "stylegan2-celebahq-256x256.pkl",
            "cmd": ["wget", "-q", "--show-progress",
                    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl",
                    "-O"],
        },
        {
            "name": "ResNet18 CelebA-HQ classifiers",
            "filename": "celebahq_Smiling_rn18_conclsf.pth",  # check file
            "cmd_zip": ["gdown", "--id", "1xbR7MbERV7wMnU4WcsNSDriYXBqsy_jZ", "-O"],
            "zip": True,
        },
    ]

    if not skip_cub:
        downloads.extend([
            {
                "name": "StyleGAN2 CUB",
                "filename": "stylegan2-cub-256x256.pkl",
                "cmd": ["gdown", "--id", "1sW7WgvUFH2REZPQx88BjFneoItP9C0XB", "-O"],
            },
            {
                "name": "ResNet50 CUB classifiers",
                "filename": "cub_Black_bill_color_rn50_conclsf.pth",  # check file
                "cmd_zip": ["gdown", "--id", "1vW5Q41FGHXdTqbraz54AXQ2uoBKispLD", "-O"],
                "zip": True,
            },
        ])

    for dl in downloads:
        check_path = os.path.join(ckpt_dir, dl["filename"])
        if os.path.exists(check_path):
            print(f"[SKIP] {dl['name']} already present.")
            continue

        print(f"[DL] Downloading {dl['name']}...")

        if dl.get("zip"):
            zip_path = os.path.join(ckpt_dir, "_tmp.zip")
            subprocess.run(dl["cmd_zip"] + [zip_path], check=True)
            subprocess.run(["unzip", "-q", "-o", zip_path, "-d", ckpt_dir], check=True)
            os.remove(zip_path)
        else:
            out_path = os.path.join(ckpt_dir, dl["filename"])
            subprocess.run(dl["cmd"] + [out_path], check=True)

        print(f"[OK] {dl['name']} ready.")


# ── 5. Verify ─────────────────────────────────────────────────────────

def verify():
    import torch

    print("\n" + "=" * 60)
    print("  CoBELa Environment Verification")
    print("=" * 60)

    # Platform
    print(f"\n  Python:    {sys.version.split()[0]}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {torch.version.cuda or 'N/A'}")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU RAM:   {mem:.1f} GB")
    else:
        print("  GPU:       NONE (training will be very slow!)")

    # Vendor files
    dnnlib_ok = os.path.exists(os.path.join(PROJECT_ROOT, "dnnlib", "util.py"))
    tu_ok = os.path.exists(os.path.join(PROJECT_ROOT, "torch_utils", "persistence.py"))
    print(f"\n  dnnlib/:       {'✓' if dnnlib_ok else '✗ MISSING — run: python setup.py --copy-vendor /path/to/cbae'}")
    print(f"  torch_utils/:  {'✓' if tu_ok else '✗ MISSING — run: python setup.py --copy-vendor /path/to/cbae'}")

    # Weights
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    for name, fname in [
        ("StyleGAN2 CelebA-HQ", "stylegan2-celebahq-256x256.pkl"),
        ("StyleGAN2 CUB", "stylegan2-cub-256x256.pkl"),
    ]:
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            print(f"  {name}: {os.path.getsize(path)/1e6:.0f} MB ✓")
        else:
            print(f"  {name}: ✗ not found")

    # Classifier files
    celeba_clfs = [f for f in os.listdir(ckpt_dir) if "rn18_conclsf" in f] if os.path.isdir(ckpt_dir) else []
    cub_clfs = [f for f in os.listdir(ckpt_dir) if "rn50_conclsf" in f] if os.path.isdir(ckpt_dir) else []
    print(f"  CelebA-HQ classifiers: {len(celeba_clfs)} files {'✓' if len(celeba_clfs) >= 8 else '✗'}")
    print(f"  CUB classifiers:       {len(cub_clfs)} files {'✓' if len(cub_clfs) >= 10 else '(optional)'}")

    # Test StyleGAN2 loading
    if dnnlib_ok and tu_ok and torch.cuda.is_available():
        stylegan_path = os.path.join(ckpt_dir, "stylegan2-celebahq-256x256.pkl")
        if os.path.exists(stylegan_path):
            print("\n  Testing StyleGAN2 load...")
            try:
                apply_patches()
                import pickle
                with open(stylegan_path, "rb") as f:
                    G = pickle.load(f)["G_ema"].cuda().eval()
                z = torch.randn(1, G.z_dim, device="cuda")
                w = G.mapping(z, None)
                img = G.synthesis(w, noise_mode="const")
                print(f"    g1: z{tuple(z.shape)} → w{tuple(w.shape)} ✓")
                print(f"    g2: w{tuple(w.shape)} → img{tuple(img.shape)} ✓")
                del G, z, w, img
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    FAILED: {e}")

    print("\n" + "=" * 60)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoBELa setup")
    parser.add_argument("--check", action="store_true", help="Verify only")
    parser.add_argument("--skip-cub", action="store_true", help="Skip CUB weights")
    parser.add_argument("--copy-vendor", type=str, default=None,
                        help="Path to CB-AE repo clone to copy dnnlib/ and torch_utils/")
    parser.add_argument("--auto-vendor", action="store_true",
                        help="Auto-clone CB-AE repo to /tmp, copy vendor files, clean up")
    args = parser.parse_args()

    if args.check:
        apply_patches()
        verify()
        return

    # Handle vendor files
    vendor_ok = os.path.exists(os.path.join(PROJECT_ROOT, "dnnlib", "util.py"))
    if args.copy_vendor:
        copy_vendor(args.copy_vendor)
    elif args.auto_vendor or not vendor_ok:
        print("[vendor] Auto-cloning CB-AE repo to copy dnnlib/ and torch_utils/...")
        import shutil
        tmp_repo = "/tmp/cbae_repo"
        if os.path.exists(tmp_repo):
            shutil.rmtree(tmp_repo)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm.git",
            tmp_repo,
        ], check=True)
        copy_vendor(tmp_repo)
        shutil.rmtree(tmp_repo)
        print("[OK] Vendor files copied and temp repo cleaned up.")

    install_deps()
    apply_patches()
    download_weights(skip_cub=args.skip_cub)
    verify()


if __name__ == "__main__":
    main()
