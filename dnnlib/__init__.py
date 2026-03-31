# This directory should contain files from the CB-AE repo.
# Run: python setup.py --copy-vendor /path/to/posthoc-generative-cbm
# Or manually copy:
#   cbae_repo/dnnlib/__init__.py  ->  dnnlib/__init__.py
#   cbae_repo/dnnlib/util.py      ->  dnnlib/util.py
raise ImportError(
    "dnnlib/ is empty. Copy it from the CB-AE repo.\n"
    "Run: python setup.py --copy-vendor /path/to/posthoc-generative-cbm"
)
