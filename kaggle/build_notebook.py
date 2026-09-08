"""
Turn run_on_kaggle.py into a Kaggle notebook.

The Python file stays the single source of truth — it is testable, diffable and
readable in an editor. This wraps it into an .ipynb so that what lands on Kaggle
is a real notebook you can open, read and re-run cell by cell, rather than an
opaque script kernel.

Split into a few cells rather than one, so that when something fails you can see
which stage failed and re-run just that part.
"""

import json
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).parent / "run_on_kaggle.py"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fashionsearch-gpu.ipynb")

INTRO = """# FashionSearch — the GPU half

Embedding 10,000 products on Databricks serverless CPU takes about 45 minutes.
Here on a T4 it takes two or three.

This notebook:

1. downloads the anchor–positive pairs from Hugging Face
2. optionally fine-tunes the encoder with contrastive learning
3. embeds the catalogue and the evaluation queries
4. **registers the encoder in Unity Catalog** — the step Databricks refuses,
   because registering from outside uses a different upload path
5. uploads the vectors to a Databricks Volume
6. calls back into GitHub, which starts the Databricks half

Generated from `kaggle/run_on_kaggle.py`. Edit that file, not this notebook —
this one is overwritten on every push.

**Before running:** Accelerator must be GPU T4. Credentials arrive through the
attached private dataset, so nothing else needs configuring.
"""


def cells():
    source = SRC.read_text()
    # Drop the __main__ guard; a notebook calls main() explicitly at the end.
    source = re.sub(r'\nif __name__ == "__main__":\n    main\(\)\n?', "\n", source)

    yield {"cell_type": "markdown", "metadata": {}, "source": INTRO.splitlines(True)}

    yield {
        "cell_type": "markdown", "metadata": {},
        "source": ["## Check the accelerator\n\n",
                   "If this says CPU, stop and set Accelerator to GPU T4 in the ",
                   "sidebar — the run would take as long as the Databricks path ",
                   "it is meant to replace.\n"],
    }
    yield {
        "cell_type": "code", "metadata": {"trusted": True},
        "execution_count": None, "outputs": [],
        "source": [
            "import torch\n",
            "print('GPU available:', torch.cuda.is_available())\n",
            "if torch.cuda.is_available():\n",
            "    print('device:', torch.cuda.get_device_name(0))\n",
            "else:\n",
            "    print('NO GPU — set Accelerator to GPU T4 in the sidebar.')\n",
        ],
    }

    yield {
        "cell_type": "markdown", "metadata": {},
        "source": ["## The pipeline code\n\n",
                   "Everything from `kaggle/run_on_kaggle.py`.\n"],
    }
    yield {
        "cell_type": "code", "metadata": {"trusted": True},
        "execution_count": None, "outputs": [],
        "source": source.splitlines(True),
    }

    yield {
        "cell_type": "markdown", "metadata": {},
        "source": ["## Run it\n\n",
                   "Roughly 3 minutes on a T4 without fine-tuning, 20 with.\n"],
    }
    yield {
        "cell_type": "code", "metadata": {"trusted": True},
        "execution_count": None, "outputs": [],
        "source": ["main()\n"],
    }


notebook = {
    "cells": list(cells()),
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUT} — {len(notebook['cells'])} cells")
