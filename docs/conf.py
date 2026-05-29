"""Sphinx configuration for boolean-bestiary docs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "boolean-bestiary"
author = "boolean-bestiary contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "classic"
html_logo = "_static/bb-logo.jpg"
html_static_path = ["_static"]

os.environ.setdefault("PYTHONHASHSEED", "0")
