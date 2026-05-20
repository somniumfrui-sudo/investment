"""Streamlit Cloud / GitHub デプロイ用エントリ（ルートから起動）。"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TOOL = _ROOT / "investment_tool"
if str(_TOOL) not in sys.path:
    sys.path.insert(0, str(_TOOL))

runpy.run_path(str(_TOOL / "main.py"), run_name="__main__")
