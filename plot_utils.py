"""Utilities for consistent matplotlib font configuration across the app.

This module centralises the logic for discovering a Chinese-capable font and
applying the required matplotlib rcParams so that charts rendered from scripts,
GUI widgets, or mplfinance all share the same configuration and avoid garbled
legend/label text.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
from matplotlib import font_manager as mpl_font_manager

# Candidate font files/families. We keep Windows defaults first, but also allow
# custom additions (e.g. a bundled fonts directory in the repo).
_FONT_CANDIDATE_PATHS: List[Path] = [
    Path(r"C:/Windows/Fonts/msyh.ttc"),
    Path(r"C:/Windows/Fonts/msyhbd.ttc"),
    Path(r"C:/Windows/Fonts/simhei.ttf"),
    Path(r"C:/Windows/Fonts/Deng.ttf"),
]

# We also search for repo-provided fonts (if the user chooses to drop one under
# `fonts/` it will be auto-detected).
_PROJECT_FONT_DIR = Path(__file__).resolve().parent / "fonts"
if _PROJECT_FONT_DIR.exists() and _PROJECT_FONT_DIR.is_dir():
    for font_file in _PROJECT_FONT_DIR.glob("*.ttf"):
        _FONT_CANDIDATE_PATHS.append(font_file)
    for font_file in _PROJECT_FONT_DIR.glob("*.otf"):
        _FONT_CANDIDATE_PATHS.append(font_file)

# Fallback family names in case font files cannot be located directly.
_FALLBACK_FAMILIES: List[str] = [
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
]


@lru_cache(maxsize=1)
def _discover_chinese_fonts() -> List[str]:
    discovered: List[str] = []

    for path in _FONT_CANDIDATE_PATHS:
        if not path.exists():
            continue
        try:
            mpl_font_manager.fontManager.addfont(str(path))
            family = mpl_font_manager.FontProperties(fname=str(path)).get_name()
            if family and family not in discovered:
                discovered.append(family)
        except Exception:
            continue

    # Add fallbacks that might already exist in the system font cache.
    for fam in _FALLBACK_FAMILIES:
        try:
            found = mpl_font_manager.findfont(fam, fallback_to_default=False)
        except Exception:
            found = None
        if found and os.path.exists(found) and fam not in discovered:
            discovered.append(fam)

    # Ensure we always return at least one placeholder to keep rcParams happy.
    if not discovered:
        discovered.append("SimHei")

    return discovered


def ensure_chinese_fonts() -> None:
    """Configure matplotlib rcParams for Chinese rendering once per process."""

    families = _discover_chinese_fonts()
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = families
    matplotlib.rcParams["axes.unicode_minus"] = False


def get_chinese_font_prop() -> Optional[mpl_font_manager.FontProperties]:
    """Return a FontProperties instance that points to a Chinese-capable font."""

    families = _discover_chinese_fonts()
    for fam in families:
        try:
            font_path = mpl_font_manager.findfont(fam, fallback_to_default=False)
        except Exception:
            continue
        if font_path and os.path.exists(font_path):
            return mpl_font_manager.FontProperties(fname=font_path)
    return None


def get_chinese_rc_params() -> dict:
    """Return an rcParams snippet that enforces the discovered Chinese font."""

    ensure_chinese_fonts()
    return {
        "font.family": matplotlib.rcParams.get("font.family", "sans-serif"),
        "font.sans-serif": matplotlib.rcParams.get("font.sans-serif", []),
        "axes.unicode_minus": False,
    }


def apply_chinese_font(ax, *, title: bool = True, xlabel: bool = True, ylabel: bool = True) -> None:
    """Utility helper to set fontproperties on common Axes labels."""

    from matplotlib.axes import Axes  # Local import to avoid circular refs

    if not isinstance(ax, Axes):
        return

    prop = get_chinese_font_prop()
    if prop is None:
        return

    if title and hasattr(ax, "get_title"):
        current = ax.get_title()
        if current:
            ax.set_title(current, fontproperties=prop)
    if xlabel and hasattr(ax, "get_xlabel"):
        current = ax.get_xlabel()
        if current:
            ax.set_xlabel(current, fontproperties=prop)
    if ylabel and hasattr(ax, "get_ylabel"):
        current = ax.get_ylabel()
        if current:
            ax.set_ylabel(current, fontproperties=prop)


__all__ = [
    "ensure_chinese_fonts",
    "get_chinese_font_prop",
    "get_chinese_rc_params",
    "apply_chinese_font",
]
