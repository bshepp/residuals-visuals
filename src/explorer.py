"""
Tier 1 Atlas Explorer: a self-contained HTML page with a 3D UMAP scatter,
orbit/zoom/pan controls, and rich hover tooltips for every point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.atlas import CATEGORY_COLORS, FAMILY_CATEGORY


def _format_params(d: dict) -> str:
    if not d:
        return "—"
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def build_hover_text(catalog: pd.DataFrame) -> list[str]:
    """One HTML-formatted tooltip per row."""
    out = []
    for _, row in catalog.iterrows():
        category = FAMILY_CATEGORY.get(row["decomp_family"], "—")
        decomp_p = _format_params(row["decomp_params"])
        upsamp_p = _format_params(row["upsamp_params"])
        text = (
            f"<b>{row['decomp_family']}</b>  ({category})<br>"
            f"  {decomp_p}<br>"
            f"<b>upsampler:</b> {row['upsamp_method']}<br>"
            f"  {upsamp_p}<br>"
            f"<span style='font-size:10px;color:#888'>{row['filename']}</span>"
        )
        out.append(text)
    return out


def build_explorer(
    coords: np.ndarray,
    catalog: pd.DataFrame,
    output_path: Path,
    title: str = "RESIDUALS Atlas Explorer",
) -> None:
    """
    Build a self-contained HTML with a 3D scatter where each trace is one
    decomposition category (so legend toggles work cleanly).
    """
    families = catalog["decomp_family"].to_numpy()
    categories = np.array([FAMILY_CATEGORY.get(f, "classical") for f in families])
    hover_texts = build_hover_text(catalog)

    fig = go.Figure()

    for category, color in CATEGORY_COLORS.items():
        mask = categories == category
        if not mask.any():
            continue
        # Real data trace — small markers, hidden from legend
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=category,
                legendgroup=category,
                showlegend=False,
                marker=dict(
                    size=2.0,
                    color=color,
                    opacity=0.65,
                    line=dict(width=0),
                ),
                hovertext=[hover_texts[i] for i in np.flatnonzero(mask)],
                hoverinfo="text",
                hoverlabel=dict(
                    bgcolor="rgba(20,20,20,0.95)",
                    bordercolor=color,
                    font=dict(family="monospace", size=11, color="white"),
                    align="left",
                ),
            )
        )
        # Legend-only proxy trace — no rendered points, big swatch in the legend
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                name=f"{category} (n={int(mask.sum()):,})",
                legendgroup=category,
                showlegend=True,
                marker=dict(size=14, color=color, opacity=1.0, line=dict(width=0)),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=title + f"  ·  {len(coords):,} residuals  ·  3D UMAP of 40-dim signature",
            font=dict(family="sans-serif", size=14, color="white"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="u", showbackground=False, color="#888", gridcolor="#222"),
            yaxis=dict(title="v", showbackground=False, color="#888", gridcolor="#222"),
            zaxis=dict(title="w", showbackground=False, color="#888", gridcolor="#222"),
            bgcolor="black",
            aspectmode="data",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(family="sans-serif", color="white"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#444",
            borderwidth=1,
            x=0.01,
            y=0.99,
            font=dict(size=12, color="white"),
            itemsizing="constant",
            itemwidth=40,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Self-contained HTML — no external CDN, double-click to open
    fig.write_html(
        str(output_path),
        include_plotlyjs="inline",
        full_html=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["sendDataToCloud"],
            "toImageButtonOptions": {"format": "png", "filename": "atlas_explorer", "scale": 2},
        },
    )
    print(f"Wrote {output_path}  ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
