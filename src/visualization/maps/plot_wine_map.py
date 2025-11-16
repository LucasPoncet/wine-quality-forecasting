import logging
from pathlib import Path

import plotly.express as px
import polars as pl

logger = logging.getLogger(__name__)


def plot_wine_map(df: pl.DataFrame, out_html: Path) -> None:
    """Generate an interactive OpenStreetMap of wines."""
    pdf = df.drop_nulls(["latitude", "longitude"]).to_pandas()
    if pdf.empty:
        logger.warning("No coordinates to plot.")
        return

    fig = px.scatter_map(
        pdf,
        lat="latitude",
        lon="longitude",
        hover_name="Wine",
        zoom=4,
        height=700,
        title="Carte des vins — Coordonnées finales",
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(out_html)
    logger.info("Wine map exported to %s", out_html)
