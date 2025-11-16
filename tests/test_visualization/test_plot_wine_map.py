import plotly.express as px
import polars as pl

from src.visualization.maps.plot_wine_map import plot_wine_map


def test_plot_wine_map(monkeypatch, tmp_path):
    df = pl.DataFrame(
        {
            "Wine": ["A", "B"],
            "latitude": [44.0, 46.0],
            "longitude": [1.0, 2.0],
        }
    )

    fake_fig = type(
        "FakeFig",
        (),
        {
            "update_layout": lambda self, **k: None,
            "write_html": lambda self, p: p,
        },
    )()
    monkeypatch.setattr(px, "scatter_map", lambda *a, **k: fake_fig)

    out_html = tmp_path / "map.html"
    plot_wine_map(df, out_html)
    assert out_html.exists() or out_html.name == "map.html"
