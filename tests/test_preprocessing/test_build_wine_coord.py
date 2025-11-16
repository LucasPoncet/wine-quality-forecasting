import polars as pl

import src.preprocessing.build_wines_coord as pipeline


def test_build_wines_coord_executes(monkeypatch, tmp_path):
    # Patch helpers to avoid heavy ops
    monkeypatch.setattr(
        pipeline,
        "load_aoc",
        lambda *a, **k: pl.DataFrame({"denom_clean": ["a"], "lat": [45.0], "lon": [2.0]}),
    )
    monkeypatch.setattr(
        pipeline,
        "load_vivino_dataset",
        lambda *a, **k: pl.DataFrame(
            {"region_clean": ["a"], "latitude": [None], "longitude": [None], "Wine": ["Test"]}
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "attach_coords",
        lambda v, a: v.with_columns(pl.lit(45.0).alias("latitude"), pl.lit(2.0).alias("longitude")),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_barycentres",
        lambda v: pl.DataFrame({"region_clean": ["a"], "cent_lat": [45.0], "cent_lon": [2.0]}),
    )
    monkeypatch.setattr(pipeline, "inject_centroids", lambda *a, **k: a[0])
    monkeypatch.setattr(pipeline, "plot_wine_map", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "attach_weather", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "recap", lambda *a, **k: None)

    args = pipeline.parse_args()
    args.output_dir = tmp_path
    pipeline.main()  # should complete end-to-end without errors
