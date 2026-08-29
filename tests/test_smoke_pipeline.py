from hackathon_everest.pipeline import run_pipeline


def test_smoke_pipeline_writes_complete_artifacts(tmp_path):
    manifest = run_pipeline(
        {
            "seed": 23,
            "episodes": 30,
            "fields": 10,
            "n_estimators": 8,
            "replay_steps": 2,
            "benchmark_fields": 1,
        },
        tmp_path,
    )
    assert manifest["sensor_channels"] == 19
    assert len(manifest["config_sha256"]) == 64
    assert set(manifest["split_field_ids"]["train"]).isdisjoint(manifest["split_field_ids"]["test"])
    for artifact in manifest["artifacts"].values():
        assert __import__("pathlib").Path(artifact).exists()
    replay = __import__("json").loads((tmp_path / "replay.json").read_text())
    evidence = replay["cross_foot_evidence"]
    assert evidence["before_left_contact"] != evidence["after_left_contact"]
