import json
from pathlib import Path

import src.teacher as teacher_module
from src.config import Config
from src.teacher import ensure_teacher_pass


def install_fakes(monkeypatch, call_log):
    """
    Stand-ins for the two expensive steps (downloading/splitting the dataset,
    running Qwen2.5-7B-Instruct inference) so ensure_teacher_pass's skip/run
    branching can be tested without a GPU or a real model.
    """

    def fake_build_and_save_splits(config):
        call_log.append("build")
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        manifest = {
            "split_a": {"path": str(Path(config.data_dir) / "split_a.parquet")},
            "split_b": {"path": str(Path(config.data_dir) / "split_b.parquet")},
        }
        with open(Path(config.data_dir) / "manifest.json", "w") as f:
            json.dump(manifest, f)

    def fake_verify_manifest(manifest):
        return True

    def fake_run_teacher_pass(config):
        call_log.append("teacher")
        manifest_path = Path(config.data_dir) / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        teacher_logits_path = Path(config.data_dir) / "teacher_logits.parquet"
        teacher_logits_path.write_text("fake parquet bytes")
        manifest["teacher_logits"] = {"path": str(teacher_logits_path)}

        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

    monkeypatch.setattr(teacher_module, "build_and_save_splits", fake_build_and_save_splits)
    monkeypatch.setattr(teacher_module, "verify_manifest", fake_verify_manifest)
    monkeypatch.setattr(teacher_module, "run_teacher_pass", fake_run_teacher_pass)


def test_first_call_builds_splits_and_runs_teacher_pass(tmp_path, monkeypatch):
    call_log = []
    install_fakes(monkeypatch, call_log)

    config = Config(data_dir=str(tmp_path))
    ensure_teacher_pass(config)

    assert call_log == ["build", "teacher"]


def test_second_call_skips_both_steps_when_data_already_exists(tmp_path, monkeypatch):
    call_log = []
    install_fakes(monkeypatch, call_log)

    config = Config(data_dir=str(tmp_path))
    ensure_teacher_pass(config)
    call_log.clear()

    ensure_teacher_pass(config)

    assert call_log == [], "A second call with existing, verified data should do no work at all"


def test_force_reruns_both_steps_even_if_data_exists(tmp_path, monkeypatch):
    call_log = []
    install_fakes(monkeypatch, call_log)

    config = Config(data_dir=str(tmp_path))
    ensure_teacher_pass(config)
    call_log.clear()

    ensure_teacher_pass(config, force=True)

    assert call_log == ["build", "teacher"]


def test_missing_teacher_logits_reruns_only_the_teacher_pass(tmp_path, monkeypatch):
    """Splits already exist and verify fine, but the teacher pass never completed."""
    call_log = []
    install_fakes(monkeypatch, call_log)

    config = Config(data_dir=str(tmp_path))
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(config.data_dir) / "manifest.json", "w") as f:
        json.dump({
            "split_a": {"path": str(Path(config.data_dir) / "split_a.parquet")},
            "split_b": {"path": str(Path(config.data_dir) / "split_b.parquet")},
        }, f)

    ensure_teacher_pass(config)

    assert call_log == ["teacher"]
