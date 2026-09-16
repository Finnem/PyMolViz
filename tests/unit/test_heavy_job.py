"""Heavy job confirmation helper (no Qt UI)."""

from pymolviz.util.solvent_surface import surface_job_fingerprint
from pymolviz.wizards.builders.heavy_job import ask_heavy_job


def test_ask_heavy_job_allows_light_job_without_ui():
    job = {"heavy": False, "n_atoms": 1, "quality": 1, "seconds": 0.1, "voxels": 100}
    allowed, ok_fp, denied_fp = ask_heavy_job(None, job, title="Test")
    assert allowed is True
    assert ok_fp is None
    assert denied_fp is None


def test_ask_heavy_job_denies_when_previously_denied():
    job = {"heavy": True, "n_atoms": 50, "quality": 5, "seconds": 120.0, "voxels": 1_000_000}
    fp = surface_job_fingerprint(job)
    allowed, ok_fp, denied_fp = ask_heavy_job(
        None, job, title="Test", previous_denied=fp,
    )
    assert allowed is False
    assert ok_fp is None
    assert denied_fp is None


def test_ask_heavy_job_allows_when_previously_ok():
    job = {"heavy": True, "n_atoms": 50, "quality": 5, "seconds": 120.0, "voxels": 1_000_000}
    fp = surface_job_fingerprint(job)
    allowed, ok_fp, denied_fp = ask_heavy_job(
        None, job, title="Test", previous_ok=fp,
    )
    assert allowed is True
    assert ok_fp is None
    assert denied_fp is None


def test_ask_heavy_job_custom_fingerprint():
    from pymolviz.fields.convert import explicit_convert_job_fingerprint

    job = {
        "heavy": True,
        "algorithm": "EXPLICIT_ISO",
        "voxels": 1000,
        "cubes": 50,
        "triangles": 200,
        "seconds": 2.0,
    }
    fp = explicit_convert_job_fingerprint(job)
    allowed, ok_fp, denied_fp = ask_heavy_job(
        None, job, title="Test", previous_denied=fp, fingerprint=explicit_convert_job_fingerprint,
    )
    assert allowed is False
    assert ok_fp is None
    assert denied_fp is None
