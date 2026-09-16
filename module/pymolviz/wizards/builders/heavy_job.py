"""Shared heavy preview/commit confirmation for surfaces and field maps."""

from __future__ import annotations

from typing import Optional, Tuple

from ...util.solvent_surface import (
    confirm_heavy_surface_job,
    format_heavy_surface_message,
    surface_job_fingerprint,
)
from ..pick import overlay_question, qt_modules


def _confirm_heavy_job(job, previous_ok, previous_denied, fingerprint):
    fp = fingerprint(job)
    if not job.get("heavy"):
        return "allow", fp
    if previous_ok == fp:
        return "allow", fp
    if previous_denied == fp:
        return "deny", fp
    return "ask", fp


def ask_heavy_job(
    page,
    job,
    *,
    title: str,
    message: Optional[str] = None,
    previous_ok=None,
    previous_denied=None,
    no_qt_default: bool = True,
    fingerprint=surface_job_fingerprint,
) -> Tuple[bool, object, object]:
    """Ask once per job fingerprint; return (allowed, new_ok_fp, new_denied_fp).

    ``new_ok_fp`` / ``new_denied_fp`` are the fingerprint when the user accepts or
    rejects; unchanged (``None``) when allow/deny came from cache without UI.
    """
    if fingerprint is surface_job_fingerprint:
        decision, fingerprint = confirm_heavy_surface_job(
            job, previous_ok=previous_ok, previous_denied=previous_denied,
        )
    else:
        decision, fingerprint = _confirm_heavy_job(
            job, previous_ok, previous_denied, fingerprint,
        )
    if decision == "allow":
        return True, None, None
    if decision == "deny":
        return False, None, None
    _, _, QtWidgets = qt_modules()
    if QtWidgets is None or page is None:
        if no_qt_default:
            return True, fingerprint, None
        return False, None, None
    text = message if message is not None else format_heavy_surface_message(job)
    result = overlay_question(
        page,
        title,
        text,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )
    if result == QtWidgets.QMessageBox.Yes:
        return True, fingerprint, None
    return False, None, fingerprint
