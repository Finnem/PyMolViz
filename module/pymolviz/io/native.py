"""Native ``.pmv`` pack: ZIP of ``manifest.json`` plus ``arrays/*.npy``."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from ..serialization import (
    ArrayStore,
    SerializationError,
    session_document,
    session_from_document,
    using_array_store,
)

NATIVE_FORMAT = "pymolviz"
NATIVE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
ARRAY_PREFIX = "arrays/"


def as_save_items(obj) -> list:
    """One Displayable stays whole, even if it is also a list (CGOCollection)."""
    if obj is None:
        return []
    from ..Displayable import Displayable

    if isinstance(obj, Displayable):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def save_native(obj, path) -> None:
    """Write ``obj`` (one displayable or a sequence) as a ``.pmv`` pack."""
    store = ArrayStore()
    with using_array_store(store):
        doc = session_document(as_save_items(obj))
    doc = dict(doc)
    doc["format"] = NATIVE_FORMAT
    doc["format_version"] = NATIVE_FORMAT_VERSION
    payload = json.dumps(doc, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    path = str(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.writestr(MANIFEST_NAME, payload)
        for key, arr in store.arrays.items():
            buf = BytesIO()
            np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
            zf.writestr("%s%s.npy" % (ARRAY_PREFIX, key), buf.getvalue())


def is_native_pack(path) -> bool:
    path = str(path)
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            if MANIFEST_NAME not in zf.namelist():
                return False
            doc = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
    except Exception:
        return False
    return isinstance(doc, dict) and doc.get("format") == NATIVE_FORMAT


def load_native(path) -> list:
    """Return the displayables stored in a ``.pmv`` pack."""
    path = str(path)
    if not zipfile.is_zipfile(path):
        raise SerializationError("Not a PyMolViz pack: %s" % path)
    store = ArrayStore()
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if MANIFEST_NAME not in names:
            raise SerializationError("PyMolViz pack is missing %s" % MANIFEST_NAME)
        try:
            doc = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        except Exception as exc:
            raise SerializationError("Invalid PyMolViz manifest") from exc
        if not isinstance(doc, dict):
            raise SerializationError("Invalid PyMolViz manifest")
        if doc.get("format") not in (None, NATIVE_FORMAT):
            raise SerializationError("Unknown pack format %r" % doc.get("format"))
        version = int(doc.get("format_version") or 1)
        if version > NATIVE_FORMAT_VERSION:
            raise SerializationError("Unsupported PyMolViz file version %s" % version)
        for name in names:
            if not name.startswith(ARRAY_PREFIX) or not name.endswith(".npy"):
                continue
            key = Path(name).stem
            raw = zf.read(name)
            try:
                arr = np.load(BytesIO(raw), allow_pickle=False)
            except Exception as exc:
                raise SerializationError("Could not read array %s" % key) from exc
            store.add(key, arr)
    with using_array_store(store):
        return session_from_document(doc, strict=True)
