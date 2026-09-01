"""Semantic point references that resolve to XYZ in a PyMOL context."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

XYZ = Tuple[float, float, float]


class PointUnresolvedError(RuntimeError):
    """Raised when an atom/pseudoatom cannot be resolved. ``last_xyz`` is left in place."""


class PointSource:
    """Abstract reference to a 3D position."""

    type_name = "PointSource"

    def resolve(self, context) -> XYZ:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def has_dynamic_source(self) -> bool:
        return False

    @property
    def last_xyz(self) -> Optional[XYZ]:
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "PointSource":
        from .serialization import point_source_from_dict

        return point_source_from_dict(data)


class FixedPoint(PointSource):
  type_name = "FixedPoint"

  def __init__(self, xyz: Sequence[float]) -> None:
      arr = np.asarray(xyz, dtype=float).reshape(3)
      self.x = float(arr[0])
      self.y = float(arr[1])
      self.z = float(arr[2])

  def resolve(self, context=None, remember=True) -> XYZ:
      return (self.x, self.y, self.z)

  @property
  def last_xyz(self) -> XYZ:
      return (self.x, self.y, self.z)

  def to_dict(self) -> dict:
      return {"type": self.type_name, "xyz": [self.x, self.y, self.z]}


class AtomPoint(PointSource):
    """Reference to a protein/ligand atom by object + ID, with residue fallback."""

    type_name = "AtomPoint"

    def __init__(
        self,
        object: str,
        atom_id: int,
        *,
        chain: str = "",
        resi: str = "",
        name: str = "",
        last_xyz: Optional[Sequence[float]] = None,
    ) -> None:
        self.object = str(object)
        self.atom_id = int(atom_id)
        self.chain = str(chain or "")
        self.resi = str(resi or "")
        self.name = str(name or "")
        if last_xyz is not None:
            arr = np.asarray(last_xyz, dtype=float).reshape(3)
            self._last_xyz: Optional[XYZ] = (float(arr[0]), float(arr[1]), float(arr[2]))
        else:
            self._last_xyz = None

    def has_dynamic_source(self) -> bool:
        return True

    @property
    def last_xyz(self) -> Optional[XYZ]:
        return self._last_xyz

    def _lookup_xyz(self, cmd, state) -> Optional[XYZ]:
        exprs = [
            'object "%s" and id %d' % (self.object, self.atom_id),
            'object "%s" and index %d' % (self.object, self.atom_id),
            "(%s)`%d" % (self.object, self.atom_id),
        ]
        fallback_parts = ['object "%s"' % self.object]
        if self.chain:
            fallback_parts.append('chain "%s"' % self.chain)
        if self.resi not in (None, ""):
            fallback_parts.append("resi %s" % self.resi)
        if self.name:
            fallback_parts.append('name "%s"' % self.name)
        exprs.append(" and ".join(fallback_parts))
        atoms = []
        for expr in exprs:
            for iterate_expr in (
                "atoms.append([x, y, z])",
                "atoms.append([x,y,z])",
            ):
                atoms.clear()
                try:
                    if state:
                        cmd.iterate_state(state, expr, iterate_expr, space={"atoms": atoms})
                    else:
                        cmd.iterate(expr, iterate_expr, space={"atoms": atoms})
                    if atoms:
                        return (
                            float(atoms[0][0]),
                            float(atoms[0][1]),
                            float(atoms[0][2]),
                        )
                except Exception:
                    continue
        return None

    def resolve(self, context, remember=True) -> XYZ:
        if context is None:
            if self._last_xyz is not None:
                return self._last_xyz
            raise PointUnresolvedError(
                "AtomPoint %r/%d cannot resolve without a PyMOL context"
                % (self.object, self.atom_id)
            )
        xyz = self._lookup_xyz(context.cmd, getattr(context, "state", 1) or 1)
        if xyz is None:
            raise PointUnresolvedError(
                "Atom not found: object=%r id=%d (fallback chain=%r resi=%r name=%r); "
                "last-known xyz %s retained"
                % (self.object, self.atom_id, self.chain, self.resi, self.name, self._last_xyz)
            )
        if remember:
            self._last_xyz = xyz
        return xyz

    def to_dict(self) -> dict:
        data = {
            "type": self.type_name,
            "object": self.object,
            "atom_id": self.atom_id,
            "chain": self.chain,
            "resi": self.resi,
            "name": self.name,
        }
        if self._last_xyz is not None:
            data["last_xyz"] = list(self._last_xyz)
        return data


class PseudoAtomPoint(PointSource):
    """Reference to a PyMolViz-owned pseudoatom object."""

    type_name = "PseudoAtomPoint"

    def __init__(
        self,
        object: str,
        atom_id: int = 0,
        *,
        last_xyz: Optional[Sequence[float]] = None,
    ) -> None:
        self.object = str(object)
        self.atom_id = int(atom_id)
        if last_xyz is not None:
            arr = np.asarray(last_xyz, dtype=float).reshape(3)
            self._last_xyz: Optional[XYZ] = (float(arr[0]), float(arr[1]), float(arr[2]))
        else:
            self._last_xyz = None

    def has_dynamic_source(self) -> bool:
        return True

    @property
    def last_xyz(self) -> Optional[XYZ]:
        return self._last_xyz

    def resolve(self, context, remember=True) -> XYZ:
        if context is None:
            if self._last_xyz is not None:
                return self._last_xyz
            raise PointUnresolvedError(
                "PseudoAtomPoint %r cannot resolve without a PyMOL context" % self.object
            )
        cmd = context.cmd
        state = getattr(context, "state", 1) or 1
        expr = 'object "%s"' % self.object
        if self.atom_id:
            expr += " and id %d" % self.atom_id
        atoms = []
        try:
            if state:
                cmd.iterate_state(state, expr, "atoms.append([x,y,z])", space={"atoms": atoms})
            else:
                cmd.iterate(expr, "atoms.append([x,y,z])", space={"atoms": atoms})
            if atoms:
                xyz = (float(atoms[0][0]), float(atoms[0][1]), float(atoms[0][2]))
                if remember:
                    self._last_xyz = xyz
                return xyz
        except Exception:
            pass
        raise PointUnresolvedError(
            "Pseudoatom not found: object=%r; last-known xyz %s retained"
            % (self.object, self._last_xyz)
        )

    def to_dict(self) -> dict:
        data = {"type": self.type_name, "object": self.object, "atom_id": self.atom_id}
        if self._last_xyz is not None:
            data["last_xyz"] = list(self._last_xyz)
        return data


def as_point_source(value: Union[PointSource, Sequence[float], np.ndarray, Any]) -> PointSource:
    if isinstance(value, PointSource):
        return value
    arr = np.asarray(value, dtype=float).reshape(3)
    return FixedPoint(arr)


def resolve_xyz(source: Union[PointSource, Sequence[float], np.ndarray], context=None) -> XYZ:
    return as_point_source(source).resolve(context)


def point_sources_from_sequence(
    values: Sequence[Union[PointSource, Sequence[float], np.ndarray]],
) -> list:
    return [as_point_source(v) for v in values]


_SOURCE_ATTRS = (
    "position",
    "center",
    "start",
    "end",
    "center_position",
    "outer_start",
)
_SOURCE_LIST_ATTRS = (
    "start_sources",
    "end_sources",
    "vertex_sources",
    "point_sources",
    "path_sources",
)


def iter_point_sources(obj):
    """Yield PointSources stored on a displayable or collection."""
    children = getattr(obj, "__iter__", None)
    if children is not None and type(obj).__name__ == "CGOCollection":
        for child in obj:
            yield from iter_point_sources(child)
        return
    for attr in _SOURCE_ATTRS:
        val = getattr(obj, attr, None)
        if isinstance(val, PointSource):
            yield val
    starts = getattr(obj, "_start_sources", None) or getattr(obj, "starts", None)
    ends = getattr(obj, "_end_sources", None) or getattr(obj, "ends", None)
    for seq in (starts, ends):
        if seq and not isinstance(seq, PointSource):
            try:
                for src in seq:
                    if isinstance(src, PointSource):
                        yield src
            except TypeError:
                pass
    for attr in _SOURCE_LIST_ATTRS:
        val = getattr(obj, attr, None)
        if val:
            for src in val:
                if isinstance(src, PointSource):
                    yield src


def has_dynamic_sources(obj) -> bool:
    return any(src.has_dynamic_source() for src in iter_point_sources(obj))
