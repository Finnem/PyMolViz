"""ResolveContext: cmd + state for PointSource.resolve."""

from __future__ import annotations


class ResolveContext:
    def __init__(self, cmd=None, state=None):
        if cmd is None:
            try:
                from pymol import cmd as _cmd
                cmd = _cmd
            except Exception:
                cmd = None
        if state is None and cmd is not None:
            try:
                state = int(cmd.get_state())
            except Exception:
                state = 1
        self.cmd = cmd
        self.state = state or 1


def try_context():
    try:
        from pymol import cmd
        return ResolveContext(cmd)
    except Exception:
        return None
