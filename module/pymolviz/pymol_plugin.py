"""PyMOL plugin entry: Plugins menu item and ``pmvw`` after startup.

Add to ``~/.pymolrc`` (or ``~/.pymolrc.py``):

    import pymolviz.pymol_plugin as pmv_plugin
    pmv_plugin.__init_plugin__()
"""

from __future__ import annotations

_PLUGIN_INITIALIZED = False


def __init_plugin__(app=None):
    """Called by PyMOL's plugin loader or from ``.pymolrc``."""
    global _PLUGIN_INITIALIZED
    if _PLUGIN_INITIALIZED:
        return
    from .runtime.integration import install
    from .wizard import register_wizard_commands, start_wizard

    register_wizard_commands()
    install()
    try:
        from pymol.plugins import addmenuitemqt

        addmenuitemqt("PyMOLViz", start_wizard)
    except Exception:
        pass
    _PLUGIN_INITIALIZED = True
