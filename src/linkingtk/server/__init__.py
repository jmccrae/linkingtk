"""REST server wrapper for LinkingTK linkers.

See [create_app][linkingtk.server.app.create_app].
"""

from __future__ import annotations

from linkingtk.server.app import create_app

__all__ = ["create_app"]
