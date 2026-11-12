"""Local graphical interface (web page served from the loopback interface).

Start it with ``scientificlab-gui`` or ``python -m scientificlab.gui``.
"""

from scientificlab.gui.server import create_server, main, serve

__all__ = ["create_server", "main", "serve"]
