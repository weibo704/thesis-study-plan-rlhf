"""WSGI entry point for the container.

`app.server.main()` builds the catalog and then starts Flask's development
server, which is not what should face students. Gunicorn imports `app`
from here instead, so the catalog is loaded once at import time and, with
--preload, shared across workers by copy-on-write rather than loaded once
per worker.

THESIS_PUBLIC=1 serves only the student study and the health check. The
researcher console exposes the reward weights, the full catalog browser
and the results tables; none of that should be one URL guess away from a
participant, and leaving it reachable would also let anyone spend the
API budget through the console's own recommend path.
"""
from __future__ import annotations

import os

from app import server, study

app = server.app

server.CAT = server.Catalog()
study.init_app(app, server.CAT)

if os.environ.get("THESIS_PUBLIC", "").lower() in ("1", "true", "yes"):
    _ALLOWED = {"study.page", "study.start", "study.choose", "study.reason",
                "study.health", "static",
                # Reachable but token-gated; see study._inspect_allowed.
                "study.inspect_page", "study.inspect_data"}

    @app.before_request
    def _public_only():
        from flask import request
        if request.endpoint and request.endpoint not in _ALLOWED:
            return ("Not found", 404)

print(f"[wsgi] {len(server.CAT.courses):,} arms · "
      f"public={'yes' if os.environ.get('THESIS_PUBLIC') else 'no'}", flush=True)
