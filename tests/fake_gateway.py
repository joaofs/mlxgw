"""A gateway process with the fake engine, for lifecycle tests (no MLX, no network).

    python tests/fake_gateway.py mlxgw.server --port N

The `mlxgw.server` argument is only there so the command line looks like a real
gateway's to control.is_our_gateway.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

from mlxgw.admin import Admin  # noqa: E402
from mlxgw.server import create_app  # noqa: E402
from tests.conftest import FakeEngine, make_settings  # noqa: E402

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1])
    settings, engine = make_settings(), FakeEngine()
    admin = Admin(settings, engine, local=lambda repo: Path("/nonexistent"), remote_size=lambda repo: None)
    uvicorn.run(create_app(engine, settings, admin), host="127.0.0.1", port=port, log_level="info")
