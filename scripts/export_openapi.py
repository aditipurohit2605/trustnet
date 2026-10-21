"""
Utility script to generate and export the OpenAPI specification for TrustNet API.

Outputs:
- docs/openapi.json
"""

import json
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trustnet.api.app import create_app


def export_openapi(output_path: str = "docs/openapi.json") -> None:
    """Instantiates the FastAPI app and dumps the OpenAPI schema as JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    app = create_app()
    openapi_schema = app.openapi()

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"[TrustNet] OpenAPI specification successfully written to: {out_file.resolve()}")


if __name__ == "__main__":
    export_openapi()
