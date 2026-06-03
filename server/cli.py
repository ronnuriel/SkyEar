from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SkyEar central API server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("server.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
