from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn

from . import APP_DESC, APP_NAME, AUTHOR, BUILD_TIME, COMMIT, VERSION
from .app import create_app
from .config import load_config
from .database import close_db, init_db
from .services import init_services, save_runtime_state, shutdown_services

BANNER = r"""
   ____       _                        
  / __ \____ / /_____  ____  __  _______
 / / / / __ `/ __/ __ \/ __ \/ / / / ___/
/ /_/ / /_/ / /_/ /_/ / /_/ / /_/ (__  ) 
\____/\__,_/\__/\____/ .___/\__,_/____/  
                    /_/                   
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=APP_NAME, description=APP_DESC)
    parser.add_argument("--config", "-c", default="data/config.json", help="設定檔路徑")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("start", help="啟動服務")
    sub.add_parser("version", help="顯示版本資訊")
    return parser


async def _startup(config_path: str):
    config = load_config(config_path)
    logging.basicConfig(level=getattr(logging, config.log.level.upper(), logging.INFO))
    init_db(config.database.type, config.database.path)
    await init_services()
    return config


async def _shutdown() -> None:
    await shutdown_services()
    save_runtime_state()
    close_db()


def run_start(config_path: str) -> None:
    print(BANNER)
    config = asyncio.run(_startup(config_path))
    app = create_app(config)

    try:
        uvicorn.run(app, host=config.server.host, port=config.server.port, log_level=config.log.level.lower())
    finally:
        asyncio.run(_shutdown())


def print_version() -> None:
    print(f"Version: {VERSION}")
    print(f"Commit: {COMMIT}")
    print(f"BuildTime: {BUILD_TIME}")
    print(f"Author: {AUTHOR}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print_version()
        return
    run_start(args.config)


if __name__ == "__main__":
    main()
