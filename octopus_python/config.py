from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class LogConfig:
    level: str = "info"


@dataclass
class DatabaseConfig:
    type: str = "sqlite"
    path: str = "data/data.db"


@dataclass
class AppConfig:
    server: ServerConfig
    log: LogConfig
    database: DatabaseConfig


DEFAULT_CONFIG = AppConfig(server=ServerConfig(), log=LogConfig(), database=DatabaseConfig())
APP_CONFIG: AppConfig = DEFAULT_CONFIG
CONFIG_PATH = "data/config.json"


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _apply_env(data: dict[str, Any]) -> None:
    prefix = "OCTOPUS_"
    mapping = {
        "OCTOPUS_SERVER_HOST": ("server", "host"),
        "OCTOPUS_SERVER_PORT": ("server", "port"),
        "OCTOPUS_DATABASE_TYPE": ("database", "type"),
        "OCTOPUS_DATABASE_PATH": ("database", "path"),
        "OCTOPUS_LOG_LEVEL": ("log", "level"),
    }
    for env_key, path in mapping.items():
        if env_key not in os.environ:
            continue
        cur: dict[str, Any] = data
        for part in path[:-1]:
            cur = cur.setdefault(part, {})
        value: Any = os.environ[env_key]
        if env_key.endswith("_PORT"):
            value = int(value)
        cur[path[-1]] = value

    # Generic OCTOPUS_SECTION_FIELD support for simple two-level keys.
    for env_key, value in os.environ.items():
        if not env_key.startswith(prefix) or env_key in mapping:
            continue
        parts = env_key[len(prefix) :].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field = parts
        if section in {"server", "database", "log"}:
            cur = data.setdefault(section, {})
            cur[field] = int(value) if field == "port" else value


def load_config(path: str = "data/config.json") -> AppConfig:
    global APP_CONFIG, CONFIG_PATH
    CONFIG_PATH = path
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(DEFAULT_CONFIG)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            _deep_update(data, loaded)
    else:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    _apply_env(data)
    APP_CONFIG = AppConfig(
        server=ServerConfig(**data.get("server", {})),
        log=LogConfig(**data.get("log", {})),
        database=DatabaseConfig(**data.get("database", {})),
    )
    return APP_CONFIG


def is_debug() -> bool:
    return os.environ.get("OCTOPUS_DEBUG", "").lower() == "true"
