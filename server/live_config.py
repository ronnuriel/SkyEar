from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


DEFAULT_LIVE_MAP_CONFIG: dict[str, Any] = {
    "mode": "schematic",
    "tile_url": None,
    "attribution": None,
    "default_latitude": None,
    "default_longitude": None,
    "default_zoom": 13,
    "allow_online_tiles": False,
}

DEFAULT_OSM_TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
DEFAULT_OSM_ATTRIBUTION = "OpenStreetMap contributors"


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _is_external_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(str(url))
    return parsed.scheme in {"http", "https"} or str(url).startswith("//")


def live_map_config_from_mapping(payload: dict[str, Any] | None) -> dict[str, Any]:
    section = dict((payload or {}).get("live_map") or {})
    config = {**DEFAULT_LIVE_MAP_CONFIG, **section}
    config["mode"] = str(config.get("mode") or "schematic").lower()
    if config["mode"] not in {"schematic", "geo"}:
        config["mode"] = "schematic"

    allow_online = bool(config.get("allow_online_tiles"))
    tile_url = config.get("tile_url")
    warning = None
    if not tile_url and config["mode"] == "geo" and allow_online:
        tile_url = DEFAULT_OSM_TILE_URL
        config["attribution"] = config.get("attribution") or DEFAULT_OSM_ATTRIBUTION
    if _is_external_url(tile_url) and not allow_online:
        warning = "External live_map.tile_url ignored because allow_online_tiles=false"
        tile_url = None
        config["mode"] = "schematic"

    config["tile_url"] = tile_url or None
    config["attribution"] = config.get("attribution") or ("Local tiles" if tile_url else None)
    config["online_tiles_allowed"] = allow_online
    config["warning"] = warning
    return config


def load_live_map_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = config_path or os.environ.get("SKYEAR_CONFIG") or "configs/config_station.yaml"
    return live_map_config_from_mapping(_load_yaml(path))
