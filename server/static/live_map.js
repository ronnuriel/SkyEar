const SkyEarLiveMap = (() => {
  const TILE_SIZE = 256;
  const EARTH_RADIUS_M = 6378137;
  const OSM_TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

  function queryOptions(search = window.location.search) {
    const params = new URLSearchParams(search);
    return {
      mode: params.get("mode"),
      lat: numberOrNull(params.get("lat")),
      lon: numberOrNull(params.get("lon")),
      zoom: numberOrNull(params.get("zoom")),
      preset: params.get("preset"),
    };
  }

  function numberOrNull(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function normalizeConfig(payload = {}, options = {}) {
    const liveMap = payload.live_map || payload || {};
    const config = {
      mode: "schematic",
      tile_url: null,
      attribution: null,
      default_latitude: null,
      default_longitude: null,
      default_zoom: 13,
      allow_online_tiles: false,
      warning: null,
      ...liveMap,
    };
    if (options.mode) config.mode = String(options.mode).toLowerCase();
    if (!["schematic", "geo"].includes(config.mode)) config.mode = "schematic";
    if (options.lat !== null && options.lat !== undefined) config.default_latitude = options.lat;
    if (options.lon !== null && options.lon !== undefined) config.default_longitude = options.lon;
    if (options.zoom !== null && options.zoom !== undefined) config.default_zoom = options.zoom;
    if (!config.tile_url && config.mode === "geo" && config.allow_online_tiles) {
      config.tile_url = OSM_TILE_URL;
      config.attribution = config.attribution || "OpenStreetMap contributors";
    }
    return config;
  }

  function latLonToPixel(lat, lon, zoom) {
    const sinLat = Math.sin((Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI) / 180);
    const scale = TILE_SIZE * 2 ** zoom;
    return {
      x: ((lon + 180) / 360) * scale,
      y: (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale,
    };
  }

  function pixelToLatLon(x, y, zoom) {
    const scale = TILE_SIZE * 2 ** zoom;
    const lon = (x / scale) * 360 - 180;
    const n = Math.PI - (2 * Math.PI * y) / scale;
    const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
    return { lat, lon };
  }

  function boundsFromPoints(points) {
    if (!points.length) return null;
    return {
      minLat: Math.min(...points.map((point) => point.lat)),
      maxLat: Math.max(...points.map((point) => point.lat)),
      minLon: Math.min(...points.map((point) => point.lon)),
      maxLon: Math.max(...points.map((point) => point.lon)),
    };
  }

  class SchematicRenderer {
    constructor({ state, els }) {
      this.state = state;
      this.els = els;
      this.mode = "schematic";
    }

    configure() {
      this.els.mapWrap.classList.remove("geo-mode");
      this.els.tileLayer.replaceChildren();
      this.els.attribution.textContent = "";
      return true;
    }

    updateBounds(points) {
      if (!points.length) return;
      let minLat = Math.min(...points.map((p) => p.lat));
      let maxLat = Math.max(...points.map((p) => p.lat));
      let minLon = Math.min(...points.map((p) => p.lon));
      let maxLon = Math.max(...points.map((p) => p.lon));
      const latPad = Math.max(0.0007, (maxLat - minLat) * 0.18);
      const lonPad = Math.max(0.0007, (maxLon - minLon) * 0.18);
      minLat -= latPad;
      maxLat += latPad;
      minLon -= lonPad;
      maxLon += lonPad;
      if (!this.state.bounds) {
        this.state.bounds = { minLat, maxLat, minLon, maxLon };
        return;
      }
      this.state.bounds.minLat = Math.min(this.state.bounds.minLat, minLat);
      this.state.bounds.maxLat = Math.max(this.state.bounds.maxLat, maxLat);
      this.state.bounds.minLon = Math.min(this.state.bounds.minLon, minLon);
      this.state.bounds.maxLon = Math.max(this.state.bounds.maxLon, maxLon);
    }

    applyAutoView() {
      this.state.zoom = 1;
      this.state.panX = 0;
      this.state.panY = 0;
    }

    project(lat, lon) {
      const b = this.state.bounds || {
        minLat: lat - 0.001,
        maxLat: lat + 0.001,
        minLon: lon - 0.001,
        maxLon: lon + 0.001,
      };
      const x = ((lon - b.minLon) / Math.max(1e-9, b.maxLon - b.minLon)) * 1000;
      const y = 1000 - ((lat - b.minLat) / Math.max(1e-9, b.maxLat - b.minLat)) * 1000;
      return { x, y };
    }

    radiusToSvg(radiusM) {
      if (!this.state.bounds) return 40;
      const midLat = (this.state.bounds.minLat + this.state.bounds.maxLat) / 2;
      const metersPerDegLat = 111320;
      const metersPerDegLon = 111320 * Math.max(0.1, Math.cos((midLat * Math.PI) / 180));
      const widthM = Math.max(1, (this.state.bounds.maxLon - this.state.bounds.minLon) * metersPerDegLon);
      const heightM = Math.max(1, (this.state.bounds.maxLat - this.state.bounds.minLat) * metersPerDegLat);
      return Math.max(12, (Math.min(Number(radiusM || 100), 350) / Math.max(widthM, heightM)) * 1000);
    }

    setWorldTransform() {
      this.els.world.setAttribute("transform", `translate(${this.state.panX} ${this.state.panY}) scale(${this.state.zoom})`);
    }

    wheel(event) {
      event.preventDefault();
      this.state.manualView = true;
      this.els.toggleFollow.checked = false;
      const delta = event.deltaY < 0 ? 1.12 : 0.9;
      this.state.zoom = Math.max(0.45, Math.min(8, this.state.zoom * delta));
      this.setWorldTransform();
    }

    drag(deltaX, deltaY) {
      this.state.panX = this.state.dragStart.panX + deltaX;
      this.state.panY = this.state.dragStart.panY + deltaY;
      this.setWorldTransform();
    }
  }

  class GeoRenderer {
    constructor({ state, els }) {
      this.state = state;
      this.els = els;
      this.mode = "geo";
      this.config = null;
    }

    configure(config) {
      if (!config.tile_url) return false;
      this.config = config;
      this.state.geo = {
        centerLat: numberOrNull(config.default_latitude) ?? 0,
        centerLon: numberOrNull(config.default_longitude) ?? 0,
        zoom: Number(config.default_zoom || 13),
      };
      this.els.mapWrap.classList.add("geo-mode");
      this.els.attribution.textContent = config.attribution || "";
      this.syncTiles();
      return true;
    }

    updateBounds(points) {
      this.state.geoBounds = boundsFromPoints(points);
    }

    applyAutoView(data, centerMode) {
      if (centerMode === "control") {
        const control = data?.map_state?.control_point;
        if (control?.latitude !== undefined && control?.longitude !== undefined) {
          this.state.geo.centerLat = Number(control.latitude);
          this.state.geo.centerLon = Number(control.longitude);
          this.syncTiles();
          return;
        }
      }
      const bounds = this.state.geoBounds;
      if (!bounds) {
        this.syncTiles();
        return;
      }
      const width = Math.max(1, this.els.map.clientWidth || 1000);
      const height = Math.max(1, this.els.map.clientHeight || 1000);
      let zoom = Math.max(1, Math.min(19, Number(this.state.geo.zoom || 13)));
      for (let candidate = 19; candidate >= 1; candidate -= 1) {
        const nw = latLonToPixel(bounds.maxLat, bounds.minLon, candidate);
        const se = latLonToPixel(bounds.minLat, bounds.maxLon, candidate);
        if (Math.abs(se.x - nw.x) <= width * 0.72 && Math.abs(se.y - nw.y) <= height * 0.72) {
          zoom = candidate;
          break;
        }
      }
      this.state.geo.zoom = zoom;
      this.state.geo.centerLat = (bounds.minLat + bounds.maxLat) / 2;
      this.state.geo.centerLon = (bounds.minLon + bounds.maxLon) / 2;
      this.syncTiles();
    }

    project(lat, lon) {
      const width = Math.max(1, this.els.map.clientWidth || 1000);
      const height = Math.max(1, this.els.map.clientHeight || 1000);
      const zoom = Number(this.state.geo.zoom || 13);
      const center = latLonToPixel(this.state.geo.centerLat, this.state.geo.centerLon, zoom);
      const point = latLonToPixel(lat, lon, zoom);
      return {
        x: 500 + (point.x - center.x) * (1000 / width),
        y: 500 + (point.y - center.y) * (1000 / height),
      };
    }

    radiusToSvg(radiusM) {
      const width = Math.max(1, this.els.map.clientWidth || 1000);
      const metersPerPixel =
        (Math.cos((Number(this.state.geo.centerLat || 0) * Math.PI) / 180) * 2 * Math.PI * EARTH_RADIUS_M) /
        (TILE_SIZE * 2 ** Number(this.state.geo.zoom || 13));
      return Math.max(8, (Math.min(Number(radiusM || 100), 350) / Math.max(0.01, metersPerPixel)) * (1000 / width));
    }

    setWorldTransform() {
      this.els.world.setAttribute("transform", "translate(0 0) scale(1)");
      this.syncTiles();
    }

    wheel(event) {
      event.preventDefault();
      this.state.manualView = true;
      this.els.toggleFollow.checked = false;
      const zoom = Number(this.state.geo.zoom || 13);
      this.state.geo.zoom = Math.max(1, Math.min(19, zoom + (event.deltaY < 0 ? 1 : -1)));
      this.syncTiles();
    }

    drag(deltaX, deltaY) {
      const zoom = Number(this.state.geo.zoom || 13);
      const center = latLonToPixel(this.state.dragStart.centerLat, this.state.dragStart.centerLon, zoom);
      const next = pixelToLatLon(center.x - deltaX, center.y - deltaY, zoom);
      this.state.geo.centerLat = next.lat;
      this.state.geo.centerLon = next.lon;
      this.syncTiles();
    }

    syncTiles() {
      const tileUrl = this.config?.tile_url;
      if (!tileUrl) return;
      const width = Math.max(1, this.els.map.clientWidth || 1000);
      const height = Math.max(1, this.els.map.clientHeight || 1000);
      const zoom = Math.round(Number(this.state.geo.zoom || 13));
      const center = latLonToPixel(this.state.geo.centerLat, this.state.geo.centerLon, zoom);
      const minTileX = Math.floor((center.x - width / 2) / TILE_SIZE) - 1;
      const maxTileX = Math.floor((center.x + width / 2) / TILE_SIZE) + 1;
      const minTileY = Math.floor((center.y - height / 2) / TILE_SIZE) - 1;
      const maxTileY = Math.floor((center.y + height / 2) / TILE_SIZE) + 1;
      const maxIndex = 2 ** zoom;
      const active = new Set();
      for (let x = minTileX; x <= maxTileX; x += 1) {
        const wrappedX = ((x % maxIndex) + maxIndex) % maxIndex;
        for (let y = minTileY; y <= maxTileY; y += 1) {
          if (y < 0 || y >= maxIndex) continue;
          const key = `${zoom}:${wrappedX}:${y}`;
          active.add(key);
          let tile = this.els.tileLayer.querySelector(`[data-tile-key="${key}"]`);
          if (!tile) {
            tile = document.createElement("img");
            tile.dataset.tileKey = key;
            tile.className = "map-tile";
            tile.alt = "";
            tile.decoding = "async";
            tile.onerror = () => {
              this.els.mapWarning.textContent = "Basemap unavailable - using schematic mode";
              this.els.mapWarning.hidden = false;
              if (typeof window.SkyEarLiveFallbackToSchematic === "function") {
                window.SkyEarLiveFallbackToSchematic();
              }
            };
            tile.src = tileUrl
              .replace("{s}", ["a", "b", "c"][Math.abs(wrappedX + y) % 3])
              .replace("{z}", String(zoom))
              .replace("{x}", String(wrappedX))
              .replace("{y}", String(y));
            this.els.tileLayer.appendChild(tile);
          }
          tile.style.left = `${x * TILE_SIZE - center.x + width / 2}px`;
          tile.style.top = `${y * TILE_SIZE - center.y + height / 2}px`;
        }
      }
      for (const tile of Array.from(this.els.tileLayer.querySelectorAll(".map-tile"))) {
        if (!active.has(tile.dataset.tileKey)) tile.remove();
      }
    }
  }

  function createRenderer(mode, deps) {
    return mode === "geo" ? new GeoRenderer(deps) : new SchematicRenderer(deps);
  }

  return {
    createRenderer,
    normalizeConfig,
    queryOptions,
    latLonToPixel,
    pixelToLatLon,
  };
})();
