const SVG_NS = "http://www.w3.org/2000/svg";
const REFRESH_MS = 500;

const els = {
  map: document.getElementById("liveMap"),
  world: document.getElementById("worldLayer"),
  stations: document.getElementById("stationLayer"),
  tracks: document.getElementById("trackLayer"),
  sectors: document.getElementById("sectorLayer"),
  estimates: document.getElementById("trackEstimateLayer"),
  fusionLevel: document.getElementById("fusionLevel"),
  fusionConfidence: document.getElementById("fusionConfidence"),
  fusionInterpretation: document.getElementById("fusionInterpretation"),
  trackCount: document.getElementById("trackCount"),
  updateAge: document.getElementById("updateAge"),
  connectionState: document.getElementById("connectionState"),
  trackList: document.getElementById("trackList"),
  stationList: document.getElementById("stationList"),
  selectedDetails: document.getElementById("selectedDetails"),
  toggleSectors: document.getElementById("toggleSectors"),
  toggleTrackEstimates: document.getElementById("toggleTrackEstimates"),
  toggleFollow: document.getElementById("toggleFollow"),
};

const state = {
  refreshInFlight: false,
  lastData: null,
  selectedKey: null,
  bounds: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragStart: null,
  manualView: false,
  stationNodes: new Map(),
  trackNodes: new Map(),
  sectorNodes: new Map(),
  estimateNodes: new Map(),
};

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, String(value));
  }
  return el;
}

function fmt(value, digits = 1, fallback = "-") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return fallback;
  return Number(value).toFixed(digits);
}

function statusColor(station) {
  const health = String(station.health || station.alive_state || "").toLowerCase();
  const status = String(station.last_status || "").toLowerCase();
  if (health === "offline" || health === "stale" || health === "degraded" || health === "error") return "#747b85";
  if (status === "alert" || status === "drone_like") return "#d9443f";
  if (status === "suspect" || status === "calibrating") return "#d5aa32";
  return "#39b56a";
}

function trackColor(level) {
  const value = Number(level || 0);
  if (value >= 3) return "#d9443f";
  if (value >= 1) return "#d5aa32";
  return "#5aa2ff";
}

function estimateColor(estimate) {
  const confidence = Number(estimate.confidence || estimate.track_confidence || 0);
  const quality = String(estimate.bearing_geometry_quality || "").toLowerCase();
  if (confidence >= 0.65 && Number(estimate.level || 0) >= 2 && quality !== "poor") return "rgba(217,68,63,0.28)";
  if (confidence >= 0.45 && quality !== "poor") return "rgba(213,170,50,0.24)";
  return "rgba(140,140,140,0.22)";
}

function pointFromItem(item) {
  const lat = item.latitude ?? item.lat ?? item?.estimated_source?.latitude;
  const lon = item.longitude ?? item.lon ?? item?.estimated_source?.longitude;
  if (lat === null || lat === undefined || lon === null || lon === undefined) return null;
  return { lat: Number(lat), lon: Number(lon) };
}

function collectPoints(data) {
  const points = [];
  for (const station of data?.map_state?.stations || []) {
    const point = pointFromItem(station);
    if (point) points.push(point);
  }
  for (const estimate of data?.map_state?.track_geo_estimates || []) {
    const point = pointFromItem(estimate);
    if (point) points.push(point);
  }
  for (const track of data?.map_state?.tracks || []) {
    const point = pointFromItem(track);
    if (point) points.push(point);
  }
  for (const cue of data?.map_state?.bearing_cues || []) {
    for (const point of cue.sector_polygon || []) {
      if (point.latitude !== undefined && point.longitude !== undefined) {
        points.push({ lat: Number(point.latitude), lon: Number(point.longitude) });
      }
    }
  }
  return points.filter((point) => Number.isFinite(point.lat) && Number.isFinite(point.lon));
}

function updateBounds(points) {
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
  if (!state.bounds) {
    state.bounds = { minLat, maxLat, minLon, maxLon };
    return;
  }
  state.bounds.minLat = Math.min(state.bounds.minLat, minLat);
  state.bounds.maxLat = Math.max(state.bounds.maxLat, maxLat);
  state.bounds.minLon = Math.min(state.bounds.minLon, minLon);
  state.bounds.maxLon = Math.max(state.bounds.maxLon, maxLon);
}

function project(lat, lon) {
  const b = state.bounds || { minLat: lat - 0.001, maxLat: lat + 0.001, minLon: lon - 0.001, maxLon: lon + 0.001 };
  const x = ((lon - b.minLon) / Math.max(1e-9, b.maxLon - b.minLon)) * 1000;
  const y = 1000 - ((lat - b.minLat) / Math.max(1e-9, b.maxLat - b.minLat)) * 1000;
  return { x, y };
}

function radiusToSvg(radiusM) {
  if (!state.bounds) return 40;
  const midLat = (state.bounds.minLat + state.bounds.maxLat) / 2;
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.max(0.1, Math.cos((midLat * Math.PI) / 180));
  const widthM = Math.max(1, (state.bounds.maxLon - state.bounds.minLon) * metersPerDegLon);
  const heightM = Math.max(1, (state.bounds.maxLat - state.bounds.minLat) * metersPerDegLat);
  return Math.max(12, (Math.min(Number(radiusM || 100), 350) / Math.max(widthM, heightM)) * 1000);
}

function setWorldTransform() {
  els.world.setAttribute("transform", `translate(${state.panX} ${state.panY}) scale(${state.zoom})`);
}

function upsertGroup(map, parent, key, create) {
  let node = map.get(key);
  if (!node) {
    node = create();
    parent.appendChild(node);
    map.set(key, node);
  }
  return node;
}

function prune(map, activeKeys) {
  for (const [key, node] of map.entries()) {
    if (!activeKeys.has(key)) {
      node.remove();
      map.delete(key);
    }
  }
}

function select(key, details) {
  state.selectedKey = key;
  els.selectedDetails.textContent = details;
  document.querySelectorAll(".list-item.selected").forEach((el) => el.classList.remove("selected"));
  document.querySelectorAll(`[data-select-key="${CSS.escape(key)}"]`).forEach((el) => el.classList.add("selected"));
}

function stationDetails(station) {
  return JSON.stringify({
    station_id: station.station_id,
    health: station.health || station.alive_state,
    status: station.last_status,
    line: station.line_id,
    latency_sec: station.latency_sec,
    heartbeat_age_sec: station.heartbeat_age_sec,
  }, null, 2);
}

function trackDetails(track, mapTrack = {}) {
  return JSON.stringify({
    track_id: track.track_id || mapTrack.track_id,
    level: track.level ?? mapTrack.level,
    confidence: track.confidence ?? mapTrack.confidence,
    interpretation: track.interpretation ?? mapTrack.interpretation,
    stations: track.station_ids || mapTrack.station_ids,
    source_ids: mapTrack.source_ids,
    eta_sec: mapTrack.target_eta_sec,
    line: mapTrack.latest_line_crossed,
  }, null, 2);
}

function updateStations(data) {
  const stations = data?.map_state?.stations || [];
  const active = new Set();
  for (const station of stations) {
    const point = pointFromItem(station);
    if (!point) continue;
    const key = `station:${station.station_id}`;
    active.add(key);
    const projected = project(point.lat, point.lon);
    const group = upsertGroup(state.stationNodes, els.stations, key, () => {
      const g = svgEl("g", { class: "station-node" });
      g.appendChild(svgEl("circle", { class: "station", r: 11 }));
      g.appendChild(svgEl("text", { class: "station-label", x: 16, y: 6 }));
      g.addEventListener("click", () => select(key, g.__details || ""));
      return g;
    });
    group.__details = stationDetails(station);
    if (state.selectedKey === key) els.selectedDetails.textContent = group.__details;
    group.setAttribute("transform", `translate(${projected.x} ${projected.y})`);
    group.querySelector("circle").setAttribute("fill", statusColor(station));
    group.querySelector("text").textContent = station.station_id || "";
  }
  prune(state.stationNodes, active);
}

function updateSectors(data) {
  const cues = els.toggleSectors.checked ? data?.map_state?.bearing_cues || [] : [];
  const active = new Set();
  for (const cue of cues) {
    const polygon = cue.sector_polygon || [];
    if (polygon.length < 3) continue;
    const key = `sector:${cue.station_id}`;
    active.add(key);
    const points = polygon
      .map((point) => project(Number(point.latitude), Number(point.longitude)))
      .map((point) => `${point.x},${point.y}`)
      .join(" ");
    const node = upsertGroup(state.sectorNodes, els.sectors, key, () => svgEl("polygon", { class: "sector" }));
    node.setAttribute("points", points);
  }
  prune(state.sectorNodes, active);
}

function updateTrackEstimates(data) {
  const estimates = els.toggleTrackEstimates.checked ? data?.map_state?.track_geo_estimates || [] : [];
  const active = new Set();
  for (const estimate of estimates) {
    const point = pointFromItem(estimate);
    if (!point) continue;
    const key = `estimate:${estimate.track_id}`;
    active.add(key);
    const projected = project(point.lat, point.lon);
    const group = upsertGroup(state.estimateNodes, els.estimates, key, () => {
      const g = svgEl("g");
      g.appendChild(svgEl("circle", { class: "estimate-ring" }));
      return g;
    });
    const circle = group.querySelector("circle");
    circle.setAttribute("cx", projected.x);
    circle.setAttribute("cy", projected.y);
    circle.setAttribute("r", radiusToSvg(estimate.radius_m || 100));
    circle.setAttribute("fill", estimateColor(estimate));
    circle.setAttribute("stroke", estimateColor(estimate).replace("0.28", "0.62").replace("0.24", "0.58").replace("0.22", "0.50"));
  }
  prune(state.estimateNodes, active);
}

function updateTracks(data) {
  const fusionTracks = data?.fusion?.tracks || [];
  const mapTracks = new Map((data?.map_state?.tracks || []).map((track) => [track.track_id, track]));
  const estimates = new Map((data?.map_state?.track_geo_estimates || []).map((estimate) => [estimate.track_id, estimate]));
  const active = new Set();
  for (const track of fusionTracks.length ? fusionTracks : data?.map_state?.tracks || []) {
    const mapTrack = mapTracks.get(track.track_id) || {};
    const estimate = estimates.get(track.track_id) || {};
    const point = pointFromItem(estimate) || pointFromItem(mapTrack) || pointFromItem(track);
    if (!point) continue;
    const key = `track:${track.track_id}`;
    active.add(key);
    const projected = project(point.lat, point.lon);
    const group = upsertGroup(state.trackNodes, els.tracks, key, () => {
      const g = svgEl("g", { class: "track-node" });
      g.appendChild(svgEl("circle", { class: "track-dot", r: 15 }));
      g.appendChild(svgEl("text", { class: "track-label", x: 20, y: -18 }));
      g.addEventListener("click", () => select(key, g.__details || ""));
      return g;
    });
    group.__details = trackDetails(track, mapTrack);
    if (state.selectedKey === key) els.selectedDetails.textContent = group.__details;
    group.setAttribute("transform", `translate(${projected.x} ${projected.y})`);
    group.querySelector("circle").setAttribute("fill", trackColor(track.level ?? mapTrack.level));
    group.querySelector("text").textContent = track.track_id || "track";
  }
  prune(state.trackNodes, active);
}

function updateTrackList(data) {
  const mapTracks = new Map((data?.map_state?.tracks || []).map((track) => [track.track_id, track]));
  const tracks = data?.fusion?.tracks || data?.map_state?.tracks || [];
  els.trackList.replaceChildren();
  for (const track of tracks) {
    const mapTrack = mapTracks.get(track.track_id) || {};
    const row = document.createElement("div");
    const key = `track:${track.track_id}`;
    row.className = `list-item${state.selectedKey === key ? " selected" : ""}`;
    row.dataset.selectKey = key;
    row.innerHTML = `
      <div class="item-title"><span>${track.track_id || "track"}</span><span>LEVEL ${track.level ?? mapTrack.level ?? 0}</span></div>
      <div class="item-meta">ETA ${fmt(mapTrack.target_eta_sec, 1)}s | line ${mapTrack.latest_line_crossed || "-"} | confidence ${fmt(track.confidence ?? mapTrack.confidence, 2)}</div>
    `;
    row.addEventListener("click", () => select(key, trackDetails(track, mapTrack)));
    els.trackList.appendChild(row);
  }
  if (!tracks.length) {
    const empty = document.createElement("div");
    empty.className = "item-meta";
    empty.textContent = "No active tracks.";
    els.trackList.appendChild(empty);
  }
}

function updateStationList(data) {
  const stations = data?.stations_health_summary || [];
  els.stationList.replaceChildren();
  for (const station of stations) {
    const row = document.createElement("div");
    const key = `station:${station.station_id}`;
    row.className = `list-item${state.selectedKey === key ? " selected" : ""}`;
    row.dataset.selectKey = key;
    row.innerHTML = `
      <div class="item-title"><span>${station.station_id}</span><span>${station.health || station.alive_state || "-"}</span></div>
      <div class="item-meta">line ${station.line_id || "-"} | status ${station.last_status || "-"} | latency ${fmt(station.latency_sec, 2)}s</div>
    `;
    row.addEventListener("click", () => select(key, stationDetails(station)));
    els.stationList.appendChild(row);
  }
}

function updateStatus(data) {
  const fusion = data?.fusion || {};
  const level = Number(fusion.level || 0);
  els.fusionLevel.textContent = `LEVEL ${level}`;
  els.fusionLevel.className = `level l${Math.min(3, Math.max(0, level))}`;
  els.fusionConfidence.textContent = fmt(fusion.confidence, 2);
  els.fusionInterpretation.textContent = fusion.interpretation || "background";
  els.trackCount.textContent = String((fusion.tracks || []).length);
  const age = data?.server_time ? Date.now() / 1000 - Number(data.server_time) : null;
  els.updateAge.textContent = age === null ? "-" : `${fmt(Math.max(0, age), 1)}s`;
  els.connectionState.textContent = "live";
  els.connectionState.className = "pill ok";
}

function render(data) {
  state.lastData = data;
  updateBounds(collectPoints(data));
  updateStatus(data);
  updateStations(data);
  updateSectors(data);
  updateTrackEstimates(data);
  updateTracks(data);
  updateTrackList(data);
  updateStationList(data);
  if (els.toggleFollow.checked && !state.manualView) {
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
  }
  setWorldTransform();
}

async function refreshLive() {
  if (state.refreshInFlight) return;
  state.refreshInFlight = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3500);
  try {
    const url = `/dashboard/live${els.toggleSectors.checked ? "?bearing_cues=1" : ""}`;
    const res = await fetch(url, { cache: "no-store", signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    render(await res.json());
  } catch (err) {
    els.connectionState.textContent = "offline";
    els.connectionState.className = "pill bad";
    console.error(err);
  } finally {
    clearTimeout(timeout);
    state.refreshInFlight = false;
  }
}

function setupMapInput() {
  els.map.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.manualView = true;
    els.toggleFollow.checked = false;
    const delta = event.deltaY < 0 ? 1.12 : 0.9;
    state.zoom = Math.max(0.45, Math.min(8, state.zoom * delta));
    setWorldTransform();
  });

  els.map.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.dragStart = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
    els.map.setPointerCapture(event.pointerId);
  });

  els.map.addEventListener("pointermove", (event) => {
    if (!state.dragging || !state.dragStart) return;
    state.manualView = true;
    els.toggleFollow.checked = false;
    state.panX = state.dragStart.panX + (event.clientX - state.dragStart.x);
    state.panY = state.dragStart.panY + (event.clientY - state.dragStart.y);
    setWorldTransform();
  });

  els.map.addEventListener("pointerup", () => {
    state.dragging = false;
    state.dragStart = null;
  });

  els.toggleSectors.addEventListener("change", refreshLive);
  els.toggleTrackEstimates.addEventListener("change", () => {
    if (state.lastData) render(state.lastData);
  });
  els.toggleFollow.addEventListener("change", () => {
    if (els.toggleFollow.checked) {
      state.manualView = false;
      state.zoom = 1;
      state.panX = 0;
      state.panY = 0;
      setWorldTransform();
    }
  });
}

setupMapInput();
setWorldTransform();
refreshLive();
setInterval(refreshLive, REFRESH_MS);
