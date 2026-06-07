const SVG_NS = "http://www.w3.org/2000/svg";
const REFRESH_MS = 500;

const els = {
  mapWrap: document.querySelector(".map-wrap"),
  map: document.getElementById("liveMap"),
  tileLayer: document.getElementById("tileLayer"),
  mapWarning: document.getElementById("mapWarning"),
  attribution: document.getElementById("mapAttribution"),
  world: document.getElementById("worldLayer"),
  coverage: document.getElementById("coverageLayer"),
  lines: document.getElementById("lineLayer"),
  stations: document.getElementById("stationLayer"),
  tracks: document.getElementById("trackLayer"),
  sectors: document.getElementById("sectorLayer"),
  estimates: document.getElementById("trackEstimateLayer"),
  fusionLevel: document.getElementById("fusionLevel"),
  fusionConfidence: document.getElementById("fusionConfidence"),
  fusionInterpretation: document.getElementById("fusionInterpretation"),
  trackCount: document.getElementById("trackCount"),
  stationCount: document.getElementById("stationCount"),
  degradedCount: document.getElementById("degradedCount"),
  nearestEta: document.getElementById("nearestEta"),
  latestLine: document.getElementById("latestLine"),
  sourceIds: document.getElementById("sourceIds"),
  updateAge: document.getElementById("updateAge"),
  connectionState: document.getElementById("connectionState"),
  trackList: document.getElementById("trackList"),
  stationList: document.getElementById("stationList"),
  selectedDetails: document.getElementById("selectedDetails"),
  mapMode: document.getElementById("mapMode"),
  centerMode: document.getElementById("centerMode"),
  toggleSectors: document.getElementById("toggleSectors"),
  toggleTrackEstimates: document.getElementById("toggleTrackEstimates"),
  toggleCoverage: document.getElementById("toggleCoverage"),
  toggleFollow: document.getElementById("toggleFollow"),
  toggleFollowTrack: document.getElementById("toggleFollowTrack"),
  togglePause: document.getElementById("togglePause"),
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
  lineNodes: new Map(),
  coverageNodes: new Map(),
  renderer: null,
  liveMapConfig: null,
  urlOptions: SkyEarLiveMap.queryOptions(),
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

function text(value, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function escapeHtml(value) {
  return text(value, "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

function compactList(values, fallback = "-") {
  if (!values || !values.length) return fallback;
  return values.join(",");
}

function statusColor(station) {
  const health = String(station.health || station.alive_state || "").toLowerCase();
  const status = String(station.last_status || "").toLowerCase();
  if (health === "offline" || health === "stale" || health === "degraded" || health === "error") return "#747b85";
  if (status === "alert" || status === "drone_like") return "#d9443f";
  if (status === "suspect" || status === "calibrating") return "#d5aa32";
  return "#39b56a";
}

function stationHealthClass(station) {
  const health = String(station.health || station.alive_state || "").toLowerCase();
  if (health === "offline" || health === "error") return "offline";
  if (health === "stale" || health === "degraded") return "degraded";
  return "";
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
  if (!item) return null;
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
  for (const track of data?.tracks || []) {
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
  state.renderer?.updateBounds(points);
}

function project(lat, lon) {
  return state.renderer.project(lat, lon);
}

function radiusToSvg(radiusM) {
  return state.renderer.radiusToSvg(radiusM);
}

function setWorldTransform() {
  state.renderer?.setWorldTransform();
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
  document.querySelectorAll(".track-node.selected").forEach((el) => el.classList.remove("selected"));
  const trackNode = state.trackNodes.get(key);
  if (trackNode) {
    trackNode.classList.add("selected");
    if (els.toggleFollowTrack.checked) {
      panToNode(trackNode);
    }
  }
}

function stationDetails(station) {
  return JSON.stringify({
    station_id: station.station_id,
    health: station.health || station.alive_state,
    status: station.last_status,
    line: station.line_id,
    latency_sec: station.latency_sec,
    heartbeat_age_sec: station.heartbeat_age_sec,
    line_distance_m: station.line_distance_m,
    fiber_node_id: station.fiber_node_id,
  }, null, 2);
}

function trackDetails(track, mapTrack = {}) {
  return JSON.stringify({
    track_id: track.track_id || mapTrack.track_id,
    level: track.level ?? mapTrack.level,
    confidence: track.confidence ?? mapTrack.confidence,
    interpretation: track.interpretation ?? mapTrack.interpretation,
    source_ids: track.source_ids || mapTrack.source_ids,
    latest_line_crossed: track.latest_line_crossed ?? mapTrack.latest_line_crossed,
    target_eta_sec: track.target_eta_sec ?? mapTrack.target_eta_sec,
    target_distance_to_control_m: track.target_distance_to_control_m ?? mapTrack.target_distance_to_control_m,
    stations: track.station_ids || mapTrack.station_ids,
    ambiguity: track.ambiguity ?? mapTrack.ambiguity,
  }, null, 2);
}

function trackPoint(track, estimates = new Map()) {
  const estimate = estimates.get(track.track_id) || {};
  return pointFromItem(track) || pointFromItem(estimate) || pointFromItem(track.estimated_source || {});
}

function panToNode(node) {
  if (!node || !node.__point) return;
  state.manualView = true;
  els.toggleFollow.checked = false;
  state.panX = 500 - node.__point.x * state.zoom;
  state.panY = 500 - node.__point.y * state.zoom;
  setWorldTransform();
}

function trackMetaLabel(track, mapTrack = {}) {
  const eta = track.target_eta_sec ?? mapTrack.target_eta_sec;
  const line = track.latest_line_crossed ?? mapTrack.latest_line_crossed;
  const parts = [];
  if (eta !== null && eta !== undefined) parts.push(`ETA ${fmt(eta, 1)}s`);
  if (line) parts.push(`line ${line}`);
  return parts.join("  ");
}

async function fetchLiveMapConfig() {
  try {
    const res = await fetch("/live/config", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("Live map config unavailable; using schematic defaults", err);
    return { live_map: { mode: "schematic" } };
  }
}

function activateRenderer(config) {
  const renderer = SkyEarLiveMap.createRenderer(config.mode, { state, els });
  if (renderer.configure(config)) {
    state.renderer = renderer;
    els.mapMode.value = renderer.mode;
    if (renderer.mode === "geo" && state.urlOptions.lat !== null && state.urlOptions.lon !== null) {
      els.centerMode.value = "manual";
      els.toggleFollow.checked = false;
      state.manualView = true;
    }
    els.mapWarning.hidden = !config.warning;
    els.mapWarning.textContent = config.warning || "";
    setWorldTransform();
    return;
  }
  const fallback = SkyEarLiveMap.createRenderer("schematic", { state, els });
  fallback.configure({});
  state.renderer = fallback;
  els.mapMode.value = "schematic";
  els.mapWarning.textContent = "Basemap unavailable - using schematic mode";
  els.mapWarning.hidden = false;
  setWorldTransform();
}

window.SkyEarLiveFallbackToSchematic = () => {
  if (state.renderer?.mode === "schematic") return;
  const fallback = SkyEarLiveMap.createRenderer("schematic", { state, els });
  fallback.configure({});
  state.renderer = fallback;
  els.mapMode.value = "schematic";
  state.bounds = null;
  state.manualView = false;
  if (state.lastData) render(state.lastData);
};

async function configureRenderer(preferredMode = null) {
  const payload = await fetchLiveMapConfig();
  const options = { ...state.urlOptions };
  if (preferredMode) options.mode = preferredMode;
  const config = SkyEarLiveMap.normalizeConfig(payload, options);
  state.liveMapConfig = config;
  state.bounds = null;
  state.manualView = false;
  activateRenderer(config);
  if (state.lastData) render(state.lastData);
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
    const circle = group.querySelector("circle");
    circle.setAttribute("fill", statusColor(station));
    circle.setAttribute("class", `station ${stationHealthClass(station)}`.trim());
    group.querySelector("text").textContent = station.station_id || "";
  }
  prune(state.stationNodes, active);
}

function updateCoverage(data) {
  const stations = els.toggleCoverage.checked ? data?.map_state?.stations || [] : [];
  const active = new Set();
  for (const station of stations) {
    const point = pointFromItem(station);
    if (!point || station.coverage_radius_m === null || station.coverage_radius_m === undefined) continue;
    const key = `coverage:${station.station_id}`;
    active.add(key);
    const projected = project(point.lat, point.lon);
    const node = upsertGroup(state.coverageNodes, els.coverage, key, () => svgEl("circle", { class: "coverage-ring" }));
    node.setAttribute("cx", projected.x);
    node.setAttribute("cy", projected.y);
    node.setAttribute("r", radiusToSvg(station.coverage_radius_m));
  }
  prune(state.coverageNodes, active);
}

function lineSortValue(station) {
  const id = String(station.station_id || "");
  const numberMatch = id.match(/(\d+)$/);
  if (numberMatch) return Number(numberMatch[1]);
  return Number(station.longitude ?? station.lon ?? 0);
}

function updateStationLines(data) {
  const stations = (data?.map_state?.stations || []).filter((station) => station.line_id && pointFromItem(station));
  const byLine = new Map();
  for (const station of stations) {
    const key = String(station.line_id);
    if (!byLine.has(key)) byLine.set(key, []);
    byLine.get(key).push(station);
  }
  const active = new Set();
  for (const [lineId, lineStations] of byLine.entries()) {
    if (lineStations.length < 2) continue;
    lineStations.sort((a, b) => lineSortValue(a) - lineSortValue(b));
    const key = `line:${lineId}`;
    active.add(key);
    const group = upsertGroup(state.lineNodes, els.lines, key, () => {
      const g = svgEl("g", { class: "station-line" });
      g.appendChild(svgEl("polyline", { class: "line-path" }));
      g.appendChild(svgEl("text", { class: "line-label" }));
      return g;
    });
    const points = lineStations
      .map((station) => pointFromItem(station))
      .map((point) => project(point.lat, point.lon));
    group.querySelector("polyline").setAttribute("points", points.map((point) => `${point.x},${point.y}`).join(" "));
    const mid = points[Math.floor(points.length / 2)];
    const distance = lineStations.find((station) => station.line_distance_m !== null && station.line_distance_m !== undefined)?.line_distance_m;
    const label = group.querySelector("text");
    label.setAttribute("x", mid.x + 12);
    label.setAttribute("y", mid.y - 12);
    label.textContent = `Line ${lineId}${distance !== undefined ? ` - ${fmt(distance, 0)}m` : ""}`;
  }
  prune(state.lineNodes, active);
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
  const fusionTracks = data?.tracks || data?.fusion?.tracks || [];
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
      g.appendChild(svgEl("text", { class: "track-meta-label", x: 20, y: 4 }));
      g.addEventListener("click", () => select(key, g.__details || ""));
      return g;
    });
    group.__details = trackDetails(track, mapTrack);
    if (state.selectedKey === key) els.selectedDetails.textContent = group.__details;
    group.classList.toggle("selected", state.selectedKey === key);
    group.__point = projected;
    group.setAttribute("transform", `translate(${projected.x} ${projected.y})`);
    group.querySelector("circle").setAttribute("fill", trackColor(track.level ?? mapTrack.level));
    group.querySelector(".track-label").textContent = track.track_id || "track";
    group.querySelector(".track-meta-label").textContent = trackMetaLabel(track, mapTrack);
    if (state.selectedKey === key && els.toggleFollowTrack.checked) {
      panToNode(group);
    }
  }
  prune(state.trackNodes, active);
}

function updateTrackList(data) {
  const mapTracks = new Map((data?.map_state?.tracks || []).map((track) => [track.track_id, track]));
  const tracks = data?.tracks || data?.fusion?.tracks || data?.map_state?.tracks || [];
  els.trackList.replaceChildren();
  for (const track of tracks) {
    const mapTrack = mapTracks.get(track.track_id) || {};
    const row = document.createElement("div");
    const key = `track:${track.track_id}`;
    row.className = `list-item${state.selectedKey === key ? " selected" : ""}`;
    row.dataset.selectKey = key;
    const sourceIds = compactList(track.source_ids || mapTrack.source_ids || []);
    const stations = compactList(track.station_ids || mapTrack.station_ids || []);
    const eta = track.target_eta_sec ?? mapTrack.target_eta_sec;
    const distance = track.target_distance_to_control_m ?? mapTrack.target_distance_to_control_m;
    const line = track.latest_line_crossed ?? mapTrack.latest_line_crossed;
    const ambiguity = track.ambiguity ?? mapTrack.ambiguity;
    row.innerHTML = `
      <div class="item-title"><span>${escapeHtml(track.track_id || "track")}</span><span>L${escapeHtml(track.level ?? mapTrack.level ?? 0)}</span></div>
      <div class="item-meta">source=${escapeHtml(sourceIds)} line=${escapeHtml(line || "-")} ETA=${fmt(eta, 1)}s distance=${fmt(distance, 0)}m confidence=${fmt(track.confidence ?? mapTrack.confidence, 2)}</div>
      <div class="item-meta">stations=${escapeHtml(stations)}${ambiguity ? ` | ${escapeHtml(ambiguity)}` : ""}</div>
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
      <div class="item-title"><span>${escapeHtml(station.station_id)}</span><span>${escapeHtml(station.health || station.alive_state || "-")}</span></div>
      <div class="item-meta">line ${escapeHtml(station.line_id || "-")} | distance ${fmt(station.line_distance_m, 0)}m | status ${escapeHtml(station.last_status || "-")} | latency ${fmt(station.latency_sec, 2)}s</div>
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
  els.trackCount.textContent = String(data?.track_count ?? fusion.track_count ?? (data?.tracks || fusion.tracks || []).length);
  const totalStations = data?.total_station_count ?? fusion.total_station_count;
  const onlineStations = data?.online_station_count ?? fusion.online_station_count;
  const degradedStations = data?.degraded_station_count ?? fusion.degraded_station_count ?? 0;
  const offlineStations = data?.offline_station_count ?? fusion.offline_station_count ?? 0;
  els.stationCount.textContent = totalStations === undefined ? "-" : `${onlineStations ?? 0}/${totalStations}`;
  els.degradedCount.textContent = totalStations === undefined ? "-" : `${degradedStations}/${offlineStations}`;
  const eta = data?.nearest_eta_sec ?? fusion.nearest_eta_sec;
  els.nearestEta.textContent = eta === null || eta === undefined ? "-" : `${fmt(eta, 1)}s`;
  els.latestLine.textContent = text(data?.latest_line_crossed ?? fusion.latest_line_crossed);
  els.sourceIds.textContent = compactList(data?.source_ids ?? fusion.source_ids ?? []);
  const age = data?.server_time ? Date.now() / 1000 - Number(data.server_time) : null;
  els.updateAge.textContent = age === null ? "-" : `${fmt(Math.max(0, age), 1)}s`;
  els.connectionState.textContent = els.togglePause.checked ? "paused" : "live";
  els.connectionState.className = "pill ok";
}

function render(data) {
  if (!state.renderer) return;
  state.lastData = data;
  const points = collectPoints(data);
  updateBounds(points);
  if (els.toggleFollow.checked && !state.manualView) {
    state.renderer.applyAutoView(data, els.centerMode.value);
  }
  updateStatus(data);
  updateCoverage(data);
  updateStationLines(data);
  updateStations(data);
  updateSectors(data);
  updateTrackEstimates(data);
  updateTracks(data);
  updateTrackList(data);
  updateStationList(data);
  setWorldTransform();
}

async function refreshLive() {
  if (!state.renderer) return;
  if (els.togglePause.checked) {
    if (state.lastData) updateStatus(state.lastData);
    return;
  }
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
    state.renderer?.wheel(event);
  });

  els.map.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.dragStart = {
      x: event.clientX,
      y: event.clientY,
      panX: state.panX,
      panY: state.panY,
      centerLat: state.geo?.centerLat,
      centerLon: state.geo?.centerLon,
    };
    els.map.setPointerCapture(event.pointerId);
  });

  els.map.addEventListener("pointermove", (event) => {
    if (!state.dragging || !state.dragStart) return;
    state.manualView = true;
    els.toggleFollow.checked = false;
    state.renderer?.drag(event.clientX - state.dragStart.x, event.clientY - state.dragStart.y);
  });

  els.map.addEventListener("pointerup", () => {
    state.dragging = false;
    state.dragStart = null;
  });

  els.toggleSectors.addEventListener("change", refreshLive);
  els.toggleTrackEstimates.addEventListener("change", () => {
    if (state.lastData) render(state.lastData);
  });
  els.toggleCoverage.addEventListener("change", () => {
    if (state.lastData) render(state.lastData);
  });
  els.toggleFollow.addEventListener("change", () => {
    if (els.toggleFollow.checked) {
      state.manualView = false;
      if (state.lastData) render(state.lastData);
    }
  });
  els.centerMode.addEventListener("change", () => {
    if (els.centerMode.value !== "manual") {
      state.manualView = false;
      els.toggleFollow.checked = true;
    }
    if (state.lastData) render(state.lastData);
  });
  els.mapMode.addEventListener("change", () => {
    configureRenderer(els.mapMode.value);
  });
  els.toggleFollowTrack.addEventListener("change", () => {
    if (els.toggleFollowTrack.checked && state.selectedKey) {
      panToNode(state.trackNodes.get(state.selectedKey));
    }
  });
  els.togglePause.addEventListener("change", () => {
    if (!els.togglePause.checked) refreshLive();
    if (state.lastData) updateStatus(state.lastData);
  });
}

setupMapInput();

async function initLive() {
  await configureRenderer();
  refreshLive();
  setInterval(refreshLive, REFRESH_MS);
}

initLive();
