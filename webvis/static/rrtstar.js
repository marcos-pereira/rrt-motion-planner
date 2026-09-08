// ── State ──────────────────────────────────────────────────────────────────
let app = null;         // PIXI.Application
let eventSource = null; // active SSE connection
let treeGraphics = null;   // cleared and redrawn each snapshot (rewiring changes parent pointers)
let overlayGraphics = null; // path + start/goal circles
let initPromise = null; // resolves when the canvas is ready
let isRunning = false;

// Preview state — shows the map + start/goal markers before any planning has run,
// and lets the user click the canvas to set them instead of typing coordinates.
let previewActive = false;
let mapWidth = 0;
let mapHeight = 0;
let canvasScale = 1;
let placeMode = null; // null | 'start' | 'goal' — which point the next click sets
let previewGeneration = 0; // guards against overlapping initPreview() calls (rapid map switches)

// ── DOM refs ───────────────────────────────────────────────────────────────
const runBtn      = document.getElementById('run-btn');
const stopBtn     = document.getElementById('stop-btn');
const statusEl    = document.getElementById('status');
const mapSelect   = document.getElementById('map-name');
const batchInput  = document.getElementById('batch-size');
const batchLabel  = document.getElementById('batch-label');
const canvasEl    = document.getElementById('canvas-container');
const placeholder = document.getElementById('placeholder');
const placeholderText = placeholder.querySelector('p');
const setStartBtn = document.getElementById('set-start-btn');
const setGoalBtn  = document.getElementById('set-goal-btn');
const x0Input      = document.getElementById('x0');
const y0Input      = document.getElementById('y0');
const xgInput      = document.getElementById('xg');
const ygInput      = document.getElementById('yg');
const goalRadiusInput = document.getElementById('goal-radius');

// ── Startup ────────────────────────────────────────────────────────────────
async function loadMapList() {
    try {
        const resp = await fetch('/maps-list');
        const maps = await resp.json();
        maps.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (name === 'smile.png') opt.selected = true;
            mapSelect.appendChild(opt);
        });
    } catch {
        setStatus('⚠️ Could not load map list.');
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function setStatus(msg) { statusEl.textContent = msg; }

function getParams() {
    const params = {
        map_name:    mapSelect.value,
        steer_delta: parseFloat(document.getElementById('steer-delta').value),
        goal_radius: parseInt(document.getElementById('goal-radius').value),
        num_nodes:   parseInt(document.getElementById('num-nodes').value),
        x0:          parseInt(document.getElementById('x0').value),
        y0:          parseInt(document.getElementById('y0').value),
        xg:          parseInt(document.getElementById('xg').value),
        yg:          parseInt(document.getElementById('yg').value),
        batch_size:  parseInt(batchInput.value),
        gamma_rrt:   parseFloat(document.getElementById('gamma-rrt').value),
        eta:         parseFloat(document.getElementById('eta').value),
    };

    const maxTimeRaw = document.getElementById('max-planning-time').value;
    if (maxTimeRaw !== '') {
        params.max_planning_time = parseFloat(maxTimeRaw);
    }

    return params;
}

function destroyApp() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    if (app) {
        app.view.removeEventListener('click', onCanvasClick);
        // texture:false — the map sprite's texture is managed by PIXI.Assets;
        // destroying it here (instead of via Assets.unload()) corrupts the Assets
        // cache for future loads of the same map.
        app.destroy(true, { children: true, texture: false });
        app = treeGraphics = overlayGraphics = null;
        const old = canvasEl.querySelector('canvas');
        if (old) old.remove();
    }
    initPromise = null;
    isRunning = false;
    previewActive = false;
    setPlaceMode(null);
}

// ── Preview (click-to-place start/goal) ───────────────────────────────────
// Shows just the map and the start/goal markers — no tree, no path — so the
// user can set start/goal either by clicking here or by typing coordinates,
// before ever running the planner.
async function initPreview() {
    const myGeneration = ++previewGeneration;
    destroyApp();
    placeholder.style.display = 'block';
    placeholderText.textContent = 'Loading map...';
    setStatus('Ready — select a map and click Run.');

    const mapName = mapSelect.value;
    if (!mapName) {
        placeholderText.textContent = 'No map selected.';
        return;
    }

    // Everything below can throw for reasons unrelated to the map fetch itself
    // (e.g. WebGL unavailable when constructing the PIXI.Application) — wrapping
    // the whole thing, not just the texture load, ensures a failure here always
    // shows a clear message instead of leaving "Loading map..." stuck forever.
    try {
        const texture = await PIXI.Assets.load(`/maps/${mapName}`);
        if (myGeneration !== previewGeneration) return; // superseded by a newer call

        mapWidth = texture.width;
        mapHeight = texture.height;

        const containerW = canvasEl.clientWidth  || 800;
        const containerH = canvasEl.clientHeight || 600;
        canvasScale = Math.min(containerW / mapWidth, containerH / mapHeight, 1);
        const canvasW = Math.floor(mapWidth  * canvasScale);
        const canvasH = Math.floor(mapHeight * canvasScale);

        app = new PIXI.Application({
            width: canvasW,
            height: canvasH,
            backgroundColor: 0xF5F5F5,
            antialias: true,
        });
        canvasEl.appendChild(app.view);
        app.stage.scale.set(canvasScale);

        const mapSprite = new PIXI.Sprite(texture);
        app.stage.addChild(mapSprite);

        overlayGraphics = new PIXI.Graphics();
        app.stage.addChild(overlayGraphics);

        app.view.style.cursor = 'crosshair';
        app.view.addEventListener('click', onCanvasClick);

        placeholder.style.display = 'none';
        previewActive = true;
        updateMarkersFromInputs();
    } catch (e) {
        if (myGeneration !== previewGeneration) return; // superseded by a newer call
        console.error('Failed to show the map preview:', e);
        placeholder.style.display = 'block';
        placeholderText.textContent = '⚠️ Could not load the map.';
        setStatus(`⚠️ Could not load map: ${e.message || e}`);
    }
}

function onCanvasClick(event) {
    if (!placeMode) return;

    const rect = app.view.getBoundingClientRect();
    const clickX = (event.clientX - rect.left) / canvasScale;
    const clickY = (event.clientY - rect.top) / canvasScale;
    const mapX = Math.round(Math.min(Math.max(clickX, 0), mapWidth - 1));
    const mapY = Math.round(Math.min(Math.max(clickY, 0), mapHeight - 1));

    if (placeMode === 'start') {
        x0Input.value = mapX;
        y0Input.value = mapY;
    } else {
        xgInput.value = mapX;
        ygInput.value = mapY;
    }

    updateMarkersFromInputs();
    setPlaceMode(null);
}

function setPlaceMode(mode) {
    // Clicking the already-active tool's button turns it back off.
    placeMode = (placeMode === mode) ? null : mode;
    setStartBtn.classList.toggle('active', placeMode === 'start');
    setGoalBtn.classList.toggle('active', placeMode === 'goal');
}

function updateMarkersFromInputs() {
    if (!overlayGraphics || !previewActive) return;

    const x0 = parseFloat(x0Input.value) || 0;
    const y0 = parseFloat(y0Input.value) || 0;
    const xg = parseFloat(xgInput.value) || 0;
    const yg = parseFloat(ygInput.value) || 0;
    const goalRadius = parseInt(goalRadiusInput.value) || 10;

    overlayGraphics.clear();
    overlayGraphics.lineStyle(0);
    overlayGraphics.beginFill(0x5CD676);
    overlayGraphics.drawCircle(xg, yg, goalRadius);
    overlayGraphics.endFill();
    overlayGraphics.beginFill(0xFFCF58);
    overlayGraphics.drawCircle(x0, y0, goalRadius);
    overlayGraphics.endFill();
}

async function ensurePreview() {
    if (!previewActive) await initPreview();
}

// ── Main flow ──────────────────────────────────────────────────────────────
async function run() {
    runBtn.disabled = true;
    stopBtn.disabled = false;
    destroyApp();
    placeholder.style.display = 'block';

    const params = getParams();
    const url = new URL('/plan-rrtstar', window.location.origin);
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));

    setStatus('🌱 Connecting to RRT* planner...');
    isRunning = true;

    eventSource = new EventSource(url.toString());

    // onmessage is an async handler: each message awaits initPromise so the
    // canvas is always ready before drawSnapshot is called, even if the first
    // two events arrive before initCanvas resolves.
    eventSource.onmessage = async (event) => {
        const snap = JSON.parse(event.data);

        if (snap.error) {
            setStatus(`❌ ${snap.error}`);
            finish();
            await initPreview();
            return;
        }

        if (!initPromise) {
            initPromise = initCanvas(snap);
        }
        await initPromise;

        drawSnapshot(snap);

        const costStr = (snap.path_found && snap.path_cost !== null)
            ? snap.path_cost.toFixed(1) : '—';
        setStatus(`🔄 Nodes: ${snap.node_count} | Best cost: ${costStr}`);

        if (snap.done) {
            finish();
            const finalMsg = snap.path_found
                ? `Path cost: ${snap.path_cost.toFixed(1)}`
                : (snap.stop_reason === 'max_time' ? 'No path found — max planning time reached' : 'No path found — max nodes reached');
            const icon = snap.path_found ? '✅' : (snap.stop_reason === 'max_time' ? '⏱️' : '⚠️');
            setStatus(`${icon} Done — Nodes: ${snap.node_count} | ${finalMsg}`);
        }
    };

    eventSource.onerror = () => {
        const wasRunning = isRunning;
        if (wasRunning) setStatus('❌ Stream error or connection closed.');
        finish();
        if (wasRunning) initPreview();
    };
}

function stop() {
    finish();
    setStatus('⏹ Stopped.');
}

function finish() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    isRunning = false;
    runBtn.disabled = false;
    stopBtn.disabled = true;
}

// ── Canvas init ────────────────────────────────────────────────────────────
async function initCanvas(data) {
    const containerW = canvasEl.clientWidth  || 800;
    const containerH = canvasEl.clientHeight || 600;
    const scale = Math.min(containerW / data.map_width, containerH / data.map_height, 1);
    const canvasW = Math.floor(data.map_width  * scale);
    const canvasH = Math.floor(data.map_height * scale);

    app = new PIXI.Application({
        width: canvasW, height: canvasH,
        backgroundColor: 0xF5F5F5,
        antialias: true,
    });
    canvasEl.appendChild(app.view);

    // Scale the stage so node coordinates map 1-to-1 with map pixels.
    // PixiJS y-axis goes DOWN (top-left origin) — same as image coords, no flip needed.
    app.stage.scale.set(scale);

    const texture = await PIXI.Assets.load(`/maps/${data.map_name}`);
    const mapSprite = new PIXI.Sprite(texture);
    app.stage.addChild(mapSprite);

    // Tree edges — cleared and redrawn on every snapshot because rewiring
    // changes parent pointers, so the tree cannot simply grow incrementally.
    treeGraphics = new PIXI.Graphics();
    app.stage.addChild(treeGraphics);

    // Path and start/goal circles sit on top of the tree.
    overlayGraphics = new PIXI.Graphics();
    app.stage.addChild(overlayGraphics);

    placeholder.style.display = 'none';
}

// ── Drawing ────────────────────────────────────────────────────────────────
function drawSnapshot(snap) {
    if (!app || !treeGraphics || !overlayGraphics) return;

    // Redraw all current tree edges from scratch.
    treeGraphics.clear();
    treeGraphics.lineStyle(1.5, 0xC400B7, 1);
    for (const [p, q] of snap.edges) {
        treeGraphics.moveTo(p[0], p[1]);
        treeGraphics.lineTo(q[0], q[1]);
    }

    // Redraw overlay: current best path + start/goal circles.
    overlayGraphics.clear();

    if (snap.path_found && snap.path.length > 1) {
        overlayGraphics.lineStyle(4, 0x0B27DB, 1);
        overlayGraphics.moveTo(snap.path[0][0], snap.path[0][1]);
        for (let i = 1; i < snap.path.length; i++) {
            overlayGraphics.lineTo(snap.path[i][0], snap.path[i][1]);
        }
    }

    // Goal circle (green when reached, muted green otherwise)
    overlayGraphics.lineStyle(0);
    overlayGraphics.beginFill(snap.path_found ? 0x0AD676 : 0x5CD676, 1);
    overlayGraphics.drawCircle(snap.x_goal[0], snap.x_goal[1], snap.goal_radius);
    overlayGraphics.endFill();

    // Start circle (yellow)
    overlayGraphics.beginFill(0xFFCF58, 1);
    overlayGraphics.drawCircle(snap.x_init[0], snap.x_init[1], snap.goal_radius);
    overlayGraphics.endFill();
}

// ── Events ─────────────────────────────────────────────────────────────────
batchInput.addEventListener('input', () => {
    batchLabel.textContent = `${batchInput.value} steps`;
});

runBtn.addEventListener('click', run);
stopBtn.addEventListener('click', stop);

setStartBtn.addEventListener('click', async () => {
    await ensurePreview();
    setPlaceMode('start');
});

setGoalBtn.addEventListener('click', async () => {
    await ensurePreview();
    setPlaceMode('goal');
});

mapSelect.addEventListener('change', () => {
    initPreview();
});

// Typing coordinates directly still works — keep the preview markers in sync.
[x0Input, y0Input, xgInput, ygInput, goalRadiusInput].forEach(input => {
    input.addEventListener('input', updateMarkersFromInputs);
});

// ── Init ───────────────────────────────────────────────────────────────────
loadMapList().then(initPreview);
