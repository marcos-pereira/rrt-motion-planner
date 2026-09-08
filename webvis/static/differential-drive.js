// ── State ──────────────────────────────────────────────────────────────────
let app = null;           // PIXI.Application
let planData = null;      // last response from /plan-differential-drive
let edgeIndex = 0;        // how many edges have been drawn
let isAnimating = false;

// PixiJS objects created per run
let renderTexture = null; // accumulates drawn edges efficiently
let edgeSprite = null;    // sprite backed by renderTexture
let tempGraphics = null;  // re-used each frame to draw a batch of edges
let overlayGraphics = null; // path + start/goal circles drawn after animation
let robotGraphics = null;   // differential-drive robot glyph (circle + heading + axle)

// Robot animation state — steps through planData.path at robotFps states/second,
// mirroring PlanDrawer.animate_differential_drive_path() in the desktop app.
let robotPath = null;
let robotIndex = 0;
let robotElapsedMs = 0;
let robotRadius = 8;
let robotFps = 10;
let isAnimatingRobot = false;

// Preview state — shows the map + start/goal markers before any planning has run,
// and lets the user click the canvas to set them instead of typing coordinates.
let previewActive = false;
let mapWidth = 0;
let mapHeight = 0;
let canvasScale = 1;
let placeMode = null; // null | 'start' | 'goal' — which point the next click sets
let previewGeneration = 0; // guards against overlapping initPreview() calls (rapid map switches)

// ── DOM refs ───────────────────────────────────────────────────────────────
const runBtn         = document.getElementById('run-btn');
const statusEl       = document.getElementById('status');
const mapSelect      = document.getElementById('map-name');
const speedInput     = document.getElementById('speed');
const speedLabel     = document.getElementById('speed-label');
const robotFpsInput  = document.getElementById('robot-fps');
const robotFpsLabel  = document.getElementById('robot-fps-label');
const canvasEl       = document.getElementById('canvas-container');
const placeholder    = document.getElementById('placeholder');
const placeholderText = placeholder.querySelector('p');
const setStartBtn    = document.getElementById('set-start-btn');
const setGoalBtn     = document.getElementById('set-goal-btn');
const x0Input         = document.getElementById('x0');
const y0Input         = document.getElementById('y0');
const xgInput         = document.getElementById('xg');
const ygInput         = document.getElementById('yg');
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
        map_name:      mapSelect.value,
        sampling_time: parseFloat(document.getElementById('sampling-time').value),
        goal_radius:   parseInt(document.getElementById('goal-radius').value),
        robot_radius:  parseFloat(document.getElementById('robot-radius').value),
        num_nodes:     parseInt(document.getElementById('num-nodes').value),
        x0: parseFloat(document.getElementById('x0').value),
        y0: parseFloat(document.getElementById('y0').value),
        xg: parseFloat(document.getElementById('xg').value),
        yg: parseFloat(document.getElementById('yg').value),
    };

    const maxTimeRaw = document.getElementById('max-planning-time').value;
    if (maxTimeRaw !== '') {
        params.max_planning_time = parseFloat(maxTimeRaw);
    }

    return params;
}

function destroyApp() {
    if (!app) return;
    app.ticker.remove(animationStep);
    app.ticker.remove(robotAnimationStep);
    app.view.removeEventListener('click', onCanvasClick);
    // texture:false — the map sprite's texture is managed by PIXI.Assets; destroying
    // it here (instead of via Assets.unload()) corrupts the Assets cache for future
    // loads of the same map. renderTexture (tree-edge accumulation) is NOT
    // Assets-managed, so it's destroyed explicitly below instead.
    app.destroy(true, { children: true, texture: false });
    if (renderTexture) renderTexture.destroy(true);
    app = renderTexture = edgeSprite = tempGraphics = overlayGraphics = robotGraphics = null;
    isAnimatingRobot = false;
    previewActive = false;
    setPlaceMode(null);
    // Remove the canvas element the old app appended
    const old = canvasEl.querySelector('canvas');
    if (old) old.remove();
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
        planData = null;
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
    destroyApp();
    placeholder.style.display = 'block';

    const params = getParams();
    setStatus('⏳ Running RRT (this may take a moment)...');

    const url = new URL('/plan-differential-drive', window.location.origin);
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));

    let data;
    try {
        const resp = await fetch(url);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        data = await resp.json();
    } catch (e) {
        setStatus(`❌ ${e.message}`);
        runBtn.disabled = false;
        await initPreview();
        return;
    }

    planData = data;
    placeholder.style.display = 'none';
    setStatus(`🎨 Animating ${data.edges.length} edges...`);

    await initCanvas(data);
    startAnimation();
}

// ── Canvas init ────────────────────────────────────────────────────────────
async function initCanvas(data) {
    // Fit the map into the available container while preserving aspect ratio
    const containerW = canvasEl.clientWidth  || 800;
    const containerH = canvasEl.clientHeight || 600;
    const scale = Math.min(containerW / data.map_width, containerH / data.map_height, 1);
    const canvasW = Math.floor(data.map_width  * scale);
    const canvasH = Math.floor(data.map_height * scale);

    app = new PIXI.Application({
        width: canvasW,
        height: canvasH,
        backgroundColor: 0xF5F5F5,
        antialias: true,
    });
    canvasEl.appendChild(app.view);

    // Scale the stage so node coordinates map 1-to-1 with map pixels
    app.stage.scale.set(scale);

    // Background: the original map PNG
    // PixiJS y-axis goes DOWN (top-left origin) — same as image coords, so no flip needed.
    const texture = await PIXI.Assets.load(`/maps/${data.map_name}`);
    const mapSprite = new PIXI.Sprite(texture);
    app.stage.addChild(mapSprite);

    // RenderTexture accumulates edges each frame without recalculating all geometry
    renderTexture = PIXI.RenderTexture.create({ width: data.map_width, height: data.map_height });
    edgeSprite = new PIXI.Sprite(renderTexture);
    app.stage.addChild(edgeSprite);

    // Temporary Graphics object re-used every frame to draw one batch
    tempGraphics = new PIXI.Graphics();

    // Overlay sits on top: path line + start/goal circles
    overlayGraphics = new PIXI.Graphics();
    app.stage.addChild(overlayGraphics);

    // Robot glyph sits on top of everything else
    robotGraphics = new PIXI.Graphics();
    app.stage.addChild(robotGraphics);
}

// ── Tree animation ─────────────────────────────────────────────────────────
function startAnimation() {
    edgeIndex = 0;
    isAnimating = true;
    app.ticker.add(animationStep);
}

function animationStep() {
    if (!isAnimating || !planData) return;

    const speed  = parseInt(speedInput.value);
    const edges  = planData.edges;
    const count  = Math.min(speed, edges.length - edgeIndex);

    if (count > 0) {
        // Draw `count` new edges into tempGraphics, then bake into renderTexture.
        // Edges are [x, y, theta] states — only x, y are used for the tree lines.
        tempGraphics.clear();
        tempGraphics.lineStyle(1.5, 0xC400B7, 1);

        for (let i = 0; i < count; i++) {
            const [p, q] = edges[edgeIndex];
            tempGraphics.moveTo(p[0], p[1]);
            tempGraphics.lineTo(q[0], q[1]);
            edgeIndex++;
        }

        // clear:false accumulates without erasing previous edges
        app.renderer.render(tempGraphics, { renderTexture, clear: false });
    }

    if (edgeIndex >= edges.length) {
        isAnimating = false;
        app.ticker.remove(animationStep);
        drawOverlay();
        runBtn.disabled = false;

        if (planData.path_found) {
            setStatus(
                `✅ Done — Path cost: ${planData.path_cost.toFixed(1)} | Nodes: ${planData.node_count}`
            );
            startRobotAnimation(planData.path, planData.robot_radius, parseInt(robotFpsInput.value));
        } else if (planData.stop_reason === 'max_time') {
            setStatus(
                `⏱️ Max planning time reached, no path found | Nodes: ${planData.node_count}`
            );
        } else {
            setStatus(
                `⚠️ Max nodes reached, no path found | Nodes: ${planData.node_count}`
            );
        }
    }
}

// ── Overlay (path + circles) ───────────────────────────────────────────────
function drawOverlay() {
    const d = planData;
    overlayGraphics.clear();

    // Path (blue) — [x, y, theta] states, only x, y are used for the line.
    if (d.path.length > 1) {
        overlayGraphics.lineStyle(4, 0x0B27DB, 1);
        overlayGraphics.moveTo(d.path[0][0], d.path[0][1]);
        for (let i = 1; i < d.path.length; i++) {
            overlayGraphics.lineTo(d.path[i][0], d.path[i][1]);
        }
    }

    // Goal circle (green)
    overlayGraphics.lineStyle(0);
    overlayGraphics.beginFill(d.path_found ? 0x0AD676 : 0x5CD676);
    overlayGraphics.drawCircle(d.x_goal[0], d.x_goal[1], d.goal_radius);
    overlayGraphics.endFill();

    // Start circle (yellow)
    overlayGraphics.beginFill(0xFFCF58);
    overlayGraphics.drawCircle(d.x_init[0], d.x_init[1], d.goal_radius);
    overlayGraphics.endFill();
}

// ── Robot animation ────────────────────────────────────────────────────────
// Draws the differential-drive robot as a circle (its footprint) with a heading line
// (center to edge, in the direction of travel) and a perpendicular axle line spanning
// the diameter, then steps it through `path` at `fps` states/second — mirroring
// DifferentialDriveRobotShape and PlanDrawer.animate_differential_drive_path() in the
// desktop app. Unlike PlanDrawer, no y-flip is needed here (see initCanvas's comment).
function startRobotAnimation(path, radius, fps) {
    if (!path || path.length === 0) return;

    robotPath = path;
    robotIndex = 0;
    robotElapsedMs = 0;
    robotRadius = radius;
    robotFps = fps;
    isAnimatingRobot = true;

    drawRobotAt(robotPath[0]);
    app.ticker.add(robotAnimationStep);
}

function robotAnimationStep() {
    if (!isAnimatingRobot || !robotPath) return;

    robotElapsedMs += app.ticker.deltaMS;
    const msPerState = 1000 / robotFps;

    if (robotElapsedMs < msPerState) return;
    robotElapsedMs = 0;
    robotIndex++;

    if (robotIndex >= robotPath.length) {
        isAnimatingRobot = false;
        app.ticker.remove(robotAnimationStep);
        return;
    }

    drawRobotAt(robotPath[robotIndex]);
}

function drawRobotAt(state) {
    const [x, y, theta] = state;

    const headingX = x + robotRadius * Math.cos(theta);
    const headingY = y + robotRadius * Math.sin(theta);
    const axleDx = robotRadius * Math.cos(theta + Math.PI / 2);
    const axleDy = robotRadius * Math.sin(theta + Math.PI / 2);

    robotGraphics.clear();

    // Body
    robotGraphics.lineStyle(0);
    robotGraphics.beginFill(0xDB540B);
    robotGraphics.drawCircle(x, y, robotRadius);
    robotGraphics.endFill();

    // Heading line
    robotGraphics.lineStyle(2, 0xFFFFFF, 1);
    robotGraphics.moveTo(x, y);
    robotGraphics.lineTo(headingX, headingY);

    // Axle line, perpendicular to heading, spanning the diameter
    robotGraphics.moveTo(x + axleDx, y + axleDy);
    robotGraphics.lineTo(x - axleDx, y - axleDy);
}

// ── Events ─────────────────────────────────────────────────────────────────
speedInput.addEventListener('input', () => {
    speedLabel.textContent = `${speedInput.value} edges / frame`;
});

robotFpsInput.addEventListener('input', () => {
    robotFpsLabel.textContent = `${robotFpsInput.value} states / s`;
});

runBtn.addEventListener('click', run);

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
