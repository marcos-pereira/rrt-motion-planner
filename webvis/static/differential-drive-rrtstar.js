// ── State ──────────────────────────────────────────────────────────────────
let app = null;         // PIXI.Application
let eventSource = null; // active SSE connection
let treeGraphics = null;   // cleared and redrawn each snapshot (rewiring changes parent pointers)
let overlayGraphics = null; // path + start/goal circles
let robotGraphics = null;   // differential-drive robot glyph (circle + heading + axle)
let initPromise = null; // resolves when the canvas is ready
let isRunning = false;

// Robot animation state — only played once, on the final ("done") snapshot, mirroring
// how the desktop app animates the robot after RRT* planning finishes (not on every
// intermediate rewiring snapshot, which would restart the animation constantly).
let robotPath = null;
let robotIndex = 0;
let robotElapsedMs = 0;
let robotRadius = 8;
let robotFps = 10;
let isAnimatingRobot = false;

// ── DOM refs ───────────────────────────────────────────────────────────────
const runBtn         = document.getElementById('run-btn');
const stopBtn        = document.getElementById('stop-btn');
const statusEl       = document.getElementById('status');
const mapSelect      = document.getElementById('map-name');
const batchInput     = document.getElementById('batch-size');
const batchLabel     = document.getElementById('batch-label');
const robotFpsInput  = document.getElementById('robot-fps');
const robotFpsLabel  = document.getElementById('robot-fps-label');
const canvasEl       = document.getElementById('canvas-container');
const placeholder    = document.getElementById('placeholder');

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
        x0:            parseFloat(document.getElementById('x0').value),
        y0:            parseFloat(document.getElementById('y0').value),
        xg:            parseFloat(document.getElementById('xg').value),
        yg:            parseFloat(document.getElementById('yg').value),
        batch_size:    parseInt(batchInput.value),
        gamma_rrt:     parseFloat(document.getElementById('gamma-rrt').value),
        eta:           parseFloat(document.getElementById('eta').value),
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
        app.ticker.remove(robotAnimationStep);
        app.destroy(true, { children: true, texture: true });
        app = treeGraphics = overlayGraphics = robotGraphics = null;
        const old = canvasEl.querySelector('canvas');
        if (old) old.remove();
    }
    initPromise = null;
    isRunning = false;
    isAnimatingRobot = false;
}

// ── Main flow ──────────────────────────────────────────────────────────────
async function run() {
    runBtn.disabled = true;
    stopBtn.disabled = false;
    destroyApp();
    placeholder.style.display = 'block';

    const params = getParams();
    const url = new URL('/plan-differential-drive-rrtstar', window.location.origin);
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

            if (snap.path_found) {
                startRobotAnimation(snap.path, snap.robot_radius, parseInt(robotFpsInput.value));
            }
        }
    };

    eventSource.onerror = () => {
        if (isRunning) setStatus('❌ Stream error or connection closed.');
        finish();
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

    // Robot glyph sits on top of everything else.
    robotGraphics = new PIXI.Graphics();
    app.stage.addChild(robotGraphics);

    placeholder.style.display = 'none';
}

// ── Drawing ────────────────────────────────────────────────────────────────
function drawSnapshot(snap) {
    if (!app || !treeGraphics || !overlayGraphics) return;

    // Redraw all current tree edges from scratch. Edges are [x, y, theta] states —
    // only x, y are used for the tree lines.
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

// ── Robot animation ────────────────────────────────────────────────────────
// Draws the differential-drive robot as a circle (its footprint) with a heading line
// (center to edge, in the direction of travel) and a perpendicular axle line spanning
// the diameter, then steps it through `path` at `fps` states/second — mirroring
// DifferentialDriveRobotShape and PlanDrawer.animate_differential_drive_path() in the
// desktop app. Unlike PlanDrawer, no y-flip is needed here (see initCanvas's comment).
function startRobotAnimation(path, radius, fps) {
    if (!path || path.length === 0 || !robotGraphics) return;

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
batchInput.addEventListener('input', () => {
    batchLabel.textContent = `${batchInput.value} steps`;
});

robotFpsInput.addEventListener('input', () => {
    robotFpsLabel.textContent = `${robotFpsInput.value} states / s`;
});

runBtn.addEventListener('click', run);
stopBtn.addEventListener('click', stop);

// ── Init ───────────────────────────────────────────────────────────────────
loadMapList();
