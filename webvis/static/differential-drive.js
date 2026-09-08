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
    app.destroy(true, { children: true, texture: true });
    app = renderTexture = edgeSprite = tempGraphics = overlayGraphics = robotGraphics = null;
    isAnimatingRobot = false;
    // Remove the canvas element the old app appended
    const old = canvasEl.querySelector('canvas');
    if (old) old.remove();
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

// ── Init ───────────────────────────────────────────────────────────────────
loadMapList();
