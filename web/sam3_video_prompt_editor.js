/*
 * SAM3 Video Prompt Editor
 *
 * Modified version: Shift/Ctrl modifiers used to draw points and boxes.
 * Dropdown prompt_mode is hidden to simplify the UI.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "SAM3VideoPromptEditor";
const INIT_NODE_TYPES = new Set([
    "SAM3InitVideoSession",
    "SAM3InitVideoSessionAdvanced",
]);

const HANDLE_SIZE = 8;
const POINT_RADIUS = 5;
const HIT_PAD = 10;
const SLIDER_H = 32;
const PAD = 8;

const INSET_LEFT = 6;
const INSET_RIGHT = 12;
const INSET_TOP = 4;
const INSET_BOTTOM = 14;

const MIN_NODE_W = 420;
const DEFAULT_W = 560;
const DEFAULT_H = 620;

function parseJSON(s, fb) {
    try { return s ? JSON.parse(s) : fb; } catch { return fb; }
}
function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}
function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    w.computeSize = () => [0, -4];
    w.type = "hidden";
}
function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
}
function badgeRadius(label) {
    const s = String(label);
    return s.length > 2 ? 10 : s.length > 1 ? 9 : 8;
}

function findUpstreamInitNode(node) {
    if (!node.inputs) return null;
    for (const inp of node.inputs) {
        if (inp.type !== "SAM3_VIDEO_SESSION" || inp.link == null) continue;
        const link = node.graph.links[inp.link];
        if (!link) continue;
        const upstream = node.graph.getNodeById(link.origin_id);
        if (!upstream) continue;
        if (INIT_NODE_TYPES.has(upstream.type)) return upstream;
        const deeper = findUpstreamInitNode(upstream);
        if (deeper) return deeper;
    }
    return null;
}

function findImageProducer(node, seen = new Set()) {
    if (!node || seen.has(node.id)) return null;
    seen.add(node.id);
    if (node.widgets) {
        for (const w of node.widgets) {
            if (
                (w.name === "video" || w.name === "image" || w.name === "file" || w.name === "upload") &&
                typeof w.value === "string" && w.value
            ) return w.value;
        }
    }
    if (!node.inputs) return null;
    for (const inp of node.inputs) {
        if (inp.link == null) continue;
        const link = node.graph.links[inp.link];
        if (!link) continue;
        const up = node.graph.getNodeById(link.origin_id);
        const res = findImageProducer(up, seen);
        if (res) return res;
    }
    return null;
}

function findUpstreamVideoFilename(node) {
    if (node && node.inputs) {
        for (const inp of node.inputs) {
            if (inp.name === "image" && inp.link != null) {
                const link = node.graph.links[inp.link];
                if (link) {
                    const up = node.graph.getNodeById(link.origin_id);
                    const fn = findImageProducer(up);
                    if (fn) return fn;
                }
            }
        }
    }
    const init = findUpstreamInitNode(node);
    if (init) return findImageProducer(init);
    return null;
}

async function apiJson(url, opts = {}) {
    try {
        const r = await api.fetchApi(url, opts);
        if (!r.ok) return null;
        return await r.json();
    } catch { return null; }
}
async function requestPreviewPrepare(preview_key, filename) {
    return apiJson("/sam3/prepare_frames", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preview_key, filename }),
    });
}
async function getPreviewFrame(key, i) { return apiJson(`/sam3/preview_frames/${key}/${i}`); }
async function getSessionMeta(sid)     { return apiJson(`/sam3/video_frames/${sid}`); }
async function getSessionFrame(sid, i) { return apiJson(`/sam3/video_frames/${sid}/${i}`); }
async function listSessions()          { return apiJson(`/sam3/video_frames/list`); }

function graphToClient(gx, gy) {
    const c = app.canvas;
    const ds = c.ds;
    const rect = c.canvas.getBoundingClientRect();
    const cx = (gx + ds.offset[0]) * ds.scale;
    const cy = (gy + ds.offset[1]) * ds.scale;
    const sx = rect.width / Math.max(1, c.canvas.width);
    const sy = rect.height / Math.max(1, c.canvas.height);
    return {
        x: rect.left + cx * sx,
        y: rect.top + cy * sy,
    };
}

function contentTop(node) {
    const title = LiteGraph?.NODE_TITLE_HEIGHT || 30;
    let y = title + 4;
    const slotH = LiteGraph?.NODE_SLOT_HEIGHT || 20;
    const nIn = node.inputs?.length || 0;
    y = Math.max(y, title + nIn * slotH);
    if (node.widgets_start_y != null) y = Math.max(y, node.widgets_start_y);
    if (node.widgets) {
        let wy = node.widgets_start_y != null ? node.widgets_start_y : y;
        for (const w of node.widgets) {
            if (w.hidden) continue;
            const sz = w.computeSize ? w.computeSize(node.size[0]) : [0, LiteGraph?.NODE_WIDGET_HEIGHT || 20];
            if ((sz[1] || 0) > 0) wy += sz[1] + 4;
        }
        y = Math.max(y, wy);
    }
    return y + 4;
}

app.registerExtension({
    name: "SAM3.VideoPromptEditor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (INIT_NODE_TYPES.has(nodeData.name)) {
            const onExec = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (msg) {
                if (msg?.session_id?.[0]) {
                    this._sam3_last_sid = msg.session_id[0];
                    if (this.graph) {
                        for (const n of this.graph._nodes) {
                            if (n.type === NODE_TYPE && n._sam3?.onUpstreamUpdate) {
                                n._sam3.onUpstreamUpdate();
                            }
                        }
                    }
                }
                return onExec ? onExec.apply(this, arguments) : undefined;
            };
            return;
        }

        if (nodeData.name !== NODE_TYPE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated ? onCreated.apply(this, arguments) : undefined;
            const node = this;

            const wPoints = findWidget(node, "points_json");
            const wBoxes  = findWidget(node, "boxes_json");
            const wCache  = findWidget(node, "ui_cache_key");
            const wFrame  = findWidget(node, "frame_index");
            const wMode   = findWidget(node, "prompt_mode");
            hideWidget(wPoints);
            hideWidget(wBoxes);
            hideWidget(wCache);
            hideWidget(wMode); // Скрываем выпадающий список за ненадобностью

            if (!node.size || node.size[0] < MIN_NODE_W) {
                node.size = [DEFAULT_W, DEFAULT_H];
            }

            const overlay = document.createElement("div");
            Object.assign(overlay.style, {
                position: "fixed",
                left: "0", top: "0",
                width: "100px", height: "100px",
                zIndex: "60",
                pointerEvents: "auto",
                display: "none",
                overflow: "hidden",
                borderRadius: "6px",
                boxShadow: "0 0 0 1px rgba(255,255,255,0.08)",
                transformOrigin: "0 0",
            });
            overlay.tabIndex = 0;

            const canvas = document.createElement("canvas");
            Object.assign(canvas.style, {
                position: "absolute", left: "0", top: "0",
                width: "100%", height: "100%",
                cursor: "crosshair",
                touchAction: "none",
                display: "block",
            });
            overlay.appendChild(canvas);
            document.body.appendChild(overlay);

            const ctx = canvas.getContext("2d");

            const state = {
                frames: {},
                sid: null,
                previewKey: null,
                curIdx: 0,
                numFrames: 0,
                imgW: 0,
                imgH: 0,

                points: parseJSON(wPoints?.value, []),
                boxes: parseJSON(wBoxes?.value, []),

                scale: 1,
                offsetX: 0,
                offsetY: 0,
                fittedOnce: false,

                drawingBox: null,
                dragTarget: null,
                dragStart: null,
                panning: false,
                panStart: null,
                hover: null,
                loadingIdx: null,
                lastFilename: null,

                viewH: 100,
                ow: 100,
                oh: 100,

                sliderDragging: false,
                btnPrev: { x: 0, y: 0, w: 30, h: 24 },
                btnNext: { x: 0, y: 0, w: 30, h: 24 },
                track: { x: 0, y: 0, w: 100, h: 10 },
                labelX: 0,
                labelY: 0,

                visible: false,
                raf: 0,
            };
            node._sam3 = state;

            function maxFrame() { return Math.max(0, (state.numFrames || 1) - 1); }

            function indexInType(kind, idx) {
                if (kind === "pos_pt" || kind === "neg_pt") {
                    const want = kind === "pos_pt" ? 1 : 0;
                    let n = 0;
                    for (let i = 0; i < idx; i++) {
                        if ((state.points[i].label ?? 1) === want) n++;
                    }
                    return n;
                }
                const wantPos = kind === "pos_box";
                let n = 0;
                for (let i = 0; i < idx; i++) {
                    const pos = state.boxes[i].positive !== false;
                    if (pos === wantPos) n++;
                }
                return n;
            }

            function layoutSlider() {
                const y = state.oh - SLIDER_H;
                const by = y + (SLIDER_H - 24) * 0.5;
                state.btnPrev = { x: 6, y: by, w: 30, h: 24 };
                state.btnNext = { x: state.ow - 36, y: by, w: 30, h: 24 };
                const labelW = 70;
                const tx = state.btnPrev.x + state.btnPrev.w + 10;
                const tr = state.btnNext.x - 10 - labelW;
                state.track = {
                    x: tx,
                    y: y + SLIDER_H * 0.5 - 5,
                    w: Math.max(20, tr - tx),
                    h: 10,
                };
                state.labelX = tr + 8;
                state.labelY = by + 12;
                state.viewH = Math.max(40, y - 2);
            }

            function syncOverlayTransform() {
                if (!node.graph || node.flags?.collapsed) {
                    overlay.style.display = "none";
                    state.visible = false;
                    return false;
                }

                const topLocal = contentTop(node) + INSET_TOP;
                const leftG = node.pos[0] + INSET_LEFT;
                const topG = node.pos[1] + topLocal;
                const wG = Math.max(40, node.size[0] - INSET_LEFT - INSET_RIGHT);
                const hG = Math.max(40, node.size[1] - topLocal - INSET_BOTTOM);

                const tl = graphToClient(leftG, topG);
                const br = graphToClient(leftG + wG, topG + hG);
                const w = Math.max(40, br.x - tl.x);
                const h = Math.max(40, br.y - tl.y);

                overlay.style.display = "block";
                overlay.style.left = `${tl.x}px`;
                overlay.style.top = `${tl.y}px`;
                overlay.style.width = `${w}px`;
                overlay.style.height = `${h}px`;
                overlay.style.transform = "none";

                const sizeChanged = Math.abs(w - state.ow) > 0.5 || Math.abs(h - state.oh) > 0.5;
                state.ow = w;
                state.oh = h;
                state.visible = true;
                layoutSlider();

                if (sizeChanged && state.imgW && state.fittedOnce) {
                    fitToView(false);
                }
                return true;
            }

            function syncBuffer() {
                const dpr = window.devicePixelRatio || 1;
                const bw = Math.max(1, Math.round(state.ow * dpr));
                const bh = Math.max(1, Math.round(state.oh * dpr));
                if (canvas.width !== bw || canvas.height !== bh) {
                    canvas.width = bw;
                    canvas.height = bh;
                }
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            }

            function fitToView(redraw = true) {
                if (!state.imgW || !state.imgH) return;
                layoutSlider();
                const aw = Math.max(10, state.ow - PAD * 2);
                const ah = Math.max(10, state.viewH - PAD * 2);
                const s = Math.min(aw / state.imgW, ah / state.imgH);
                state.scale = s;
                state.offsetX = PAD + (aw - state.imgW * s) * 0.5;
                state.offsetY = PAD + (ah - state.imgH * s) * 0.5;
                state.fittedOnce = true;
                if (redraw) draw();
            }

            function imgToView(x, y) {
                return [state.offsetX + x * state.scale, state.offsetY + y * state.scale];
            }
            function viewToImg(vx, vy) {
                const sc = Math.max(1e-8, state.scale);
                return [(vx - state.offsetX) / sc, (vy - state.offsetY) / sc];
            }

            function localXY(ev) {
                const rect = overlay.getBoundingClientRect();
                const rw = Math.max(1e-6, rect.width);
                const rh = Math.max(1e-6, rect.height);
                return [
                    ((ev.clientX - rect.left) * state.ow) / rw,
                    ((ev.clientY - rect.top) * state.oh) / rh,
                ];
            }

            function inside(nx, ny, r) {
                return nx >= r.x && nx <= r.x + r.w && ny >= r.y && ny <= r.y + r.h;
            }

            function thumbX() {
                const t = state.track;
                const max = maxFrame();
                const ratio = max > 0 ? state.curIdx / max : 0;
                return t.x + ratio * t.w;
            }
            function frameFromX(nx) {
                const t = state.track;
                const max = maxFrame();
                if (max <= 0) return 0;
                return Math.round(clamp((nx - t.x) / Math.max(1, t.w), 0, 1) * max);
            }

            function draw() {
                if (!state.visible) return;
                syncBuffer();
                layoutSlider();
                const w = state.ow, h = state.oh;

                ctx.clearRect(0, 0, w, h);
                ctx.fillStyle = "#141414";
                ctx.fillRect(0, 0, w, h);

                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, w, state.viewH);
                ctx.clip();

                if (state.imgW && state.imgH && !state.fittedOnce) fitToView(false);

                const img = state.frames[state.curIdx];
                if (img && state.imgW) {
                    ctx.drawImage(
                        img,
                        state.offsetX, state.offsetY,
                        state.imgW * state.scale, state.imgH * state.scale
                    );
                } else {
                    ctx.fillStyle = "#888";
                    ctx.font = "13px sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(
                        state.loadingIdx != null
                            ? `Loading ${state.loadingIdx}...`
                            : (state.sid || state.previewKey
                                ? "Waiting for frame..."
                                : "Connect Init Session or Image input"),
                        w * 0.5, state.viewH * 0.5
                    );
                }

                for (let i = 0; i < state.boxes.length; i++) drawBox(state.boxes[i], i);
                if (state.drawingBox) drawBox(state.drawingBox, -1, true);
                
                for (let i = 0; i < state.points.length; i++) drawPoint(state.points[i], i);
                
                ctx.restore();

                const sy = h - SLIDER_H;
                ctx.fillStyle = "rgba(0,0,0,0.72)";
                ctx.fillRect(0, sy, w, SLIDER_H);

                drawBtn(state.btnPrev, "◀");
                drawBtn(state.btnNext, "▶");

                const t = state.track;
                ctx.fillStyle = "rgba(255,255,255,0.15)";
                roundRect(t.x, t.y, t.w, t.h, 5); ctx.fill();
                const tx = thumbX();
                const fw = Math.max(0, tx - t.x);
                if (fw > 0) {
                    ctx.fillStyle = "rgba(80,160,255,0.55)";
                    roundRect(t.x, t.y, fw, t.h, 5); ctx.fill();
                }
                ctx.beginPath();
                ctx.arc(tx, t.y + t.h * 0.5, 8, 0, Math.PI * 2);
                ctx.fillStyle = "#4aa3ff";
                ctx.fill();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = "#ddd";
                ctx.font = "bold 12px sans-serif";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(`${state.curIdx} / ${maxFrame()}`, state.labelX, state.labelY);
            }

            function roundRect(x, y, w, h, r) {
                const rr = Math.min(r, w * 0.5, h * 0.5);
                ctx.beginPath();
                ctx.moveTo(x + rr, y);
                ctx.arcTo(x + w, y, x + w, y + h, rr);
                ctx.arcTo(x + w, y + h, x, y + h, rr);
                ctx.arcTo(x, y + h, x, y, rr);
                ctx.arcTo(x, y, x + w, y, rr);
                ctx.closePath();
            }
            function drawBtn(r, text) {
                ctx.fillStyle = "#2a2a2a";
                roundRect(r.x, r.y, r.w, r.h, 4); ctx.fill();
                ctx.strokeStyle = "#555"; ctx.lineWidth = 1; ctx.stroke();
                ctx.fillStyle = "#eee";
                ctx.font = "13px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(text, r.x + r.w * 0.5, r.y + r.h * 0.5 + 0.5);
            }

            function drawIndexBadge(vx, vy, rad, color, index, hot) {
                const label = String(index);
                const br = badgeRadius(label);
                const off = rad * 0.85 + 2;
                let bx = vx + off;
                let by = vy - off;
                if (bx + br > state.ow - 2) bx = vx - off;
                if (by - br < 2) by = vy + off;

                ctx.beginPath();
                ctx.arc(bx, by, br + (hot ? 1 : 0), 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.fillStyle = "#fff";
                ctx.font = "bold 10px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(label, bx, by + 0.5);
            }

            function drawPoint(p, idx) {
                const [vx, vy] = imgToView(p.x, p.y);
                const rad = Math.max(4, POINT_RADIUS);
                const isPos = (p.label ?? 1) === 1;
                const color = isPos ? "#3fdc3f" : "#ff3f3f";
                const hot = state.hover?.type === "point" && state.hover.idx === idx;
                const typeIdx = indexInType(isPos ? "pos_pt" : "neg_pt", idx);

                ctx.beginPath();
                ctx.arc(vx, vy, rad + (hot ? 3 : 0), 0, Math.PI * 2);
                ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke();

                ctx.beginPath(); ctx.arc(vx, vy, 2, 0, Math.PI * 2);
                ctx.fillStyle = "#ff5050"; ctx.fill();

                drawIndexBadge(vx, vy, rad, color, typeIdx, hot);
            }

            function getBoxHandles(b) {
                const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1);
                const y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
                const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
                return [
                    { x: x0, y: y0, id: "nw" }, { x: cx, y: y0, id: "n" }, { x: x1, y: y0, id: "ne" },
                    { x: x1, y: cy, id: "e" },
                    { x: x1, y: y1, id: "se" }, { x: cx, y: y1, id: "s" }, { x: x0, y: y1, id: "sw" },
                    { x: x0, y: cy, id: "w" },
                ];
            }

            function drawBox(b, idx, preview = false) {
                const [x0, y0] = imgToView(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1));
                const [x1, y1] = imgToView(Math.max(b.x0, b.x1), Math.max(b.y0, b.y1));
                const isPos = b.positive !== false;
                const color = isPos ? "#4aa3ff" : "#ff8040";
                const hot = state.hover?.type === "box" && state.hover.idx === idx;

                ctx.save();
                ctx.fillStyle = isPos ? "rgba(74,163,255,0.12)" : "rgba(255,128,64,0.18)";
                ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                ctx.strokeStyle = color;
                ctx.lineWidth = hot ? 3 : 2;
                if (preview) ctx.setLineDash([6, 4]);
                ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                ctx.setLineDash([]);

                if (!preview) {
                    ctx.fillStyle = color;
                    for (const h of getBoxHandles(b)) {
                        const [hx, hy] = imgToView(h.x, h.y);
                        ctx.fillRect(hx - HANDLE_SIZE / 2, hy - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
                    }
                    if (idx >= 0) {
                        const typeIdx = indexInType(isPos ? "pos_box" : "neg_box", idx);
                        const label = String(typeIdx);
                        const br = badgeRadius(label);
                        const bx = x0 + br + 2;
                        const by = y0 + br + 2;
                        ctx.beginPath();
                        ctx.arc(bx, by, br, 0, Math.PI * 2);
                        ctx.fillStyle = color;
                        ctx.fill();
                        ctx.fillStyle = "#fff";
                        ctx.font = "bold 10px sans-serif";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText(label, bx, by + 0.5);
                    }
                }
                ctx.restore();
            }

            function hitTest(ix, iy) {
                const sc = Math.max(1e-6, state.scale);
                const hitImg = HIT_PAD / sc;
                
                for (let i = state.points.length - 1; i >= 0; i--) {
                    const p = state.points[i];
                    const dx = p.x - ix, dy = p.y - iy;
                    const rad = hitImg + POINT_RADIUS / sc;
                    if (dx * dx + dy * dy <= rad * rad) return { type: "point", idx: i };
                }
                const hR = (HANDLE_SIZE / 2 + HIT_PAD) / sc;
                for (let i = state.boxes.length - 1; i >= 0; i--) {
                    const b = state.boxes[i];
                    for (const h of getBoxHandles(b)) {
                        if (Math.abs(h.x - ix) <= hR && Math.abs(h.y - iy) <= hR)
                            return { type: "box", idx: i, handle: h.id };
                    }
                }
                for (let i = state.boxes.length - 1; i >= 0; i--) {
                    const b = state.boxes[i];
                    const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1);
                    const y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
                    if (ix >= x0 && ix <= x1 && iy >= y0 && iy <= y1)
                        return { type: "box", idx: i, handle: "move" };
                }
                return null;
            }

            function commitState() {
                if (wPoints) wPoints.value = JSON.stringify(state.points);
                if (wBoxes) wBoxes.value = JSON.stringify(state.boxes);
                draw();
            }

            function setFrame(idx) {
                loadFrameFromCurrentSource(clamp(idx | 0, 0, maxFrame()));
            }

            async function loadFrameFromCurrentSource(idx) {
                if (idx == null) idx = state.curIdx;
                idx = clamp(idx | 0, 0, maxFrame());
                if (state.frames[idx]) {
                    state.curIdx = idx;
                    if (wFrame) wFrame.value = idx;
                    draw();
                    return;
                }
                state.loadingIdx = idx;
                draw();
                let data = null;
                if (state.sid) data = await getSessionFrame(state.sid, idx);
                else if (state.previewKey) data = await getPreviewFrame(state.previewKey, idx);
                state.loadingIdx = null;
                if (!data?.b64) { return; }
                const im = new Image();
                im.onload = () => {
                    state.frames[idx] = im;
                    state.imgW = data.w;
                    state.imgH = data.h;
                    state.curIdx = idx;
                    if (wFrame) wFrame.value = idx;
                    if (!state.fittedOnce) fitToView();
                    else draw();
                };
                im.src = "data:image/jpeg;base64," + data.b64;
            }

            async function refreshFromUpstream() {
                const filename = findUpstreamVideoFilename(node);
                if (filename) {
                    const key = `p_fn_${filename.replace(/[^a-zA-Z0-9]/g, "_")}`;
                    if (state.previewKey === key && state.numFrames > 0) {
                        await loadFrameFromCurrentSource(state.curIdx);
                        return;
                    }
                    const prep = await requestPreviewPrepare(key, filename);
                    if (prep && !prep.error) {
                        state.sid = null;
                        state.previewKey = key;
                        state.numFrames = prep.n;
                        state.imgW = prep.w; state.imgH = prep.h;
                        state.frames = {};
                        state.fittedOnce = false;
                        if (state.curIdx >= prep.n) state.curIdx = 0;
                        state.lastFilename = filename;
                        await loadFrameFromCurrentSource(state.curIdx);
                        return;
                    }
                }

                const init = findUpstreamInitNode(node);
                if (init) {
                    let sid = init._sam3_last_sid || null;
                    if (!sid) {
                        const lst = await listSessions();
                        if (lst?.sessions?.length) sid = lst.sessions[lst.sessions.length - 1].sid;
                    }
                    if (sid) {
                        const meta = await getSessionMeta(sid);
                        if (meta) {
                            if (state.sid !== sid) {
                                state.sid = sid;
                                state.previewKey = null;
                                state.numFrames = meta.n;
                                state.frames = {};
                                state.imgW = meta.w; state.imgH = meta.h;
                                state.fittedOnce = false;
                                if (state.curIdx >= meta.n) state.curIdx = 0;
                            }
                            await loadFrameFromCurrentSource(state.curIdx);
                            return;
                        }
                    }
                }

                state.sid = null; state.previewKey = null;
                state.numFrames = 0; state.frames = {};
                draw();
            }
            state.onUpstreamUpdate = refreshFromUpstream;

            overlay.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                e.stopPropagation();
            });

            overlay.addEventListener("wheel", (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (app.canvas?.processMouseWheel) {
                    app.canvas.processMouseWheel(e);
                } else if (app.canvas?.canvas) {
                    app.canvas.canvas.dispatchEvent(new WheelEvent("wheel", e));
                }
            }, { passive: false });

            overlay.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                e.stopPropagation();
                try { overlay.setPointerCapture(e.pointerId); } catch {}
                overlay.focus();

                const [mx, my] = localXY(e);
                layoutSlider();

                if (my >= state.oh - SLIDER_H) {
                    if (e.button === 0) {
                        if (inside(mx, my, state.btnPrev)) { setFrame(state.curIdx - 1); return; }
                        if (inside(mx, my, state.btnNext)) { setFrame(state.curIdx + 1); return; }
                        state.sliderDragging = true;
                        setFrame(frameFromX(mx));
                    }
                    return;
                }

                const [ix, iy] = viewToImg(mx, my);

                if (e.button === 1) {
                    state.panning = true;
                    state.panStart = { mx, my, ox: state.offsetX, oy: state.offsetY };
                    return;
                }

                // КНОПКИ МОДИФИКАТОРЫ (Быстрое рисование без переключения модов)
                const isCtrl = e.ctrlKey || e.metaKey;
                const isShift = e.shiftKey;

                // 1. Рисование Точек (Ctrl + ЛКМ / ПКМ)
                if (isCtrl) {
                    const positive = e.button !== 2;
                    state.points.push({ x: ix, y: iy, label: positive ? 1 : 0, img_w: state.imgW, img_h: state.imgH });
                    commitState();
                    return;
                }

                // 2. Рисование Боксов (Shift + ЛКМ / ПКМ)
                if (isShift) {
                    const positive = e.button !== 2;
                    state.drawingBox = {
                        x0: ix, y0: iy, x1: ix, y1: iy, positive,
                    };
                    draw();
                    return;
                }

                const hit = hitTest(ix, iy);

                if (e.button === 2) {
                    if (hit) {
                        if (hit.type === "point") state.points.splice(hit.idx, 1);
                        else state.boxes.splice(hit.idx, 1);
                        commitState();
                    }
                    return;
                }

                if (e.button === 0 && hit) {
                    state.dragTarget = hit;
                    state.dragStart = {
                        mx: ix, my: iy,
                        orig: hit.type === "point"
                            ? { ...state.points[hit.idx] }
                            : { ...state.boxes[hit.idx] },
                    };
                }
            });

            function onMove(e) {
                if (!state.visible) return;
                const [mx, my] = localXY(e);

                if (state.sliderDragging) {
                    setFrame(frameFromX(mx));
                    return;
                }
                if (state.panning && state.panStart) {
                    state.offsetX = state.panStart.ox + (mx - state.panStart.mx);
                    state.offsetY = state.panStart.oy + (my - state.panStart.my);
                    draw();
                    return;
                }
                if (state.drawingBox) {
                    const [ix, iy] = viewToImg(mx, my);
                    state.drawingBox.x1 = ix;
                    state.drawingBox.y1 = iy;
                    draw();
                    return;
                }
                if (state.dragTarget && state.dragStart) {
                    const [ix, iy] = viewToImg(mx, my);
                    if (state.dragTarget.type === "point") {
                        const p = state.points[state.dragTarget.idx];
                        p.x = state.dragStart.orig.x + (ix - state.dragStart.mx);
                        p.y = state.dragStart.orig.y + (iy - state.dragStart.my);
                        p.img_w = state.imgW;
                        p.img_h = state.imgH;
                    } else {
                        const b = state.boxes[state.dragTarget.idx];
                        const o = state.dragStart.orig;
                        const dx = ix - state.dragStart.mx;
                        const dy = iy - state.dragStart.my;
                        const h = state.dragTarget.handle;
                        if (h === "move") {
                            b.x0 = o.x0 + dx; b.x1 = o.x1 + dx;
                            b.y0 = o.y0 + dy; b.y1 = o.y1 + dy;
                        } else {
                            const nx0 = Math.min(o.x0, o.x1), nx1 = Math.max(o.x0, o.x1);
                            const ny0 = Math.min(o.y0, o.y1), ny1 = Math.max(o.y0, o.y1);
                            let X0 = nx0, X1 = nx1, Y0 = ny0, Y1 = ny1;
                            if (h.includes("w")) X0 = nx0 + dx;
                            if (h.includes("e")) X1 = nx1 + dx;
                            if (h.includes("n")) Y0 = ny0 + dy;
                            if (h.includes("s")) Y1 = ny1 + dy;
                            b.x0 = X0; b.x1 = X1; b.y0 = Y0; b.y1 = Y1;
                        }
                        b.img_w = state.imgW;
                        b.img_h = state.imgH;
                    }
                    draw();
                    return;
                }

                if (my < state.viewH) {
                    const [ix, iy] = viewToImg(mx, my);
                    const hit = hitTest(ix, iy);
                    if (JSON.stringify(hit) !== JSON.stringify(state.hover)) {
                        state.hover = hit;
                        canvas.style.cursor = hit
                            ? (hit.handle && hit.handle !== "move" ? "nwse-resize" : "move")
                            : "crosshair";
                        draw();
                    }
                }
            }

            function onUp(e) {
                try { overlay.releasePointerCapture?.(e.pointerId); } catch {}
                if (state.sliderDragging) { state.sliderDragging = false; return; }
                if (state.panning) { state.panning = false; state.panStart = null; return; }
                if (state.drawingBox) {
                    const b = state.drawingBox;
                    if (Math.abs(b.x1 - b.x0) > 3 && Math.abs(b.y1 - b.y0) > 3) {
                        state.boxes.push({
                            x0: Math.min(b.x0, b.x1), y0: Math.min(b.y0, b.y1),
                            x1: Math.max(b.x0, b.x1), y1: Math.max(b.y0, b.y1),
                            positive: b.positive !== false,
                            img_w: state.imgW,
                            img_h: state.imgH,
                        });
                        commitState();
                    }
                    state.drawingBox = null;
                    draw();
                    return;
                }
                if (state.dragTarget) {
                    state.dragTarget = null;
                    state.dragStart = null;
                    for (const b of state.boxes) {
                        const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1);
                        const y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
                        b.x0 = x0; b.x1 = x1; b.y0 = y0; b.y1 = y1;
                        b.img_w = state.imgW; b.img_h = state.imgH;
                    }
                    commitState();
                }
            }

            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);

            overlay.addEventListener("dblclick", (e) => {
                e.preventDefault();
                const [, my] = localXY(e);
                if (my < state.viewH) fitToView();
            });

            overlay.addEventListener("keydown", (e) => {
                if (e.key === "f" || e.key === "F") { fitToView(); e.preventDefault(); }
                else if (e.key === "ArrowLeft") { setFrame(state.curIdx - 1); e.preventDefault(); }
                else if (e.key === "ArrowRight") { setFrame(state.curIdx + 1); e.preventDefault(); }
                else if ((e.key === "Delete" || e.key === "Backspace") && state.hover) {
                    const hit = state.hover;
                    if (hit.type === "point") state.points.splice(hit.idx, 1);
                    else state.boxes.splice(hit.idx, 1);
                    state.hover = null;
                    commitState();
                    e.preventDefault();
                }
            });

            function tick() {
                state.raf = requestAnimationFrame(tick);
                if (!node.graph) return;
                const was = state.ow + "x" + state.oh;
                if (syncOverlayTransform()) {
                    if (was !== state.ow + "x" + state.oh) draw();
                }
            }
            state.raf = requestAnimationFrame(tick);

            const _onDrawBg = node.onDrawBackground;
            node.onDrawBackground = function () {
                syncOverlayTransform();
                draw();
                return _onDrawBg ? _onDrawBg.apply(this, arguments) : undefined;
            };

            const _onResize = node.onResize;
            node.onResize = function () {
                const ret = _onResize ? _onResize.apply(this, arguments) : undefined;
                syncOverlayTransform();
                fitToView();
                return ret;
            };

            const onConnChange = node.onConnectionsChange;
            node.onConnectionsChange = function () {
                const ret = onConnChange ? onConnChange.apply(this, arguments) : undefined;
                setTimeout(refreshFromUpstream, 60);
                return ret;
            };

            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                const ret = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                state.points = parseJSON(wPoints?.value, []);
                state.boxes = parseJSON(wBoxes?.value, []);
                state.fittedOnce = false;
                setTimeout(() => {
                    refreshFromUpstream();
                    syncOverlayTransform();
                    fitToView();
                }, 120);
                return ret;
            };

            const onExecuted = node.onExecuted;
            node.onExecuted = function (msg) {
                if (msg?.session_id?.[0] && state.sid !== msg.session_id[0]) {
                    state.sid = msg.session_id[0];
                    state.previewKey = null;
                    state.frames = {};
                    state.fittedOnce = false;
                    refreshFromUpstream();
                }
                return onExecuted ? onExecuted.apply(this, arguments) : undefined;
            };

            node.onRemoved = ((prev) => function () {
                cancelAnimationFrame(state.raf);
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                overlay.remove();
                return prev ? prev.apply(this, arguments) : undefined;
            })(node.onRemoved);

            setTimeout(() => {
                syncOverlayTransform();
                refreshFromUpstream();
                fitToView();
            }, 100);

            setInterval(() => {
                const fn = findUpstreamVideoFilename(node);
                if (fn && fn !== state.lastFilename) refreshFromUpstream();
            }, 1500);

            return r;
        };
    },
});
