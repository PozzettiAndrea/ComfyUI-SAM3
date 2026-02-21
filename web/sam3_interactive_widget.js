/**
 * SAM3 Interactive Collector Widget
 * Multi-region prompt collection with live segmentation preview via "Run" button.
 * Based on sam3_multiregion_widget.js
 */

import { app } from "../../scripts/app.js";

console.log("[SAM3] ===== INTERACTIVE COLLECTOR VERSION 1 =====");

const PROMPT_COLORS = [
    { name: "cyan",    primary: "#00FFFF", dim: "#006666" },
    { name: "yellow",  primary: "#FFFF00", dim: "#666600" },
    { name: "magenta", primary: "#FF00FF", dim: "#660066" },
    { name: "lime",    primary: "#00FF00", dim: "#006600" },
    { name: "orange",  primary: "#FF8000", dim: "#663300" },
    { name: "pink",    primary: "#FF69B4", dim: "#662944" },
    { name: "blue",    primary: "#4169E1", dim: "#1a2a5c" },
    { name: "teal",    primary: "#20B2AA", dim: "#0d4744" },
];

const MAX_PROMPTS = PROMPT_COLORS.length;

function hideWidgetForGood(node, widget, suffix = '') {
    if (!widget) return;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget" + suffix;
    widget.hidden = true;
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }
}

app.registerExtension({
    name: "Comfy.SAM3.InteractiveCollector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3InteractiveCollector") return;

        console.log("[SAM3] Registering SAM3InteractiveCollector node");
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // ---- DOM structure ----
            const container = document.createElement("div");
            container.style.cssText = "position: relative; width: 100%; background: #222; overflow: hidden; box-sizing: border-box; display: flex; flex-direction: column;";

            // Info bar (top)
            const infoBar = document.createElement("div");
            infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between; align-items: center;";
            container.appendChild(infoBar);

            // Counter
            const counter = document.createElement("div");
            counter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
            counter.textContent = "Prompt 1: 0 pts, 0 boxes";
            infoBar.appendChild(counter);

            // Button container
            const buttonContainer = document.createElement("div");
            buttonContainer.style.cssText = "display: flex; gap: 5px;";
            infoBar.appendChild(buttonContainer);

            // ---- RUN button (the key addition) ----
            const runBtn = document.createElement("button");
            runBtn.textContent = "Run";
            runBtn.style.cssText = "padding: 5px 14px; background: #2a7a2a; color: #fff; border: 1px solid #3a9a3a; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: bold;";
            runBtn.onmouseover = () => { if (!runBtn.disabled) runBtn.style.background = "#3a9a3a"; };
            runBtn.onmouseout = () => { if (!runBtn.disabled) runBtn.style.background = "#2a7a2a"; };
            buttonContainer.appendChild(runBtn);

            // Clear Prompt button
            const clearPromptBtn = document.createElement("button");
            clearPromptBtn.textContent = "Clear Prompt";
            clearPromptBtn.style.cssText = "padding: 5px 10px; background: #a50; color: #fff; border: 1px solid #830; border-radius: 3px; cursor: pointer; font-size: 11px;";
            clearPromptBtn.onmouseover = () => clearPromptBtn.style.background = "#c60";
            clearPromptBtn.onmouseout = () => clearPromptBtn.style.background = "#a50";
            buttonContainer.appendChild(clearPromptBtn);

            // Clear All button
            const clearAllBtn = document.createElement("button");
            clearAllBtn.textContent = "Clear All";
            clearAllBtn.style.cssText = "padding: 5px 10px; background: #d44; color: #fff; border: 1px solid #a22; border-radius: 3px; cursor: pointer; font-size: 11px;";
            clearAllBtn.onmouseover = () => clearAllBtn.style.background = "#e55";
            clearAllBtn.onmouseout = () => clearAllBtn.style.background = "#d44";
            buttonContainer.appendChild(clearAllBtn);

            // Canvas wrapper
            const canvasWrapper = document.createElement("div");
            canvasWrapper.style.cssText = "flex: 1; display: flex; align-items: center; justify-content: center; min-height: 200px;";
            container.appendChild(canvasWrapper);

            // Canvas
            const canvas = document.createElement("canvas");
            canvas.width = 512;
            canvas.height = 512;
            canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: crosshair;";
            canvasWrapper.appendChild(canvas);

            const ctx = canvas.getContext("2d");

            // Tab bar (bottom)
            const tabBar = document.createElement("div");
            tabBar.style.cssText = "display: flex; flex-wrap: wrap; gap: 4px; padding: 6px; background: #1a1a1a; border-top: 1px solid #333;";
            container.appendChild(tabBar);

            // State
            this.canvasWidget = {
                canvas, ctx, container, canvasWrapper,
                image: null,
                overlayImage: null,   // mask overlay from Run
                overlayStale: false,  // true when prompts changed since last Run
                isRunning: false,     // true during API call
                prompts: [{
                    positive_points: [],
                    negative_points: [],
                    positive_boxes: [],
                    negative_boxes: []
                }],
                activePromptIndex: 0,
                currentBox: null,
                isDrawingBox: false,
                hoveredItem: null,
                tabBar, counter, runBtn
            };

            // Add DOM widget
            const widget = this.addDOMWidget("canvas", "customCanvas", container);
            this.canvasWidget.domWidget = widget;

            widget.computeSize = (width) => {
                const nodeHeight = this.size ? this.size[1] : 520;
                return [width, Math.max(250, nodeHeight - 80)];
            };

            this.rebuildTabBar();

            // ---- Run button handler ----
            runBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.runInteractiveSegment();
            });

            clearPromptBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.clearActivePrompt();
            });

            clearAllBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.clearAllPrompts();
            });

            // Hide storage widget
            const storeWidget = this.widgets.find(w => w.name === "multi_prompts_store");
            if (storeWidget) {
                storeWidget.value = storeWidget.value || "[]";
                this._hiddenWidgets = { multi_prompts_store: storeWidget };
                hideWidgetForGood(this, storeWidget);
            }

            // Override draw foreground
            const originalDrawForeground = this.onDrawForeground;
            this.onDrawForeground = function(ctx) {
                const hiddenWidgets = this.widgets.filter(w => w.type?.includes("converted-widget"));
                const originalTypes = hiddenWidgets.map(w => w.type);
                hiddenWidgets.forEach(w => w.type = null);
                if (originalDrawForeground) originalDrawForeground.apply(this, arguments);
                hiddenWidgets.forEach((w, i) => w.type = originalTypes[i]);

                const containerHeight = Math.max(250, this.size[1] - 80);
                if (container.style.height !== containerHeight + "px") {
                    container.style.height = containerHeight + "px";
                }
            };

            // ---- Mouse events (same as multiregion) ----
            canvas.addEventListener("mousedown", (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                const activePrompt = this.canvasWidget.prompts[this.canvasWidget.activePromptIndex];
                const isNegative = e.button === 2;

                if (e.shiftKey) {
                    this.canvasWidget.currentBox = { x1: x, y1: y, x2: x, y2: y, isNegative };
                    this.canvasWidget.isDrawingBox = true;
                    return;
                }

                const pointList = isNegative ? activePrompt.negative_points : activePrompt.positive_points;
                pointList.push({ x, y });
                this.updateStorage();
                this.redrawCanvas();
            });

            canvas.addEventListener("mousemove", (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

                if (this.canvasWidget.isDrawingBox && this.canvasWidget.currentBox) {
                    this.canvasWidget.currentBox.x2 = x;
                    this.canvasWidget.currentBox.y2 = y;
                    this.redrawCanvas();
                } else {
                    const hovered = this.findItemAt(x, y);
                    if (hovered !== this.canvasWidget.hoveredItem) {
                        this.canvasWidget.hoveredItem = hovered;
                        this.redrawCanvas();
                    }
                }
            });

            canvas.addEventListener("mouseup", (e) => {
                if (this.canvasWidget.isDrawingBox && this.canvasWidget.currentBox) {
                    const box = this.canvasWidget.currentBox;
                    const width = Math.abs(box.x2 - box.x1);
                    const height = Math.abs(box.y2 - box.y1);

                    if (width > 5 && height > 5) {
                        const normalizedBox = {
                            x1: Math.min(box.x1, box.x2),
                            y1: Math.min(box.y1, box.y2),
                            x2: Math.max(box.x1, box.x2),
                            y2: Math.max(box.y1, box.y2)
                        };
                        const activePrompt = this.canvasWidget.prompts[this.canvasWidget.activePromptIndex];
                        const boxList = box.isNegative ? activePrompt.negative_boxes : activePrompt.positive_boxes;
                        boxList.push(normalizedBox);
                        this.updateStorage();
                    }

                    this.canvasWidget.currentBox = null;
                    this.canvasWidget.isDrawingBox = false;
                    this.redrawCanvas();
                }
            });

            canvas.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                canvas.dispatchEvent(new MouseEvent('mousedown', {
                    button: 2,
                    clientX: e.clientX,
                    clientY: e.clientY,
                    shiftKey: e.shiftKey
                }));
            });

            // ---- onExecuted — receive images from workflow execution ----
            this.onExecuted = (message) => {
                if (message.bg_image && message.bg_image[0]) {
                    const img = new Image();
                    img.onload = () => {
                        this.canvasWidget.image = img;
                        canvas.width = img.width;
                        canvas.height = img.height;
                        this.redrawCanvas();
                    };
                    img.src = "data:image/jpeg;base64," + message.bg_image[0];
                }
                if (message.overlay_image && message.overlay_image[0]) {
                    const oimg = new Image();
                    oimg.onload = () => {
                        this.canvasWidget.overlayImage = oimg;
                        this.canvasWidget.overlayStale = false;
                        this.redrawCanvas();
                    };
                    oimg.src = "data:image/jpeg;base64," + message.overlay_image[0];
                }
            };

            // Handle resize
            const originalOnResize = this.onResize;
            this.onResize = function(size) {
                if (originalOnResize) originalOnResize.apply(this, arguments);
                container.style.height = Math.max(250, size[1] - 80) + "px";
            };

            this.redrawCanvas();
            this.setSize([400, 520]);
            container.style.height = "440px";

            return result;
        };

        // ---- Restore on configure ----
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            onConfigure?.apply(this, arguments);
            const storeWidget = this._hiddenWidgets?.multi_prompts_store;
            if (storeWidget && storeWidget.value) {
                try {
                    const stored = JSON.parse(storeWidget.value);
                    if (Array.isArray(stored) && stored.length > 0) {
                        this.canvasWidget.prompts = stored;
                        this.canvasWidget.activePromptIndex = 0;
                        this.rebuildTabBar();
                        this.redrawCanvas();
                    }
                } catch (e) {
                    console.log("[SAM3] Failed to restore prompts:", e);
                }
            }
        };

        // ---- Run interactive segment via API ----
        nodeType.prototype.runInteractiveSegment = async function() {
            const cw = this.canvasWidget;
            if (cw.isRunning) return;
            if (!cw.image) {
                console.warn("[SAM3] No image loaded. Queue the workflow first.");
                return;
            }

            // Check if any prompts have content
            const hasContent = cw.prompts.some(p =>
                p.positive_points.length > 0 || p.negative_points.length > 0 ||
                p.positive_boxes.length > 0 || p.negative_boxes.length > 0
            );
            if (!hasContent) return;

            // Auto-open a new prompt tab immediately on Run click
            this.addNewPrompt();

            // Set loading state
            cw.isRunning = true;
            cw.runBtn.textContent = "...";
            cw.runBtn.style.background = "#555";
            cw.runBtn.disabled = true;

            try {
                const resp = await fetch("/sam3/interactive_segment", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        node_id: String(this.id),
                        prompts: cw.prompts
                    })
                });
                const data = await resp.json();

                if (!resp.ok || data.error) {
                    console.error("[SAM3] Interactive segment error:", data.error);
                    // Brief flash red
                    cw.runBtn.style.background = "#a22";
                    setTimeout(() => { cw.runBtn.style.background = "#2a7a2a"; }, 800);
                    return;
                }

                if (data.overlay) {
                    const oimg = new Image();
                    oimg.onload = () => {
                        cw.overlayImage = oimg;
                        cw.overlayStale = false;
                        this.redrawCanvas();
                    };
                    oimg.src = "data:image/jpeg;base64," + data.overlay;
                }
            } catch (err) {
                console.error("[SAM3] Interactive segment fetch failed:", err);
                cw.runBtn.style.background = "#a22";
                setTimeout(() => { cw.runBtn.style.background = "#2a7a2a"; }, 800);
            } finally {
                cw.isRunning = false;
                cw.runBtn.textContent = "Run";
                cw.runBtn.style.background = "#2a7a2a";
                cw.runBtn.disabled = false;
            }
        };

        // ---- Tab bar ----
        nodeType.prototype.rebuildTabBar = function() {
            const tabBar = this.canvasWidget.tabBar;
            tabBar.innerHTML = "";

            this.canvasWidget.prompts.forEach((prompt, idx) => {
                const tab = document.createElement("div");
                const color = PROMPT_COLORS[idx % PROMPT_COLORS.length];
                const isActive = idx === this.canvasWidget.activePromptIndex;

                tab.style.cssText = `
                    display: flex; align-items: center; gap: 6px;
                    padding: 4px 8px; background: ${isActive ? '#333' : '#2a2a2a'};
                    border: 1px solid ${isActive ? color.primary : '#444'};
                    border-radius: 4px; cursor: pointer; font-size: 11px;
                    color: ${isActive ? '#fff' : '#aaa'};
                `;

                const colorDot = document.createElement("span");
                colorDot.style.cssText = `width: 10px; height: 10px; border-radius: 2px; background: ${color.primary};`;
                tab.appendChild(colorDot);

                const label = document.createElement("span");
                label.textContent = `Prompt ${idx + 1}`;
                tab.appendChild(label);

                if (this.canvasWidget.prompts.length > 1) {
                    const deleteBtn = document.createElement("span");
                    deleteBtn.textContent = "×";
                    deleteBtn.style.cssText = "color: #888; cursor: pointer; font-size: 14px; padding: 0 2px; margin-left: 2px;";
                    deleteBtn.onmouseover = () => deleteBtn.style.color = "#f00";
                    deleteBtn.onmouseout = () => deleteBtn.style.color = "#888";
                    deleteBtn.onclick = (e) => {
                        e.stopPropagation();
                        this.deletePrompt(idx);
                    };
                    tab.appendChild(deleteBtn);
                }

                tab.onclick = () => this.setActivePrompt(idx);
                tab.onmouseover = () => { if (!isActive) tab.style.background = '#3a3a3a'; };
                tab.onmouseout = () => { if (!isActive) tab.style.background = '#2a2a2a'; };
                tabBar.appendChild(tab);
            });

            if (this.canvasWidget.prompts.length < MAX_PROMPTS) {
                const addBtn = document.createElement("button");
                addBtn.textContent = "+";
                addBtn.style.cssText = `
                    padding: 4px 12px; background: #2a5a2a; border: 1px solid #3a7a3a;
                    border-radius: 4px; color: #8f8; cursor: pointer; font-size: 14px; font-weight: bold;
                `;
                addBtn.onmouseover = () => addBtn.style.background = "#3a6a3a";
                addBtn.onmouseout = () => addBtn.style.background = "#2a5a2a";
                addBtn.onclick = () => this.addNewPrompt();
                tabBar.appendChild(addBtn);
            }

            this.updateCounter();
        };

        nodeType.prototype.setActivePrompt = function(index) {
            this.canvasWidget.activePromptIndex = index;
            this.rebuildTabBar();
            this.redrawCanvas();
        };

        nodeType.prototype.addNewPrompt = function() {
            if (this.canvasWidget.prompts.length >= MAX_PROMPTS) return;
            this.canvasWidget.prompts.push({
                positive_points: [],
                negative_points: [],
                positive_boxes: [],
                negative_boxes: []
            });
            this.canvasWidget.activePromptIndex = this.canvasWidget.prompts.length - 1;
            this.rebuildTabBar();
            this.updateStorage();
            this.redrawCanvas();
        };

        nodeType.prototype.deletePrompt = function(index) {
            if (this.canvasWidget.prompts.length <= 1) {
                this.clearActivePrompt();
                return;
            }
            this.canvasWidget.prompts.splice(index, 1);
            if (this.canvasWidget.activePromptIndex >= this.canvasWidget.prompts.length) {
                this.canvasWidget.activePromptIndex = this.canvasWidget.prompts.length - 1;
            }
            this.rebuildTabBar();
            this.updateStorage();
            this.redrawCanvas();
        };

        nodeType.prototype.clearActivePrompt = function() {
            const prompt = this.canvasWidget.prompts[this.canvasWidget.activePromptIndex];
            prompt.positive_points = [];
            prompt.negative_points = [];
            prompt.positive_boxes = [];
            prompt.negative_boxes = [];
            this.updateStorage();
            this.redrawCanvas();
        };

        nodeType.prototype.clearAllPrompts = function() {
            this.canvasWidget.prompts = [{
                positive_points: [],
                negative_points: [],
                positive_boxes: [],
                negative_boxes: []
            }];
            this.canvasWidget.activePromptIndex = 0;
            this.canvasWidget.overlayImage = null;
            this.canvasWidget.overlayStale = false;
            this.rebuildTabBar();
            this.updateStorage();
            this.redrawCanvas();
        };

        nodeType.prototype.findItemAt = function(x, y) {
            const threshold = 10;
            const pIdx = this.canvasWidget.activePromptIndex;
            const prompt = this.canvasWidget.prompts[pIdx];

            for (let i = 0; i < prompt.positive_points.length; i++) {
                const pt = prompt.positive_points[i];
                if (Math.abs(pt.x - x) < threshold && Math.abs(pt.y - y) < threshold)
                    return { type: "point", index: i, promptIndex: pIdx, isNegative: false };
            }
            for (let i = 0; i < prompt.negative_points.length; i++) {
                const pt = prompt.negative_points[i];
                if (Math.abs(pt.x - x) < threshold && Math.abs(pt.y - y) < threshold)
                    return { type: "point", index: i, promptIndex: pIdx, isNegative: true };
            }
            for (let i = 0; i < prompt.positive_boxes.length; i++) {
                const box = prompt.positive_boxes[i];
                if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2)
                    return { type: "box", index: i, promptIndex: pIdx, isNegative: false };
            }
            for (let i = 0; i < prompt.negative_boxes.length; i++) {
                const box = prompt.negative_boxes[i];
                if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2)
                    return { type: "box", index: i, promptIndex: pIdx, isNegative: true };
            }
            return null;
        };

        nodeType.prototype.updateStorage = function() {
            const widget = this._hiddenWidgets?.multi_prompts_store;
            if (widget) {
                widget.value = JSON.stringify(this.canvasWidget.prompts);
            }
            // Mark overlay as stale since prompts changed
            this.canvasWidget.overlayStale = true;
            this.updateCounter();
        };

        nodeType.prototype.updateCounter = function() {
            const prompt = this.canvasWidget.prompts[this.canvasWidget.activePromptIndex];
            const pts = prompt.positive_points.length + prompt.negative_points.length;
            const boxes = prompt.positive_boxes.length + prompt.negative_boxes.length;
            this.canvasWidget.counter.textContent = `Prompt ${this.canvasWidget.activePromptIndex + 1}: ${pts} pts, ${boxes} boxes`;
        };

        // ---- Canvas rendering ----
        nodeType.prototype.redrawCanvas = function() {
            const { canvas, ctx, image, overlayImage, overlayStale,
                    prompts, activePromptIndex, currentBox, hoveredItem } = this.canvasWidget;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw background: overlay (if exists) or original image
            if (overlayImage) {
                if (overlayStale) {
                    ctx.globalAlpha = 0.45;
                    ctx.drawImage(overlayImage, 0, 0, canvas.width, canvas.height);
                    ctx.globalAlpha = 1.0;
                    // Draw "stale" indicator
                    ctx.fillStyle = "rgba(0,0,0,0.6)";
                    ctx.fillRect(canvas.width / 2 - 55, canvas.height - 50, 110, 24);
                    ctx.fillStyle = "#ff8";
                    ctx.font = "12px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText("prompts changed", canvas.width / 2, canvas.height - 33);
                } else {
                    ctx.drawImage(overlayImage, 0, 0, canvas.width, canvas.height);
                }
            } else if (image) {
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#666";
                ctx.font = "14px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Queue workflow to load image (Ctrl+Enter)", canvas.width / 2, canvas.height / 2 - 20);
                ctx.fillText("Click: Positive point | Right-click: Negative point", canvas.width / 2, canvas.height / 2 + 5);
                ctx.fillText("Shift+Drag: Box | Then click Run for live masks", canvas.width / 2, canvas.height / 2 + 30);
            }

            // Draw active prompt annotations
            const prompt = prompts[activePromptIndex];
            const color = PROMPT_COLORS[activePromptIndex % PROMPT_COLORS.length];

            this.drawBoxes(ctx, prompt.positive_boxes, color.primary, 1.0, false, activePromptIndex, hoveredItem);
            this.drawBoxes(ctx, prompt.negative_boxes, color.primary, 1.0, true, activePromptIndex, hoveredItem);
            this.drawPoints(ctx, prompt.positive_points, color.primary, 1.0, false, activePromptIndex, hoveredItem);
            this.drawPoints(ctx, prompt.negative_points, color.primary, 1.0, true, activePromptIndex, hoveredItem);

            // Current box being drawn
            if (currentBox) {
                const c = PROMPT_COLORS[activePromptIndex % PROMPT_COLORS.length];
                ctx.setLineDash([5, 5]);
                ctx.strokeStyle = currentBox.isNegative ? "#f80" : c.primary;
                ctx.lineWidth = 2;
                const w = currentBox.x2 - currentBox.x1;
                const h = currentBox.y2 - currentBox.y1;
                ctx.strokeRect(currentBox.x1, currentBox.y1, w, h);
                ctx.setLineDash([]);
                ctx.fillStyle = currentBox.isNegative ? "rgba(255,128,0,0.1)" : this.colorWithAlpha(c.primary, 0.1);
                ctx.fillRect(currentBox.x1, currentBox.y1, w, h);
            }

            // Image dimensions
            if (image || overlayImage) {
                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(5, canvas.height - 25, 150, 20);
                ctx.fillStyle = "#0f0";
                ctx.font = "12px monospace";
                ctx.textAlign = "left";
                ctx.fillText(`Image: ${canvas.width}x${canvas.height}`, 10, canvas.height - 10);
            }
        };

        nodeType.prototype.drawPoints = function(ctx, points, color, alpha, isNegative, promptIndex, hoveredItem) {
            const canvas = this.canvasWidget.canvas;
            const scaleFactor = Math.max(0.5, canvas.height / 1080);
            const baseRadius = 6 * scaleFactor;
            const hoverRadius = 8 * scaleFactor;

            for (let i = 0; i < points.length; i++) {
                const pt = points[i];
                const isHovered = hoveredItem?.type === "point" &&
                                  hoveredItem?.promptIndex === promptIndex &&
                                  hoveredItem?.index === i &&
                                  hoveredItem?.isNegative === isNegative;
                const radius = isHovered ? hoverRadius : baseRadius;

                ctx.beginPath();
                ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);

                if (isNegative) {
                    ctx.fillStyle = `rgba(255, 0, 0, ${alpha * 0.8})`;
                } else {
                    ctx.fillStyle = this.colorWithAlpha(color, alpha * 0.8);
                }
                ctx.fill();

                ctx.strokeStyle = isHovered ? "#fff" : this.colorWithAlpha(color, alpha);
                ctx.lineWidth = (isHovered ? 3 : 2) * scaleFactor;
                ctx.stroke();

                if (isNegative) {
                    const xSize = 3 * scaleFactor;
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 2 * scaleFactor;
                    ctx.beginPath();
                    ctx.moveTo(pt.x - xSize, pt.y - xSize);
                    ctx.lineTo(pt.x + xSize, pt.y + xSize);
                    ctx.moveTo(pt.x + xSize, pt.y - xSize);
                    ctx.lineTo(pt.x - xSize, pt.y + xSize);
                    ctx.stroke();
                }
            }
        };

        nodeType.prototype.drawBoxes = function(ctx, boxes, color, alpha, isNegative, promptIndex, hoveredItem) {
            for (let i = 0; i < boxes.length; i++) {
                const box = boxes[i];
                const w = box.x2 - box.x1;
                const h = box.y2 - box.y1;
                const isHovered = hoveredItem?.type === "box" &&
                                  hoveredItem?.promptIndex === promptIndex &&
                                  hoveredItem?.index === i &&
                                  hoveredItem?.isNegative === isNegative;

                if (isNegative) {
                    ctx.fillStyle = `rgba(255, 0, 0, ${alpha * 0.15})`;
                } else {
                    ctx.fillStyle = this.colorWithAlpha(color, alpha * 0.15);
                }
                ctx.fillRect(box.x1, box.y1, w, h);

                ctx.strokeStyle = isHovered ? "#fff" : (isNegative ? `rgba(255,0,0,${alpha})` : this.colorWithAlpha(color, alpha));
                ctx.lineWidth = isHovered ? 3 : 2;
                if (isNegative) ctx.setLineDash([4, 4]);
                ctx.strokeRect(box.x1, box.y1, w, h);
                ctx.setLineDash([]);
            }
        };

        nodeType.prototype.colorWithAlpha = function(hexColor, alpha) {
            const r = parseInt(hexColor.slice(1, 3), 16);
            const g = parseInt(hexColor.slice(3, 5), 16);
            const b = parseInt(hexColor.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        };
    }
});
