/**
 * SAM3 Multi-Region Collector Widget
 * Combines point and box collection with multiple prompt regions
 * Version: 2025-01-20-v1-MULTIREGION
 */

import { app } from "../../scripts/app.js";

console.log("[SAM3] ===== MULTI-REGION COLLECTOR VERSION 1 =====");

// Color palette for different prompt regions (max 8)
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

// Helper function to hide widgets
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
    name: "Comfy.SAM3.MultiRegionCollector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SAM3MultiRegionCollector") return;

        console.log("[SAM3] Registering SAM3MultiRegionCollector node");
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Create main container
            const container = document.createElement("div");
            container.style.cssText = "position: relative; width: 100%; background: #222; overflow: hidden; box-sizing: border-box; display: flex; flex-direction: column;";

            // Create info bar (top)
            const infoBar = document.createElement("div");
            infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between; align-items: center;";
            container.appendChild(infoBar);

            // Counter display
            const counter = document.createElement("div");
            counter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
            counter.textContent = "Prompt 1: 0 pts, 0 boxes";
            infoBar.appendChild(counter);

            // Button container
            const buttonContainer = document.createElement("div");
            buttonContainer.style.cssText = "display: flex; gap: 5px;";
            infoBar.appendChild(buttonContainer);

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

            // Canvas wrapper (for centering)
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

            // Store state
            this.canvasWidget = {
                canvas: canvas,
                ctx: ctx,
                container: container,
                canvasWrapper: canvasWrapper,
                image: null,
                // Multi-prompt state
                prompts: [{
                    positive_points: [],
                    negative_points: [],
                    positive_boxes: [],
                    negative_boxes: []
                }],
                activePromptIndex: 0,
                // Drawing state
                currentBox: null,
                isDrawingBox: false,
                hoveredItem: null,
                // UI elements
                tabBar: tabBar,
                counter: counter
            };

            // Add as DOM widget
            const widget = this.addDOMWidget("canvas", "customCanvas", container);
            this.canvasWidget.domWidget = widget;

            widget.computeSize = (width) => {
                const nodeHeight = this.size ? this.size[1] : 520;
                const widgetHeight = Math.max(250, nodeHeight - 80);
                return [width, widgetHeight];
            };

            // Build initial tab bar
            this.rebuildTabBar();

            // Clear Prompt button handler
            clearPromptBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.clearActivePrompt();
            });

            // Clear All button handler
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

            // Override onDrawForeground to hide converted widgets
            const originalDrawForeground = this.onDrawForeground;
            this.onDrawForeground = function(ctx) {
                const hiddenWidgets = this.widgets.filter(w => w.type?.includes("converted-widget"));
                const originalTypes = hiddenWidgets.map(w => w.type);
                hiddenWidgets.forEach(w => w.type = null);
                if (originalDrawForeground) originalDrawForeground.apply(this, arguments);
                hiddenWidgets.forEach((w, i) => w.type = originalTypes[i]);

                // Update container height
                const containerHeight = Math.max(250, this.size[1] - 80);
                if (container.style.height !== containerHeight + "px") {
                    container.style.height = containerHeight + "px";
                }
            };

            // Mouse event handlers
            canvas.addEventListener("mousedown", (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

                const activePrompt = this.canvasWidget.prompts[this.canvasWidget.activePromptIndex];
                const isNegative = e.button === 2;

                // Shift key = box mode
                if (e.shiftKey) {
                    this.canvasWidget.currentBox = { x1: x, y1: y, x2: x, y2: y, isNegative };
                    this.canvasWidget.isDrawingBox = true;
                    return;
                }

                // Add point (left-click = positive, right-click = negative)
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
                    // Check hover
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

            // Handle image loading
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
            };

            // Handle node resize
            const originalOnResize = this.onResize;
            this.onResize = function(size) {
                if (originalOnResize) originalOnResize.apply(this, arguments);
                const containerHeight = Math.max(250, size[1] - 80);
                container.style.height = containerHeight + "px";
            };

            // Initial draw
            this.redrawCanvas();

            // Set initial size
            this.setSize([400, 520]);
            container.style.height = "440px";

            return result;
        };

        // Restore state when workflow is loaded
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            onConfigure?.apply(this, arguments);

            // Restore prompts from stored widget value
            const storeWidget = this._hiddenWidgets?.multi_prompts_store;
            if (storeWidget && storeWidget.value) {
                try {
                    const stored = JSON.parse(storeWidget.value);
                    if (Array.isArray(stored) && stored.length > 0) {
                        this.canvasWidget.prompts = stored;
                        this.canvasWidget.activePromptIndex = 0;
                        this.rebuildTabBar();
                        this.redrawCanvas();
                        console.log("[SAM3] Restored", stored.length, "prompts from saved state");
                    }
                } catch (e) {
                    console.log("[SAM3] Failed to restore prompts:", e);
                }
            }
        };

        // Rebuild tab bar
        nodeType.prototype.rebuildTabBar = function() {
            const tabBar = this.canvasWidget.tabBar;
            tabBar.innerHTML = "";

            // Create tabs for each prompt
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
                tab.dataset.promptIndex = idx;

                // Color indicator
                const colorDot = document.createElement("span");
                colorDot.style.cssText = `width: 10px; height: 10px; border-radius: 2px; background: ${color.primary};`;
                tab.appendChild(colorDot);

                // Label
                const label = document.createElement("span");
                label.textContent = `Prompt ${idx + 1}`;
                tab.appendChild(label);

                // Delete button (only if more than 1 prompt)
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

                // Tab click handler
                tab.onclick = () => this.setActivePrompt(idx);
                tab.onmouseover = () => { if (!isActive) tab.style.background = '#3a3a3a'; };
                tab.onmouseout = () => { if (!isActive) tab.style.background = '#2a2a2a'; };

                tabBar.appendChild(tab);
            });

            // Add prompt button
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

        // Set active prompt
        nodeType.prototype.setActivePrompt = function(index) {
            this.canvasWidget.activePromptIndex = index;
            this.rebuildTabBar();
            this.redrawCanvas();
        };

        // Add new prompt
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

        // Delete prompt
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

        // Clear active prompt
        nodeType.prototype.clearActivePrompt = function() {
            const prompt = this.canvasWidget.prompts[this.canvasWidget.activePromptIndex];
            prompt.positive_points = [];
            prompt.negative_points = [];
            prompt.positive_boxes = [];
            prompt.negative_boxes = [];
            this.updateStorage();
            this.redrawCanvas();
        };

        // Clear all prompts
        nodeType.prototype.clearAllPrompts = function() {
            this.canvasWidget.prompts = [{
                positive_points: [],
                negative_points: [],
                positive_boxes: [],
                negative_boxes: []
            }];
            this.canvasWidget.activePromptIndex = 0;
            this.rebuildTabBar();
            this.updateStorage();
            this.redrawCanvas();
        };

        // Find item at coordinates (only checks active prompt)
        nodeType.prototype.findItemAt = function(x, y) {
            const threshold = 10;
            const pIdx = this.canvasWidget.activePromptIndex;
            const prompt = this.canvasWidget.prompts[pIdx];

            // Check points
            for (let i = 0; i < prompt.positive_points.length; i++) {
                const pt = prompt.positive_points[i];
                if (Math.abs(pt.x - x) < threshold && Math.abs(pt.y - y) < threshold) {
                    return { type: "point", index: i, promptIndex: pIdx, isNegative: false };
                }
            }
            for (let i = 0; i < prompt.negative_points.length; i++) {
                const pt = prompt.negative_points[i];
                if (Math.abs(pt.x - x) < threshold && Math.abs(pt.y - y) < threshold) {
                    return { type: "point", index: i, promptIndex: pIdx, isNegative: true };
                }
            }

            // Check boxes
            for (let i = 0; i < prompt.positive_boxes.length; i++) {
                const box = prompt.positive_boxes[i];
                if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {
                    return { type: "box", index: i, promptIndex: pIdx, isNegative: false };
                }
            }
            for (let i = 0; i < prompt.negative_boxes.length; i++) {
                const box = prompt.negative_boxes[i];
                if (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2) {
                    return { type: "box", index: i, promptIndex: pIdx, isNegative: true };
                }
            }

            return null;
        };

        // Update storage widget
        nodeType.prototype.updateStorage = function() {
            const widget = this._hiddenWidgets?.multi_prompts_store;
            if (widget) {
                widget.value = JSON.stringify(this.canvasWidget.prompts);
            }
            this.updateCounter();
        };

        // Update counter display
        nodeType.prototype.updateCounter = function() {
            const prompt = this.canvasWidget.prompts[this.canvasWidget.activePromptIndex];
            const pts = prompt.positive_points.length + prompt.negative_points.length;
            const boxes = prompt.positive_boxes.length + prompt.negative_boxes.length;
            this.canvasWidget.counter.textContent = `Prompt ${this.canvasWidget.activePromptIndex + 1}: ${pts} pts, ${boxes} boxes`;
        };

        // Redraw canvas
        nodeType.prototype.redrawCanvas = function() {
            const { canvas, ctx, image, prompts, activePromptIndex, currentBox, hoveredItem } = this.canvasWidget;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw image or placeholder
            if (image) {
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#666";
                ctx.font = "14px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Click: Positive point | Right-click: Negative point", canvas.width / 2, canvas.height / 2 - 10);
                ctx.fillText("Shift+Drag: Box | Shift+Right-drag: Negative box", canvas.width / 2, canvas.height / 2 + 15);
            }

            // Only draw the active prompt
            const prompt = prompts[activePromptIndex];
            const color = PROMPT_COLORS[activePromptIndex % PROMPT_COLORS.length];

            // Draw boxes first
            this.drawBoxes(ctx, prompt.positive_boxes, color.primary, 1.0, false, activePromptIndex, hoveredItem);
            this.drawBoxes(ctx, prompt.negative_boxes, color.primary, 1.0, true, activePromptIndex, hoveredItem);

            // Draw points
            this.drawPoints(ctx, prompt.positive_points, color.primary, 1.0, false, activePromptIndex, hoveredItem);
            this.drawPoints(ctx, prompt.negative_points, color.primary, 1.0, true, activePromptIndex, hoveredItem);

            // Draw current box being drawn
            if (currentBox) {
                const color = PROMPT_COLORS[activePromptIndex % PROMPT_COLORS.length];
                ctx.setLineDash([5, 5]);
                ctx.strokeStyle = currentBox.isNegative ? "#f80" : color.primary;
                ctx.lineWidth = 2;
                const w = currentBox.x2 - currentBox.x1;
                const h = currentBox.y2 - currentBox.y1;
                ctx.strokeRect(currentBox.x1, currentBox.y1, w, h);
                ctx.setLineDash([]);

                ctx.fillStyle = currentBox.isNegative ? "rgba(255,128,0,0.1)" : this.colorWithAlpha(color.primary, 0.1);
                ctx.fillRect(currentBox.x1, currentBox.y1, w, h);
            }

            // Draw image dimensions
            if (image) {
                ctx.fillStyle = "rgba(0,0,0,0.7)";
                ctx.fillRect(5, canvas.height - 25, 150, 20);
                ctx.fillStyle = "#0f0";
                ctx.font = "12px monospace";
                ctx.textAlign = "left";
                ctx.fillText(`Image: ${canvas.width}x${canvas.height}`, 10, canvas.height - 10);
            }
        };

        // Draw points
        nodeType.prototype.drawPoints = function(ctx, points, color, alpha, isNegative, promptIndex, hoveredItem) {
            // Scale point size relative to image height (1080p = baseline)
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

                // Fill
                if (isNegative) {
                    ctx.fillStyle = `rgba(255, 0, 0, ${alpha * 0.8})`;
                } else {
                    ctx.fillStyle = this.colorWithAlpha(color, alpha * 0.8);
                }
                ctx.fill();

                // Stroke
                ctx.strokeStyle = isHovered ? "#fff" : this.colorWithAlpha(color, alpha);
                ctx.lineWidth = (isHovered ? 3 : 2) * scaleFactor;
                ctx.stroke();

                // X marker for negative
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

        // Draw boxes
        nodeType.prototype.drawBoxes = function(ctx, boxes, color, alpha, isNegative, promptIndex, hoveredItem) {
            for (let i = 0; i < boxes.length; i++) {
                const box = boxes[i];
                const w = box.x2 - box.x1;
                const h = box.y2 - box.y1;
                const isHovered = hoveredItem?.type === "box" &&
                                  hoveredItem?.promptIndex === promptIndex &&
                                  hoveredItem?.index === i &&
                                  hoveredItem?.isNegative === isNegative;

                // Fill
                if (isNegative) {
                    ctx.fillStyle = `rgba(255, 0, 0, ${alpha * 0.15})`;
                } else {
                    ctx.fillStyle = this.colorWithAlpha(color, alpha * 0.15);
                }
                ctx.fillRect(box.x1, box.y1, w, h);

                // Stroke
                ctx.strokeStyle = isHovered ? "#fff" : (isNegative ? `rgba(255,0,0,${alpha})` : this.colorWithAlpha(color, alpha));
                ctx.lineWidth = isHovered ? 3 : 2;

                if (isNegative) {
                    ctx.setLineDash([4, 4]);
                }
                ctx.strokeRect(box.x1, box.y1, w, h);
                ctx.setLineDash([]);
            }
        };

        // Helper: color with alpha
        nodeType.prototype.colorWithAlpha = function(hexColor, alpha) {
            const r = parseInt(hexColor.slice(1, 3), 16);
            const g = parseInt(hexColor.slice(3, 5), 16);
            const b = parseInt(hexColor.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        };
    }
});
