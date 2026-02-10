/**
 * SAM3 Simple BBox Collector
 * Uses plain HTML5 Canvas for drawing bounding boxes
 * Version: 2025-01-20-v1-BBOX
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

console.log("[SAM3] ===== BBOX COLLECTOR VERSION 1 =====");

// Helper function to properly hide widgets (enhanced for complete hiding)
function hideWidgetForGood(node, widget, suffix = '') {
    if (!widget) return;

    // Save original properties
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;

    // Multiple hiding approaches to ensure widget is fully hidden
    widget.computeSize = () => [0, -4];  // -4 compensates for litegraph's automatic widget gap
    widget.type = "converted-widget" + suffix;
    widget.hidden = true;  // Mark as hidden

    // IMPORTANT: Keep serialization enabled so values are sent to backend
    // (We just hide it visually, but it still needs to send data)

    // Make the widget completely invisible in the DOM if it has element
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }

    // Handle linked widgets recursively
    if (widget.linkedWidgets) {
        for (const w of widget.linkedWidgets) {
            hideWidgetForGood(node, w, ':' + widget.name);
        }
    }
}

app.registerExtension({
    name: "Comfy.SAM3.SimpleBBoxCollector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[SAM3] beforeRegisterNodeDef called for:", nodeData.name);

        if (nodeData.name === "SAM3BBoxCollector") {
            console.log("[SAM3] Registering SAM3BBoxCollector node");
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                console.log("[SAM3] onNodeCreated called for SAM3BBoxCollector");

                // Call original onNodeCreated FIRST to create all widgets
                const result = onNodeCreated?.apply(this, arguments);

                console.log("[SAM3] Widgets after creation:", this.widgets?.map(w => w.name));

                console.log("[SAM3] Creating canvas container");
                // Create canvas container - dynamically sized based on node height
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #222; overflow: hidden; box-sizing: border-box; margin: 0; padding: 0; display: flex; align-items: center; justify-content: center;";

                // Create info/button bar
                const infoBar = document.createElement("div");
                infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between; align-items: center;";
                container.appendChild(infoBar);

                // Create bbox counter
                const bboxCounter = document.createElement("div");
                bboxCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
                bboxCounter.textContent = "Bboxes: 0 pos, 0 neg";
                infoBar.appendChild(bboxCounter);

                // Create clear button
                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear All";
                clearButton.style.cssText = "padding: 5px 10px; background: #d44; color: #fff; border: 1px solid #a22; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: bold;";
                clearButton.onmouseover = () => clearButton.style.background = "#e55";
                clearButton.onmouseout = () => clearButton.style.background = "#d44";
                infoBar.appendChild(clearButton);

                // Create canvas for image and bboxes
                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                // Use max-width and max-height instead of width/height 100% to prevent overflow
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: crosshair; margin: 0 auto;";
                container.appendChild(canvas);

                const ctx = canvas.getContext("2d");
                console.log("[SAM3] Canvas created:", canvas);

                // Store state
                this.canvasWidget = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    image: null,
                    positiveBBoxes: [],
                    negativeBBoxes: [],
                    currentBBox: null,  // BBox being drawn {bbox, isNegative}
                    hoveredBBox: null,
                    bboxCounter: bboxCounter
                };

                // Add as DOM widget
                console.log("[SAM3] Adding DOM widget via addDOMWidget");
                const widget = this.addDOMWidget("canvas", "customCanvas", container);
                console.log("[SAM3] addDOMWidget returned:", widget);

                // Store widget reference for updates
                this.canvasWidget.domWidget = widget;

                // Make widget dynamically sized - override computeSize
                widget.computeSize = (width) => {
                    // Widget height = node height - title bar/padding (approx 80px)
                    const nodeHeight = this.size ? this.size[1] : 480;
                    const widgetHeight = Math.max(200, nodeHeight - 80);
                    return [width, widgetHeight];
                };

                // Clear button handler
                clearButton.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log("[SAM3] Clearing all bboxes");
                    this.canvasWidget.positiveBBoxes = [];
                    this.canvasWidget.negativeBBoxes = [];
                    this.updateBBoxes();
                    this.redrawCanvas();
                });

                // Hide the string storage widgets
                console.log("[SAM3] Attempting to hide widgets...");
                console.log("[SAM3] Widgets before hiding:", this.widgets.map(w => w.name));

                const bboxesWidget = this.widgets.find(w => w.name === "bboxes");
                const negBboxesWidget = this.widgets.find(w => w.name === "neg_bboxes");

                console.log("[SAM3] Found widgets to hide:", { bboxesWidget, negBboxesWidget });

                // Initialize default values BEFORE hiding
                if (bboxesWidget) {
                    bboxesWidget.value = bboxesWidget.value || "[]";
                }
                if (negBboxesWidget) {
                    negBboxesWidget.value = negBboxesWidget.value || "[]";
                }

                // Store references before hiding
                this._hiddenWidgets = {
                    bboxes: bboxesWidget,
                    neg_bboxes: negBboxesWidget
                };

                // Apply hiding
                if (bboxesWidget) {
                    hideWidgetForGood(this, bboxesWidget);
                    console.log("[SAM3] bboxes - type:", bboxesWidget.type, "hidden:", bboxesWidget.hidden, "value:", bboxesWidget.value);
                }
                if (negBboxesWidget) {
                    hideWidgetForGood(this, negBboxesWidget);
                    console.log("[SAM3] neg_bboxes - type:", negBboxesWidget.type, "hidden:", negBboxesWidget.hidden, "value:", negBboxesWidget.value);
                }

                // CRITICAL FIX: Override onDrawForeground to skip rendering hidden widgets
                const originalDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    // Temporarily hide converted widgets from rendering
                    const hiddenWidgets = this.widgets.filter(w => w.type?.includes("converted-widget"));
                    const originalTypes = hiddenWidgets.map(w => w.type);

                    // Temporarily set to null to prevent rendering
                    hiddenWidgets.forEach(w => w.type = null);

                    // Call original draw
                    if (originalDrawForeground) {
                        originalDrawForeground.apply(this, arguments);
                    }

                    // Restore types
                    hiddenWidgets.forEach((w, i) => w.type = originalTypes[i]);

                    // Update container height based on current node size
                    const containerHeight = Math.max(200, this.size[1] - 80);
                    if (container.style.height !== containerHeight + "px") {
                        container.style.height = containerHeight + "px";
                    }
                };

                console.log("[SAM3] Widgets after hiding:", this.widgets.map(w => `${w.name}(${w.type})`));
                console.log("[SAM3] All widgets processing complete");

                // Mouse event handlers for bbox drawing
                canvas.addEventListener("mousedown", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                    console.log(`[SAM3] MouseDown at canvas coords: (${x.toFixed(1)}, ${y.toFixed(1)})`);

                    // Check if right-click on existing bbox to delete
                    const clickedBBox = this.findBBoxAt(x, y);
                    if (clickedBBox && e.button === 2) {
                        // Right-click on existing bbox = delete
                        if (clickedBBox.type === 'positive') {
                            this.canvasWidget.positiveBBoxes = this.canvasWidget.positiveBBoxes.filter(b => b !== clickedBBox.bbox);
                        } else {
                            this.canvasWidget.negativeBBoxes = this.canvasWidget.negativeBBoxes.filter(b => b !== clickedBBox.bbox);
                        }
                        this.updateBBoxes();
                        this.redrawCanvas();
                        return;
                    }

                    // Start drawing new bbox
                    // Determine if it's negative (shift+click or right-click and drag)
                    const isNegative = e.shiftKey || e.button === 2;
                    this.canvasWidget.currentBBox = {
                        bbox: {
                            x1: x,
                            y1: y,
                            x2: x,
                            y2: y
                        },
                        isNegative: isNegative
                    };
                });

                canvas.addEventListener("mousemove", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

                    // Update current bbox if drawing
                    if (this.canvasWidget.currentBBox) {
                        this.canvasWidget.currentBBox.bbox.x2 = x;
                        this.canvasWidget.currentBBox.bbox.y2 = y;
                        this.redrawCanvas();
                    } else {
                        // Check for hover
                        const hovered = this.findBBoxAt(x, y);
                        if (hovered !== this.canvasWidget.hoveredBBox) {
                            this.canvasWidget.hoveredBBox = hovered;
                            this.redrawCanvas();
                        }
                    }
                });

                canvas.addEventListener("mouseup", (e) => {
                    if (this.canvasWidget.currentBBox) {
                        const rect = canvas.getBoundingClientRect();
                        const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                        const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

                        this.canvasWidget.currentBBox.bbox.x2 = x;
                        this.canvasWidget.currentBBox.bbox.y2 = y;

                        // Only add bbox if it has some size
                        const width = Math.abs(this.canvasWidget.currentBBox.bbox.x2 - this.canvasWidget.currentBBox.bbox.x1);
                        const height = Math.abs(this.canvasWidget.currentBBox.bbox.y2 - this.canvasWidget.currentBBox.bbox.y1);

                        if (width > 5 && height > 5) {
                            // Normalize so x1,y1 is top-left and x2,y2 is bottom-right
                            const bbox = {
                                x1: Math.min(this.canvasWidget.currentBBox.bbox.x1, this.canvasWidget.currentBBox.bbox.x2),
                                y1: Math.min(this.canvasWidget.currentBBox.bbox.y1, this.canvasWidget.currentBBox.bbox.y2),
                                x2: Math.max(this.canvasWidget.currentBBox.bbox.x1, this.canvasWidget.currentBBox.bbox.x2),
                                y2: Math.max(this.canvasWidget.currentBBox.bbox.y1, this.canvasWidget.currentBBox.bbox.y2)
                            };

                            // Add to appropriate list
                            if (this.canvasWidget.currentBBox.isNegative) {
                                this.canvasWidget.negativeBBoxes.push(bbox);
                                console.log(`[SAM3] Added negative bbox: (${bbox.x1.toFixed(1)}, ${bbox.y1.toFixed(1)}) to (${bbox.x2.toFixed(1)}, ${bbox.y2.toFixed(1)})`);
                            } else {
                                this.canvasWidget.positiveBBoxes.push(bbox);
                                console.log(`[SAM3] Added positive bbox: (${bbox.x1.toFixed(1)}, ${bbox.y1.toFixed(1)}) to (${bbox.x2.toFixed(1)}, ${bbox.y2.toFixed(1)})`);
                            }
                            this.updateBBoxes();
                        }

                        this.canvasWidget.currentBBox = null;
                        this.redrawCanvas();
                    }
                });

                canvas.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                    // Trigger mousedown with right button flag
                    canvas.dispatchEvent(new MouseEvent('mousedown', {
                        button: 2,
                        clientX: e.clientX,
                        clientY: e.clientY
                    }));
                });

                // Handle image input changes
                this.onExecuted = (message) => {
                    console.log("[SAM3] onExecuted called with message:", message);
                    if (message.bg_image && message.bg_image[0]) {
                        const img = new Image();
                        img.onload = () => {
                            console.log(`[SAM3] Image loaded: ${img.width}x${img.height}`);
                            this.canvasWidget.image = img;
                            canvas.width = img.width;
                            canvas.height = img.height;
                            console.log(`[SAM3] Canvas resized to: ${canvas.width}x${canvas.height}`);
                            this.redrawCanvas();
                        };
                        img.src = "data:image/jpeg;base64," + message.bg_image[0];
                    }
                };

                // Update container height dynamically when node size changes
                const originalOnResize = this.onResize;
                this.onResize = function(size) {
                    if (originalOnResize) {
                        originalOnResize.apply(this, arguments);
                    }
                    // Update container to match widget size
                    const containerHeight = Math.max(200, size[1] - 80);
                    container.style.height = containerHeight + "px";
                    console.log(`[SAM3] Node resized to: ${size[0]}x${size[1]}, container height: ${containerHeight}px`);
                };

                // Also update on draw to handle any size changes
                const originalOnDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    if (originalOnDrawForeground) {
                        originalOnDrawForeground.apply(this, arguments);
                    }
                    // Update container height based on current node size
                    const containerHeight = Math.max(200, this.size[1] - 80);
                    if (container.style.height !== containerHeight + "px") {
                        container.style.height = containerHeight + "px";
                    }
                };

                // Draw initial placeholder
                console.log("[SAM3] Drawing initial placeholder");
                this.redrawCanvas();

                // Set initial node size
                const nodeWidth = Math.max(400, this.size[0] || 400);
                const nodeHeight = 480; // Initial height: canvas (400) + space (80)
                this.setSize([nodeWidth, nodeHeight]);

                // Set initial container height
                container.style.height = "400px";

                console.log("[SAM3] Node size set to:", [nodeWidth, nodeHeight]);
                console.log("[SAM3] onNodeCreated complete");
                return result;
            };

            // Helper: Find bbox at coordinates
            nodeType.prototype.findBBoxAt = function(x, y) {
                // Check positive bboxes
                for (const bbox of this.canvasWidget.positiveBBoxes) {
                    if (x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2) {
                        return {type: 'positive', bbox};
                    }
                }

                // Check negative bboxes
                for (const bbox of this.canvasWidget.negativeBBoxes) {
                    if (x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2) {
                        return {type: 'negative', bbox};
                    }
                }

                return null;
            };

            // Helper: Update widget values
            nodeType.prototype.updateBBoxes = function() {
                // Use stored hidden widget references
                const bboxesWidget = this._hiddenWidgets?.bboxes || this.widgets.find(w => w.name === "bboxes");
                const negBboxesWidget = this._hiddenWidgets?.neg_bboxes || this.widgets.find(w => w.name === "neg_bboxes");

                if (bboxesWidget) {
                    bboxesWidget.value = JSON.stringify(this.canvasWidget.positiveBBoxes);
                }
                if (negBboxesWidget) {
                    negBboxesWidget.value = JSON.stringify(this.canvasWidget.negativeBBoxes);
                }

                // Update bbox counter display
                const posCount = this.canvasWidget.positiveBBoxes.length;
                const negCount = this.canvasWidget.negativeBBoxes.length;
                this.canvasWidget.bboxCounter.textContent = `Bboxes: ${posCount} pos, ${negCount} neg`;
            };

            // Helper: Redraw canvas
            nodeType.prototype.redrawCanvas = function() {
                const {canvas, ctx, image, positiveBBoxes, negativeBBoxes, currentBBox, hoveredBBox} = this.canvasWidget;

                // Clear
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw image if available
                if (image) {
                    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                } else {
                    // Placeholder
                    ctx.fillStyle = "#333";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#666";
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Click and drag to draw bounding boxes", canvas.width / 2, canvas.height / 2);
                    ctx.fillText("Left-click: Positive (cyan)", canvas.width / 2, canvas.height / 2 + 25);
                    ctx.fillText("Shift/Right-click: Negative (red)", canvas.width / 2, canvas.height / 2 + 50);
                }

                // Draw canvas dimensions overlay (helpful for debugging)
                if (image) {
                    ctx.fillStyle = "rgba(0,0,0,0.7)";
                    ctx.fillRect(5, canvas.height - 25, 150, 20);
                    ctx.fillStyle = "#0f0";
                    ctx.font = "12px monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(`Image: ${canvas.width}x${canvas.height}`, 10, canvas.height - 10);
                }

                // Draw positive bboxes (cyan)
                for (const bbox of positiveBBoxes) {
                    const isHovered = hoveredBBox?.bbox === bbox && hoveredBBox?.type === 'positive';
                    if (isHovered) {
                        ctx.strokeStyle = "#ff0";
                        ctx.lineWidth = 3;
                    } else {
                        ctx.strokeStyle = "#0ff";
                        ctx.lineWidth = 2;
                    }

                    const width = bbox.x2 - bbox.x1;
                    const height = bbox.y2 - bbox.y1;
                    ctx.strokeRect(bbox.x1, bbox.y1, width, height);

                    // Draw semi-transparent fill for hovered
                    if (isHovered) {
                        ctx.fillStyle = "rgba(255, 255, 0, 0.1)";
                        ctx.fillRect(bbox.x1, bbox.y1, width, height);
                    }
                }

                // Draw negative bboxes (red)
                for (const bbox of negativeBBoxes) {
                    const isHovered = hoveredBBox?.bbox === bbox && hoveredBBox?.type === 'negative';
                    if (isHovered) {
                        ctx.strokeStyle = "#ff0";
                        ctx.lineWidth = 3;
                    } else {
                        ctx.strokeStyle = "#f00";
                        ctx.lineWidth = 2;
                    }

                    const width = bbox.x2 - bbox.x1;
                    const height = bbox.y2 - bbox.y1;
                    ctx.strokeRect(bbox.x1, bbox.y1, width, height);

                    // Draw semi-transparent fill for hovered
                    if (isHovered) {
                        ctx.fillStyle = "rgba(255, 255, 0, 0.1)";
                        ctx.fillRect(bbox.x1, bbox.y1, width, height);
                    }
                }

                // Draw current bbox being drawn (yellow for positive, orange for negative, with dotted line)
                if (currentBBox) {
                    ctx.strokeStyle = currentBBox.isNegative ? "#f80" : "#ff0";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    const width = currentBBox.bbox.x2 - currentBBox.bbox.x1;
                    const height = currentBBox.bbox.y2 - currentBBox.bbox.y1;
                    ctx.strokeRect(currentBBox.bbox.x1, currentBBox.bbox.y1, width, height);
                    ctx.setLineDash([]);

                    // Draw semi-transparent fill
                    if (currentBBox.isNegative) {
                        ctx.fillStyle = "rgba(255, 128, 0, 0.1)";
                    } else {
                        ctx.fillStyle = "rgba(255, 255, 0, 0.1)";
                    }
                    ctx.fillRect(currentBBox.bbox.x1, currentBBox.bbox.y1, width, height);
                }
            };
        }
    }
});
