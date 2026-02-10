/**
 * SAM3 Simple Point Collector
 * Uses plain HTML5 Canvas instead of Protovis for better compatibility
 * Version: 2025-01-20-v8-SERIALIZE-FIX
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

console.log("[SAM3] ===== VERSION 8 - SERIALIZATION ENABLED =====");

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
    name: "Comfy.SAM3.SimplePointCollector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[SAM3] beforeRegisterNodeDef called for:", nodeData.name);

        if (nodeData.name === "SAM3PointCollector") {
            console.log("[SAM3] Registering SAM3PointCollector node");
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                console.log("[SAM3] onNodeCreated called for SAM3PointCollector");

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

                // Create points counter
                const pointsCounter = document.createElement("div");
                pointsCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
                pointsCounter.textContent = "Points: 0 pos, 0 neg";
                infoBar.appendChild(pointsCounter);

                // Create clear button
                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear All";
                clearButton.style.cssText = "padding: 5px 10px; background: #d44; color: #fff; border: 1px solid #a22; border-radius: 3px; cursor: pointer; font-size: 12px; font-weight: bold;";
                clearButton.onmouseover = () => clearButton.style.background = "#e55";
                clearButton.onmouseout = () => clearButton.style.background = "#d44";
                infoBar.appendChild(clearButton);

                // Create canvas for image and points
                const canvas = document.createElement("canvas");
                canvas.width = 400;
                canvas.height = 300;
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
                    positivePoints: [],
                    negativePoints: [],
                    hoveredPoint: null,
                    pointsCounter: pointsCounter
                };

                // Add as DOM widget
                console.log("[SAM3] Adding DOM widget via addDOMWidget");
                const widget = this.addDOMWidget("canvas", "customCanvas", container);
                console.log("[SAM3] addDOMWidget returned:", widget);

                // Store widget reference for updates
                this.canvasWidget.domWidget = widget;

                // Dynamic widget height - updated when image loads
                this.canvasWidget.widgetHeight = 300; // Default initial height
                widget.computeSize = (width) => {
                    return [width, this.canvasWidget.widgetHeight];
                };

                // Clear button handler
                clearButton.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log("[SAM3] Clearing all points");
                    this.canvasWidget.positivePoints = [];
                    this.canvasWidget.negativePoints = [];
                    this.updatePoints();
                    this.redrawCanvas();
                });

                // Hide the string storage widgets - multiple approaches
                console.log("[SAM3] Attempting to hide widgets...");
                console.log("[SAM3] Widgets before hiding:", this.widgets.map(w => w.name));

                const coordsWidget = this.widgets.find(w => w.name === "coordinates");
                const negCoordsWidget = this.widgets.find(w => w.name === "neg_coordinates");
                const storeWidget = this.widgets.find(w => w.name === "points_store");

                console.log("[SAM3] Found widgets to hide:", { coordsWidget, negCoordsWidget, storeWidget });

                // Initialize default values BEFORE hiding
                if (coordsWidget) {
                    coordsWidget.value = coordsWidget.value || "[]";
                }
                if (negCoordsWidget) {
                    negCoordsWidget.value = negCoordsWidget.value || "[]";
                }
                if (storeWidget) {
                    storeWidget.value = storeWidget.value || "{}";
                }

                // Store references before hiding
                this._hiddenWidgets = {
                    coordinates: coordsWidget,
                    neg_coordinates: negCoordsWidget,
                    points_store: storeWidget
                };

                // Apply hiding
                if (coordsWidget) {
                    hideWidgetForGood(this, coordsWidget);
                    console.log("[SAM3] coordinates - type:", coordsWidget.type, "hidden:", coordsWidget.hidden, "value:", coordsWidget.value);
                }
                if (negCoordsWidget) {
                    hideWidgetForGood(this, negCoordsWidget);
                    console.log("[SAM3] neg_coordinates - type:", negCoordsWidget.type, "hidden:", negCoordsWidget.hidden, "value:", negCoordsWidget.value);
                }
                if (storeWidget) {
                    hideWidgetForGood(this, storeWidget);
                    console.log("[SAM3] points_store - type:", storeWidget.type, "hidden:", storeWidget.hidden, "value:", storeWidget.value);
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
                };

                console.log("[SAM3] Widgets after hiding:", this.widgets.map(w => `${w.name}(${w.type})`));
                console.log("[SAM3] All widgets processing complete");

                // Mouse event handlers
                canvas.addEventListener("click", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    // Calculate coordinates relative to the actual image dimensions
                    // The canvas might be scaled, so we need to map from display coords to image coords
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                    console.log(`[SAM3] Click at canvas coords: (${x.toFixed(1)}, ${y.toFixed(1)}), canvas size: ${canvas.width}x${canvas.height}`);

                    // Check if clicking existing point to delete
                    const clickedPoint = this.findPointAt(x, y);
                    if (clickedPoint && e.button === 2) {
                        // Right-click on existing point = delete
                        if (clickedPoint.type === 'positive') {
                            this.canvasWidget.positivePoints = this.canvasWidget.positivePoints.filter(p => p !== clickedPoint.point);
                        } else {
                            this.canvasWidget.negativePoints = this.canvasWidget.negativePoints.filter(p => p !== clickedPoint.point);
                        }
                    } else {
                        // Add new point
                        if (e.shiftKey || e.button === 2) {
                            // Negative point
                            this.canvasWidget.negativePoints.push({x, y});
                        } else {
                            // Positive point
                            this.canvasWidget.positivePoints.push({x, y});
                        }
                    }

                    this.updatePoints();
                    this.redrawCanvas();
                });

                canvas.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                    // Trigger click with right button flag
                    canvas.dispatchEvent(new MouseEvent('click', {
                        button: 2,
                        clientX: e.clientX,
                        clientY: e.clientY,
                        shiftKey: e.shiftKey
                    }));
                });

                canvas.addEventListener("mousemove", (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

                    const hovered = this.findPointAt(x, y);
                    if (hovered !== this.canvasWidget.hoveredPoint) {
                        this.canvasWidget.hoveredPoint = hovered;
                        this.redrawCanvas();
                    }
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

                            // Calculate widget height based on image aspect ratio
                            const nodeWidth = this.size[0] || 400;
                            const availableWidth = nodeWidth - 20; // Account for padding
                            const aspectRatio = img.height / img.width;
                            const newWidgetHeight = Math.round(availableWidth * aspectRatio);

                            // Update widget height and resize node
                            this._isResizing = true;  // Prevent onResize from fighting
                            this.canvasWidget.widgetHeight = newWidgetHeight;
                            container.style.height = newWidgetHeight + "px";
                            this.setSize([nodeWidth, newWidgetHeight + 80]); // +80 for title/padding
                            setTimeout(() => { this._isResizing = false; }, 50);

                            console.log(`[SAM3] Widget resized to match image: ${newWidgetHeight}px`);
                            this.redrawCanvas();
                        };
                        img.src = "data:image/jpeg;base64," + message.bg_image[0];
                    }
                };

                // Handle manual node resize (user dragging)
                const originalOnResize = this.onResize;
                this.onResize = function(size) {
                    if (originalOnResize) {
                        originalOnResize.apply(this, arguments);
                    }

                    // Prevent feedback loop - only update if not already resizing
                    if (this._isResizing) return;
                    this._isResizing = true;

                    // Calculate new widget height from node size
                    const newWidgetHeight = Math.max(200, size[1] - 80);

                    // Only update if significantly different (prevents micro-adjustments)
                    if (Math.abs(newWidgetHeight - this.canvasWidget.widgetHeight) > 5) {
                        this.canvasWidget.widgetHeight = newWidgetHeight;
                        container.style.height = newWidgetHeight + "px";
                    }

                    // Reset flag after a short delay to allow resize to settle
                    setTimeout(() => { this._isResizing = false; }, 50);
                };

                // Draw initial placeholder
                console.log("[SAM3] Drawing initial placeholder");
                this.redrawCanvas();

                // Set initial node size (smaller default, will resize when image loads)
                const nodeWidth = Math.max(400, this.size[0] || 400);
                const nodeHeight = 380; // Initial height: widget (300) + space (80)
                this.setSize([nodeWidth, nodeHeight]);

                // Set initial container height
                container.style.height = "300px";

                console.log("[SAM3] Node size set to:", [nodeWidth, nodeHeight]);
                console.log("[SAM3] onNodeCreated complete");
                return result;
            };

            // Helper: Find point at coordinates
            nodeType.prototype.findPointAt = function(x, y) {
                const threshold = 10;

                for (const point of this.canvasWidget.positivePoints) {
                    if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) {
                        return {type: 'positive', point};
                    }
                }

                for (const point of this.canvasWidget.negativePoints) {
                    if (Math.abs(point.x - x) < threshold && Math.abs(point.y - y) < threshold) {
                        return {type: 'negative', point};
                    }
                }

                return null;
            };

            // Helper: Update widget values
            nodeType.prototype.updatePoints = function() {
                // Use stored hidden widget references
                const coordsWidget = this._hiddenWidgets?.coordinates || this.widgets.find(w => w.name === "coordinates");
                const negCoordsWidget = this._hiddenWidgets?.neg_coordinates || this.widgets.find(w => w.name === "neg_coordinates");
                const storeWidget = this._hiddenWidgets?.points_store || this.widgets.find(w => w.name === "points_store");

                if (coordsWidget) {
                    coordsWidget.value = JSON.stringify(this.canvasWidget.positivePoints);
                }
                if (negCoordsWidget) {
                    negCoordsWidget.value = JSON.stringify(this.canvasWidget.negativePoints);
                }
                if (storeWidget) {
                    storeWidget.value = JSON.stringify({
                        positive: this.canvasWidget.positivePoints,
                        negative: this.canvasWidget.negativePoints
                    });
                }

                // Update points counter display
                const posCount = this.canvasWidget.positivePoints.length;
                const negCount = this.canvasWidget.negativePoints.length;
                this.canvasWidget.pointsCounter.textContent = `Points: ${posCount} pos, ${negCount} neg`;
            };

            // Helper: Redraw canvas
            nodeType.prototype.redrawCanvas = function() {
                const {canvas, ctx, image, positivePoints, negativePoints, hoveredPoint} = this.canvasWidget;

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
                    ctx.fillText("Click to add points", canvas.width / 2, canvas.height / 2);
                    ctx.fillText("Left-click: Positive (green)", canvas.width / 2, canvas.height / 2 + 25);
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

                // Draw positive points (green)
                ctx.strokeStyle = "#0f0";
                ctx.fillStyle = "#0f0";
                for (const point of positivePoints) {
                    const isHovered = hoveredPoint?.point === point;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, isHovered ? 8 : 6, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                // Draw negative points (red)
                ctx.strokeStyle = "#f00";
                ctx.fillStyle = "#f00";
                for (const point of negativePoints) {
                    const isHovered = hoveredPoint?.point === point;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, isHovered ? 8 : 6, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            };
        }
    }
});
