(() => {
  "use strict";

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  class InteractiveGaussianView {
    constructor(surface, image, status) {
      this.surface = surface;
      this.image = image;
      this.status = status;
      this.endpoint = null;
      this.visible = false;
      this.panMode = false;
      this.camera = { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 };
      this.bindControls();
    }

    setEndpoint(endpoint) {
      if (this.endpoint === endpoint) return;
      this.abort?.abort();
      this.endpoint = endpoint;
      this.clearImage();
      this.reset();
    }

    setVisible(visible) {
      if (visible === this.visible) return;
      this.visible = visible;
      if (visible) this.invalidate();
      else {
        clearTimeout(this.settleTimer);
        this.moving = false;
        this.abort?.abort();
      }
    }

    setPanMode(enabled) {
      this.panMode = enabled;
      this.surface.dataset.pan = String(enabled);
    }

    clearImage() {
      this.image.removeAttribute("src");
      if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
      this.imageUrl = null;
      this.image.hidden = true;
    }

    reset() {
      this.camera = { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 };
      clearTimeout(this.settleTimer);
      this.moving = false;
      this.invalidate();
    }

    move(dx, dy, pan) {
      const camera = this.camera;
      if (pan) {
        const scale = 1.15 / (Math.max(this.surface.clientHeight, 140) * camera.zoom);
        camera.panX = clamp(camera.panX - dx * scale, -1.5, 1.5);
        camera.panY = clamp(camera.panY + dy * scale, -1.5, 1.5);
      } else {
        const rotation = camera.yaw - dx * 0.008;
        camera.yaw = Math.atan2(Math.sin(rotation), Math.cos(rotation));
        camera.pitch = clamp(camera.pitch + dy * 0.006, -1.22, 1.22);
      }
      this.invalidateForMotion();
    }

    zoomBy(factor) {
      this.camera.zoom = clamp(this.camera.zoom * factor, 0.35, 4);
      this.invalidateForMotion();
    }

    bindControls() {
      const surface = this.surface;
      surface.addEventListener("pointerdown", (event) => {
        if ((event.button !== 0 && event.button !== 2) || this.drag) return;
        event.preventDefault();
        surface.focus({ preventScroll: true });
        this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, pan: event.button === 2 };
        surface.setPointerCapture(event.pointerId);
        surface.dataset.dragging = "true";
      });
      surface.addEventListener("pointermove", (event) => {
        if (event.pointerId !== this.drag?.id) return;
        this.move(event.clientX - this.drag.x, event.clientY - this.drag.y, this.panMode || this.drag.pan || event.shiftKey);
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
      });
      const finishDrag = (event) => {
        if (this.drag?.id !== event.pointerId) return;
        this.drag = null;
        delete surface.dataset.dragging;
      };
      for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
        surface.addEventListener(event, finishDrag);
      }
      surface.addEventListener("contextmenu", (event) => event.preventDefault());
      surface.addEventListener("dragstart", (event) => event.preventDefault());
      surface.addEventListener("wheel", (event) => {
        event.preventDefault();
        const delta = event.deltaY * ([1, 16, 200][event.deltaMode] || 1);
        this.zoomBy(Math.exp(clamp(-delta * 0.0013, -0.28, 0.28)));
      }, { passive: false });
      surface.addEventListener("dblclick", () => this.reset());
      surface.addEventListener("keydown", (event) => {
        const directions = { ArrowLeft: [-18, 0], ArrowRight: [18, 0], ArrowUp: [0, -18], ArrowDown: [0, 18] };
        if (Object.hasOwn(directions, event.key)) this.move(...directions[event.key], this.panMode || event.shiftKey);
        else if (event.key === "+" || event.key === "=") this.zoomBy(1.15);
        else if (event.key === "-") this.zoomBy(1 / 1.15);
        else if (event.key === "Home") this.reset();
        else return;
        event.preventDefault();
      });
    }

    invalidate() {
      this.dirty = true;
      if (!this.visible || !this.endpoint || this.abort || this.pending) return;
      this.pending = requestAnimationFrame(() => {
        this.pending = null;
        void this.render();
      });
    }

    invalidateForMotion() {
      this.moving = true;
      clearTimeout(this.settleTimer);
      this.settleTimer = setTimeout(() => {
        this.moving = false;
        this.invalidate();
      }, 180);
      this.invalidate();
    }

    async render() {
      const endpoint = this.endpoint;
      if (!this.visible || !endpoint || this.abort) return;
      this.dirty = false;
      this.abort = new AbortController();
      if (!this.imageUrl) {
        this.status.textContent = "正在加载 3D 模型…";
        this.status.hidden = false;
      }
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...this.camera, quality: this.moving ? "preview" : "high" }),
          signal: this.abort.signal,
          cache: "no-store",
        });
        if (!response.ok) {
          const body = await response.json();
          throw new Error(body.error || "无法渲染这个视角");
        }
        const frame = await response.blob();
        if (!this.visible || endpoint !== this.endpoint) return;
        if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
        this.imageUrl = URL.createObjectURL(frame);
        this.image.src = this.imageUrl;
        this.image.hidden = false;
        this.status.hidden = true;
      } catch (error) {
        if (error.name !== "AbortError" && this.visible && endpoint === this.endpoint) {
          this.status.textContent = error.message;
          this.status.hidden = false;
        }
      } finally {
        this.abort = null;
        if (this.dirty) this.invalidate();
      }
    }
  }

  window.InteractiveGaussianView = InteractiveGaussianView;
})();
