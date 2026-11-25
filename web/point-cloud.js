(() => {
  'use strict';

  const DEFAULT_AZIMUTH = -0.52;
  const DEFAULT_ELEVATION = 0.245;
  const CAMERA_DISTANCE = 5.2;
  const DEFAULT_POINT_COLOR = [79, 112, 143];
  const clamp = (number, min, max) => Math.min(Math.max(number, min), max);

  function viewProjection(azimuth, elevation) {
    const ca = Math.cos(azimuth), sa = Math.sin(azimuth);
    const ce = Math.cos(elevation), se = Math.sin(elevation);
    return (x, y, z) => {
      const across = x * ca - y * sa;
      const into = x * sa + y * ca;
      const depth = into * ce - z * se;
      const perspective = CAMERA_DISTANCE / Math.max(0.35, CAMERA_DISTANCE + depth);
      return [across * perspective, -(z * ce + into * se) * perspective, depth, perspective];
    };
  }

  function pointBounds(positions, axis, edge) {
    const values = new Float32Array(positions.length / 3);
    for (let index = 0; index < values.length; index++) values[index] = positions[index * 3 + axis];
    values.sort();
    return [values[edge], values[values.length - 1 - edge]];
  }

  function createDemoCloud() {
    let seed = 61725;
    const random = () => {
      seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const rows = [];
    const footing = [61, 91, 118];
    const wall = [78, 106, 132];
    const limestone = [71, 105, 137];
    const trim = [57, 94, 128];
    const bedrock = [81, 124, 139];

    function stone(x, y, z, color, shade) {
      const grain = (random() - 0.48) * 68 + Math.sin(x * 13 + y * 7 + z * 8) * 10;
      const grainColor = random();
      const tint = grainColor > 0.969 ? [65, 34, 2] : grainColor < 0.05 ? [-10, 14, 20] : [0, 0, 0];
      const colorChannels = color.map((value, channel) =>
        clamp(Math.round(value * shade + grain + tint[channel]), 0, 255));
      const drift = () => (random() - 0.5) * 0.027;
      rows.push([x + drift(), y + drift(), z + drift(), ...colorChannels]);
    }

    function surface(origin, u, v, density, color, shade, finish = () => 1) {
      const area = Math.hypot(
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
      );
      const count = Math.round(area * density);
      for (let index = 0; index < count; index++) {
        const a = random();
        const b = random();
        const x = origin[0] + a * u[0] + b * v[0];
        const y = origin[1] + a * u[1] + b * v[1];
        const z = origin[2] + a * u[2] + b * v[2];
        const texture = finish(x, y, z);
        if (texture > 0) stone(x, y, z, color, shade * texture);
      }
    }

    function block(x0, x1, y0, y1, z0, z1, density, color, finish) {
      const x = [x1 - x0, 0, 0];
      const y = [0, y1 - y0, 0];
      const z = [0, 0, z1 - z0];
      surface([x0, y0, z0], x, z, density, color, 0.9, finish);
      surface([x0, y1, z0], x, z, density, color, 0.68, finish);
      surface([x0, y0, z0], y, z, density, color, 0.78, finish);
      surface([x1, y0, z0], y, z, density, color, 0.96, finish);
      surface([x0, y0, z1], x, y, density, color, 1.04, finish);
    }

    function cylinder(x, y, low, high, radiusLow, radiusHigh, count, color, flutes = 0) {
      for (let index = 0; index < count; index++) {
        const angle = random() * Math.PI * 2;
        const distance = random();
        const z = low + distance * (high - low);
        const groove = flutes ? Math.cos(angle * flutes) * 0.012 : 0;
        const radius = radiusLow + (radiusHigh - radiusLow) * distance + groove;
        const shade = 0.86 + Math.cos(angle + 0.68) * 0.16 + groove * 6;
        const seam = flutes && (z * 2.45) % 1 < 0.034 ? 0.72 : 1;
        stone(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius, z, color, shade * seam);
      }
    }

    function column(x, y, high = 4.45) {
      block(x - 0.37, x + 0.37, y - 0.37, y + 0.37, 0.76, 0.90, 121, footing);
      cylinder(x, y, 0.90, 1.00, 0.37, 0.34, 138, trim);
      cylinder(x, y, 1.00, 1.10, 0.34, 0.29, 117, limestone);
      cylinder(x, y, 1.10, high - 0.25, 0.28, 0.224, high < 4 ? 690 : 1160, limestone, 18);
      if (high < 4) {
        block(x - 0.19, x + 0.19, y - 0.19, y + 0.19, high - 0.28, high - 0.22, 150, limestone);
        return;
      }
      cylinder(x, y, high - 0.25, high - 0.12, 0.23, 0.37, 152, trim);
      block(x - 0.38, x + 0.38, y - 0.38, y + 0.38, high - 0.12, high, 138, trim);
    }

    function masonry(x, y, z) {
      const course = Math.floor(z / 0.31);
      const join = (x + y + 16 + (course % 2) * 0.34) % 0.70;
      return z % 0.31 < 0.026 || join < 0.031 ? 0.55 : 1;
    }

    // Stepped plinth and partially excavated ground.
    block(-5.22, 5.22, -3.15, 3.1, 0.08, 0.27, 37, bedrock, masonry);
    block(-5.04, 5.04, -2.96, 2.94, 0.27, 0.44, 44, footing, masonry);
    block(-4.85, 4.85, -2.76, 2.75, 0.44, 0.60, 45, footing, masonry);
    block(-4.64, 4.64, -2.54, 2.55, 0.60, 0.77, 39, wall, masonry);
    for (let index = 0; index < 520; index++) {
      const x = (random() - 0.5) * 11.4;
      const y = (random() - 0.5) * 7.1;
      if (Math.abs(x) > 5.1 || Math.abs(y) > 3.02) stone(x, y, 0.10, bedrock, 0.65);
    }

    // The surviving portico and shortened shafts along the right aisle.
    for (const x of [-4.05, -2.57, -1.10, 1.10, 2.57, 4.05]) column(x, -2.03);
    for (const [x, y, high] of [[-4.05, -0.48, 4.45], [-4.05, 1.15, 4.45], [-4.05, 2.02, 3.38],
      [4.05, -0.48, 4.45], [4.05, 1.15, 3.62], [4.05, 2.02, 2.47]]) {
      column(x, y, high);
    }

    // Fractured cella walls and the voussoirs of the stone portal.
    block(-3.03, 3.03, -0.96, -0.69, 0.77, 4.64, 157, wall, (x, y, z) => {
      if (Math.abs(x) < 1.04 && z < 2.57 + Math.sqrt(1.04 ** 2 - x ** 2)) return 0;
      return masonry(x, y, z);
    });
    for (const side of [-1, 1]) {
      const x = side * 2.89;
      block(x - 0.15, x + 0.15, -0.69, 2.18, 0.77, 4.55, 104, wall, (px, y, z) => {
        const courseBreak = side === 1 && y > -0.24 ? Math.floor((y + 1) * 2.1) * 0.32 : 0;
        return z < 4.45 - courseBreak - Math.abs(Math.sin(y * 14)) * 0.09 ? masonry(px, y, z) : 0;
      });
    }
    for (let index = 0; index < 1300; index++) {
      const angle = random() * Math.PI;
      const radius = 1.04 + random() * 0.27;
      const seam = (angle * 4.8) % 1 < 0.078 ? 0.55 : 1;
      stone(Math.cos(angle) * radius, -0.978 - random() * 0.044, 2.57 + Math.sin(angle) * radius, trim, seam);
    }
    block(-1.28, -1.04, -1.05, -0.93, 0.78, 2.58, 174, trim);
    block(1.04, 1.28, -1.05, -0.93, 0.78, 2.58, 174, trim);

    // Architrave, dentil course and triangular pediment.
    block(-4.47, 4.47, -2.48, -1.59, 4.43, 4.65, 105, limestone);
    block(-4.36, 4.36, -2.38, -1.67, 4.65, 4.91, 104, wall, masonry);
    block(-4.56, 4.56, -2.55, -1.55, 4.92, 5.11, 112, trim);
    block(-4.41, -3.73, -1.63, 1.79, 4.44, 4.91, 116, limestone, masonry);
    block(3.73, 4.41, -1.63, -0.14, 4.44, 4.93, 116, limestone, masonry);
    for (let x = -4.22; x < 4.36; x += 0.38) {
      block(x, x + 0.19, -2.43, -2.36, 4.71, 4.92, 140, trim);
    }
    const peak = (x) => 6.53 - Math.abs(x) * 0.316;
    for (const y of [-2.42, -1.72]) {
      surface([-4.43, y, 5.10], [8.86, 0, 0], [0, 0, 1.47], 245, limestone, y < -2 ? 0.91 : 0.73,
        (x, py, z) => z < peak(x) - 0.12 ? masonry(x, py, z) : 0);
    }
    for (const side of [-1, 1]) {
      const slope = [side * 4.64, 0, -1.468];
      surface([0, -2.58, 6.56], slope, [0, 1.03, 0], 174, trim, 1.04);
      surface([0, -2.58, 6.44], slope, [0, 0, 0.12], 452, trim, 1.04);
      surface([0, -2.45, 6.30], slope, [0, 0, 0.08], 387, wall, 0.90);
    }
    // A remnant of the roof and fallen masonry on the east side.
    surface([0, -1.54, 6.53], [-4.48, 0, -1.42], [0, 2.82, 0], 122, wall, 0.88,
      (x, y) => y < 0.27 + Math.cos(x * 3.7) * 0.58 ? (y * 3 + 12) % 1 < 0.08 ? 0.62 : 1 : 0);
    for (const [x, y, width, depth, height] of [[3.71, 1.37, 0.39, 0.44, 0.25], [3.16, 1.95, 0.48, 0.36, 0.31],
      [3.50, 2.23, 0.41, 0.32, 0.19], [4.27, 1.69, 0.31, 0.40, 0.21], [-3.37, 2.04, 0.44, 0.35, 0.34],
      [3.38, -0.56, 0.27, 0.33, 0.20], [4.18, -3.02, 0.30, 0.24, 0.19]]) {
      block(x, x + width, y, y + depth, 0.77, 0.77 + height, 135, limestone);
    }

    return rows;
  }

  let demoCloud;

  class PointCloudViewer {
    constructor(canvas) {
      this.canvas = canvas;
      this.context = canvas.getContext('2d');
      this.layer = document.createElement('canvas');
      this.layerContext = this.layer.getContext('2d');
      this.positions = new Float32Array(0);
      this.colors = new Uint8Array(0);
      this.opacities = new Float32Array(0);
      this.ground = -0.55;
      this.frame = null;
      this.drag = null;
      this.panMode = false;
      this.initialAzimuth = DEFAULT_AZIMUTH;
      this.initialElevation = DEFAULT_ELEVATION;
      this.focusSize = null;
      this.canvas.style.touchAction = 'none';
      if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
      this.bindControls();
      this.resizeObserver = new ResizeObserver(() => this.invalidate());
      this.resizeObserver.observe(canvas);
      this.reset();
    }

    get pointCount() {
      return this.positions.length / 3;
    }

    showDemo() {
      demoCloud ??= createDemoCloud();
      this.setPoints(demoCloud);
    }

    setPoints(rows, viewFrame = null, focused = false) {
      const usable = rows.filter((row) => Array.isArray(row) && Number.isFinite(row[0]) &&
        Number.isFinite(row[1]) && Number.isFinite(row[2]));
      const count = usable.length;
      const hasFrame = viewFrame && [viewFrame.center, viewFrame.right, viewFrame.forward, viewFrame.up]
        .every((vector) => Array.isArray(vector) && vector.length === 3 && vector.every(Number.isFinite));
      const hasFocus = hasFrame && focused && Number.isFinite(viewFrame.radius) && viewFrame.radius > 0;
      this.initialAzimuth = hasFrame ? 0 : DEFAULT_AZIMUTH;
      this.initialElevation = hasFrame && Number.isFinite(viewFrame.elevation)
        ? clamp(viewFrame.elevation, -1.51, 1.51) : DEFAULT_ELEVATION;
      this.focusSize = null;
      this.positions = new Float32Array(count * 3);
      this.colors = new Uint8Array(count * 3);
      this.opacities = new Float32Array(count);
      if (count === 0) {
        this.ground = -0.55;
        this.reset();
        return;
      }
      const basis = hasFrame ? [viewFrame.right, viewFrame.forward, viewFrame.up] : null;
      usable.forEach((point, index) => {
        const offset = hasFrame ? [point[0] - viewFrame.center[0], point[1] - viewFrame.center[1],
          point[2] - viewFrame.center[2]] : point;
        for (let axis = 0; axis < 3; axis++) {
          const direction = basis?.[axis];
          this.positions[index * 3 + axis] = direction
            ? offset[0] * direction[0] + offset[1] * direction[1] + offset[2] * direction[2] : offset[axis];
          const color = point[axis + 3];
          this.colors[index * 3 + axis] = Number.isFinite(color) ? clamp(color, 0, 255) : DEFAULT_POINT_COLOR[axis];
        }
        this.opacities[index] = Number.isFinite(point[6]) ? Math.sqrt(clamp(point[6], 0, 1)) : 1;
      });

      const edge = count > 200 ? Math.floor(count * 0.004) : 0;
      const bounds = [0, 1, 2].map((axis) => pointBounds(this.positions, axis, edge));
      const scale = hasFocus ? 1 / viewFrame.radius : 2 / (Math.max(...bounds.map(([min, max]) => max - min)) || 1);
      const center = hasFocus ? [0, 0, 0] : bounds.map(([min, max]) => (min + max) / 2);
      for (let index = 0; index < this.positions.length; index++) {
        this.positions[index] = (this.positions[index] - center[index % 3]) * scale;
      }
      this.ground = (bounds[2][0] - center[2]) * scale - 0.025;
      if (hasFocus) this.measureFocus(edge);
      this.reset();
    }

    measureFocus(edge) {
      const project = viewProjection(this.initialAzimuth, this.initialElevation);
      const width = new Float32Array(this.pointCount);
      const height = new Float32Array(this.pointCount);
      for (let index = 0; index < this.positions.length; index += 3) {
        const [x, y] = project(this.positions[index], this.positions[index + 1], this.positions[index + 2]);
        width[index / 3] = Math.abs(x);
        height[index / 3] = Math.abs(y);
      }
      width.sort();
      height.sort();
      const upper = this.pointCount - 1 - edge;
      this.focusSize = [Math.max(0.55, width[upper]), Math.max(0.55, height[upper])];
    }

    reset() {
      this.azimuth = this.initialAzimuth;
      this.elevation = this.initialElevation;
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
      this.invalidate();
    }

    orbit(dx, dy) {
      this.azimuth += dx;
      this.elevation = clamp(this.elevation + dy, -1.51, 1.51);
      this.invalidate();
    }

    zoomBy(factor) {
      this.zoom = clamp(this.zoom * factor, 0.38, 4.5);
      this.invalidate();
    }

    setPanMode(enabled) {
      this.panMode = Boolean(enabled);
      if (!this.drag) this.canvas.style.cursor = this.panMode ? 'move' : '';
    }

    panBy(dx, dy) {
      this.panX += dx;
      this.panY += dy;
      this.invalidate();
    }

    bindControls() {
      const canvas = this.canvas;
      canvas.addEventListener('pointerdown', (event) => {
        if ((event.button !== 0 && event.button !== 2) || this.drag) return;
        const pan = this.panMode || event.shiftKey || event.button === 2;
        this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, pan };
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = pan ? 'move' : 'grabbing';
      });
      canvas.addEventListener('pointermove', (event) => {
        if (this.drag?.id !== event.pointerId) return;
        const dx = event.clientX - this.drag.x;
        const dy = event.clientY - this.drag.y;
        if (this.drag.pan) this.panBy(dx, dy);
        else this.orbit(dx * 0.009, dy * 0.007);
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
      });
      canvas.addEventListener('lostpointercapture', () => {
        this.drag = null;
        this.setPanMode(this.panMode);
      });
      canvas.addEventListener('contextmenu', (event) => event.preventDefault());
      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const delta = event.deltaY * [1, 16, 200][event.deltaMode];
        this.zoomBy(Math.exp(clamp(-delta * 0.0013, -0.25, 0.25)));
      }, { passive: false });
      canvas.addEventListener('dblclick', () => this.reset());
      canvas.addEventListener('keydown', (event) => {
        const pan = this.panMode || event.shiftKey;
        const actions = {
          ArrowLeft: () => pan ? this.panBy(-24, 0) : this.orbit(-0.14, 0),
          ArrowRight: () => pan ? this.panBy(24, 0) : this.orbit(0.14, 0),
          ArrowUp: () => pan ? this.panBy(0, -24) : this.orbit(0, 0.12),
          ArrowDown: () => pan ? this.panBy(0, 24) : this.orbit(0, -0.12),
          '+': () => this.zoomBy(1.14),
          '=': () => this.zoomBy(1.14),
          '-': () => this.zoomBy(1 / 1.14),
          Home: () => this.reset(),
        };
        if (!Object.hasOwn(actions, event.key)) return;
        event.preventDefault();
        actions[event.key]();
      });
    }

    invalidate() {
      if (this.frame !== null) return;
      this.frame = requestAnimationFrame(() => {
        this.frame = null;
        this.draw();
      });
    }

    draw() {
      const width = this.canvas.clientWidth;
      const height = this.canvas.clientHeight;
      if (width < 1 || height < 1) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const pixelsWide = Math.round(width * dpr);
      const pixelsHigh = Math.round(height * dpr);
      if (this.canvas.width !== pixelsWide || this.canvas.height !== pixelsHigh || !this.image) {
        this.canvas.width = this.layer.width = pixelsWide;
        this.canvas.height = this.layer.height = pixelsHigh;
        this.image = this.layerContext.createImageData(pixelsWide, pixelsHigh);
        this.depthBuffer = new Float32Array(pixelsWide * pixelsHigh);
      }

      const context = this.context;
      const fit = this.focusSize
        ? Math.min(width * 0.43 / this.focusSize[0], height * 0.42 / this.focusSize[1])
        : Math.min(width * 0.335, height * 0.448);
      const focal = fit * this.zoom;
      const centerX = width / 2 + this.panX;
      const centerY = height * (this.focusSize ? 0.5 : 0.404) + this.panY;
      const view = viewProjection(this.azimuth, this.elevation);
      const project = (x, y, z) => {
        const [across, down, depth, perspective] = view(x, y, z);
        return [centerX + across * focal, centerY + down * focal, depth, perspective];
      };

      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      const backdrop = context.createRadialGradient(width * 0.51, height * 0.44, 0,
        width * 0.51, height * 0.44, Math.max(width * 0.68, height * 0.88));
      backdrop.addColorStop(0, '#f8fbff');
      backdrop.addColorStop(1, '#e8f0fa');
      context.fillStyle = backdrop;
      context.fillRect(0, 0, width, height);
      this.drawGrid(context, project);
      this.drawPoints(project, dpr, pixelsWide, pixelsHigh);
      context.drawImage(this.layer, 0, 0, width, height);
    }

    drawGrid(context, project) {
      const size = 2.4;
      context.lineWidth = 0.65;
      for (let step = -12; step <= 12; step++) {
        const coordinate = step * 0.2;
        const fade = 1 - Math.abs(step) / 15;
        context.strokeStyle = `rgba(96, 125, 165, ${fade * (step === 0 ? 0.24 : 0.14)})`;
        const [x0, y0] = project(coordinate, -size, this.ground);
        const [x1, y1] = project(coordinate, size, this.ground);
        const [x2, y2] = project(-size, coordinate, this.ground);
        const [x3, y3] = project(size, coordinate, this.ground);
        context.beginPath();
        context.moveTo(x0, y0);
        context.lineTo(x1, y1);
        context.moveTo(x2, y2);
        context.lineTo(x3, y3);
        context.stroke();
      }
    }

    drawPoints(project, dpr, width, height) {
      const data = this.image.data;
      const depthBuffer = this.depthBuffer;
      const pointSize = this.focusSize ? 1.27 : 0.93;
      const contrast = this.focusSize ? 0.9 : 1;
      data.fill(0);
      depthBuffer.fill(Infinity);
      for (let i = 0; i < this.positions.length; i += 3) {
        const weight = this.opacities[i / 3];
        if (weight === 0) continue;
        const [x, y, depth, scale] = project(this.positions[i], this.positions[i + 1], this.positions[i + 2]);
        if (depth < 0.35 - CAMERA_DISTANCE) continue;
        const xCenter = x * dpr;
        const yCenter = y * dpr;
        const radius = dpr * clamp((pointSize + (i % 13) * 0.022) * scale * this.zoom ** 0.27, 0.72, 1.8);
        const radiusSquared = radius * radius;
        const left = Math.max(0, Math.floor(xCenter - radius));
        const top = Math.max(0, Math.floor(yCenter - radius));
        const right = Math.min(width - 1, Math.floor(xCenter + radius));
        const bottom = Math.min(height - 1, Math.floor(yCenter + radius));
        const opacity = clamp(0.87 - depth * 0.16, 0.42, 1) * weight;
        for (let py = top; py <= bottom; py++) {
          for (let px = left; px <= right; px++) {
            const distance = ((px + 0.5 - xCenter) ** 2 + (py + 0.5 - yCenter) ** 2) / radiusSquared;
            const index = py * width + px;
            if (distance >= 1 || depth >= depthBuffer[index]) continue;
            depthBuffer[index] = depth;
            const pixel = index * 4;
            data[pixel] = this.colors[i] * contrast;
            data[pixel + 1] = this.colors[i + 1] * contrast;
            data[pixel + 2] = this.colors[i + 2] * contrast;
            data[pixel + 3] = Math.round(clamp((1 - distance) * 2.15, 0, 1) * opacity * 255);
          }
        }
      }
      this.layerContext.putImageData(this.image, 0, 0);
    }
  }

  window.PointCloudViewer = PointCloudViewer;
})();
