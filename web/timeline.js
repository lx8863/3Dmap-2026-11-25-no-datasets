(() => {
  "use strict";

  function formatTime(ms) {
    const tenths = Math.round(ms / 100);
    const minutes = String(Math.floor(tenths / 600)).padStart(2, "0");
    const seconds = String(Math.floor(tenths / 10) % 60).padStart(2, "0");
    return `${minutes}:${seconds}.${tenths % 10}`;
  }

  class DatasetTimeline {
    constructor(root) {
      for (const key of ["info", "status", "image", "previous", "play", "next", "range", "duration", "time"]) {
        this[key] = root.querySelector(`#timeline-${key}`);
      }
      this.stage = root.querySelector("#timeline-stage");
      this.frameNumber = root.querySelector("#timeline-frame-number");
      this.frames = [];
      this.index = -1;
      this.target = 0;
      this.active = false;
      this.playing = false;
      this.previous.addEventListener("click", () => this.seek(this.target - 1));
      this.next.addEventListener("click", () => this.seek(this.target + 1));
      this.range.addEventListener("input", () => this.seek(Number(this.range.value)));
      this.play.addEventListener("click", () => this.togglePlayback());
    }

    activate(active) {
      this.active = active;
      if (!active) this.pause();
      else if (!this.frames.length && !this.loadingManifest) void this.loadManifest();
      else void this.present();
    }

    message(value) {
      this.status.textContent = value;
      this.status.hidden = !value;
    }

    async loadManifest() {
      this.loadingManifest = true;
      this.message("正在读取时间轴…");
      try {
        const response = await fetch("/api/timeline", { cache: "no-store" });
        if (!response.ok) throw new Error();
        const { scene, frames } = await response.json();
        this.frames = frames;
        this.info.textContent = `${scene} · 前视原始影像 · ${frames.length} 帧`;
        this.range.max = String(frames.length - 1);
        this.duration.textContent = formatTime(frames.at(-1).offsetMs);
        this.updateControls();
        await this.present();
      } catch {
        this.message("时间轴暂时无法加载。");
      } finally {
        this.loadingManifest = false;
      }
    }

    updateControls() {
      const length = this.frames.length;
      this.range.disabled = length < 2;
      this.play.disabled = length < 2;
      this.previous.disabled = length < 2 || this.target === 0;
      this.next.disabled = length < 2 || this.target === length - 1;
      this.play.textContent = this.playing ? "暂停" : "播放";
      this.play.setAttribute("aria-pressed", String(this.playing));
    }

    pause() {
      this.playing = false;
      clearTimeout(this.timer);
      this.updateControls();
    }

    seek(index) {
      this.pause();
      this.request(index);
    }

    request(index) {
      if (!this.frames.length) return;
      this.target = Math.max(0, Math.min(index, this.frames.length - 1));
      this.range.value = String(this.target);
      this.updateControls();
      void this.present();
    }

    togglePlayback() {
      if (this.playing) {
        this.pause();
        return;
      }
      if (this.frames.length < 2) return;
      this.playing = true;
      this.updateControls();
      if (this.target >= this.frames.length - 1) this.request(0);
      else if (this.index === this.target) this.scheduleNext(0);
    }

    scheduleNext(loadTime) {
      const next = this.frames[this.index + 1];
      if (!next) {
        this.pause();
        return;
      }
      const wait = Math.max(0, next.offsetMs - this.frames[this.index].offsetMs - loadTime);
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.request(this.index + 1), wait);
    }

    async present() {
      if (!this.active || !this.frames.length || this.loading || this.target === this.index) return;
      const index = this.target;
      const frame = this.frames[index];
      const time = formatTime(frame.offsetMs);
      const started = performance.now();
      this.loading = true;
      this.stage.setAttribute("aria-busy", "true");
      const image = new Image();
      image.alt = `前视原图 · ${time}`;
      image.dataset.timestamp = frame.timestamp;
      image.src = `/api/timeline/${index}/reference`;
      let loaded = false;
      try {
        await image.decode();
        loaded = true;
      } catch {
        loaded = false;
      }
      this.loading = false;
      this.stage.setAttribute("aria-busy", "false");
      if (this.active && index === this.target) {
        this.image.replaceChildren(...(loaded ? [image] : []));
        this.index = index;
        this.time.textContent = time;
        this.frameNumber.textContent = `${index + 1} / ${this.frames.length} 帧`;
        this.range.setAttribute("aria-valuetext", `${time}，第 ${index + 1} 帧`);
        this.message(loaded ? "" : "当前帧暂时无法读取。");
        if (this.playing) this.scheduleNext(performance.now() - started);
      }
      if (this.active && this.target !== this.index) void this.present();
    }
  }

  window.DatasetTimeline = DatasetTimeline;
})();
