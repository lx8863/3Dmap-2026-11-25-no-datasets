(() => {
  "use strict";
  if (["#timeline", "#training", "#training-3d"].includes(location.hash)) {
    location.replace("/workbench" + location.hash);
    return;
  }

  const $ = (id) => document.getElementById(id);
  const format = new Intl.NumberFormat("zh-CN");
  const limits = { vggt: [2, 12], colmap: [3, 128] };
  const mib = 1024 ** 2;
  const form = $("create-form");
  const photoInput = $("photos");
  const photos = [];
  const viewer = new window.InteractiveGaussianView($("free-view"), $("free-image"), $("free-status"));
  const sampleViewer = new window.InteractiveGaussianView($("sample-view"), $("sample-frame"), $("sample-status"));
  let sampleOpen = false;
  let status;
  let online = false;
  let sending = false;
  let notice = "";
  let uploadPercent = 0;
  let statusRevision = 0;
  let result;
  let viewIndex = 0;
  let viewMode = "render";

  function engine() {
    return form.querySelector('input[name="engine"]:checked').value;
  }

  function addPhotos(files) {
    if (sending || status?.job?.state === "running") return;
    notice = "";
    let bytes = photos.reduce((total, { file }) => total + file.size, 0);
    for (const file of files) {
      if (!/\.(jpe?g|png|webp)$/i.test(file.name) || !file.size || file.size > 32 * mib) {
        notice = "请添加每张不超过 32 MB 的 JPG、PNG 或 WebP 照片。";
        continue;
      }
      const duplicate = photos.some(({ file: item }) =>
        item.name === file.name && item.size === file.size && item.lastModified === file.lastModified);
      if (duplicate) continue;
      if (photos.length >= 128 || bytes + file.size > 512 * mib) {
        notice = "一次最多选 128 张照片，合计不超过 512 MB。";
        break;
      }
      photos.push({ file, url: URL.createObjectURL(file) });
      bytes += file.size;
    }
    showPhotos();
    renderStatus();
  }

  function showPhotos() {
    const grid = $("photo-grid");
    grid.replaceChildren(...photos.map((photo) => {
      const tile = document.createElement("div");
      tile.className = "photo";
      tile.title = photo.file.name;
      const preview = document.createElement("img");
      preview.src = photo.url;
      preview.alt = photo.file.name;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", "移除 " + photo.file.name);
      remove.addEventListener("click", () => {
        photos.splice(photos.indexOf(photo), 1);
        URL.revokeObjectURL(photo.url);
        notice = "";
        showPhotos();
        renderStatus();
      });
      tile.append(preview, remove);
      return tile;
    }));
    grid.hidden = !photos.length;
    $("photo-count").textContent = photos.length + " 张";
    $("pick-label").textContent = photos.length ? "继续添加照片" : "选择照片";
  }

  function renderStatus() {
    const selected = engine();
    const [minimum, maximum] = limits[selected];
    const busy = sending || status?.job?.state === "running";
    const connection = $("connection");
    connection.dataset.state = online ? "ready" : status ? "offline" : "pending";
    connection.textContent = online ? "本地服务已连接" : status ? "本地服务已断开" : "连接本地服务";
    const engineError = selected === "vggt" ? status?.vggtError : status?.studioColmapError;
    const error = status?.studioTrainingError || engineError;
    const tooMany = photos.length > maximum;
    const iterations = $("iterations");
    const valid = iterations.value !== "" && Number.isInteger(Number(iterations.value)) &&
      Number(iterations.value) >= 1 && Number(iterations.value) <= 30000;
    $("settings").disabled = busy;
    $("show-free").disabled = busy;
    $("start").disabled = !(online && !error && !busy && !tooMany && photos.length >= minimum && valid);
    $("start").textContent = sending ? "正在导入照片" : busy ? "正在生成" : "开始生成";
    $("photo-hint").textContent = "选择 " + minimum + "–" + maximum + " 张有视角重叠的 JPG、PNG 或 WebP 照片。";
    const message = notice || (!online && status ? "请启动本地 3Dmap 后刷新。" : error)
      || (tooMany ? "当前标定方式最多支持 " + maximum + " 张，请移除多余照片。" : "");
    $("notice").hidden = !message;
    $("notice").textContent = message || "";
    for (const button of form.querySelectorAll("[data-steps]")) {
      button.setAttribute("aria-pressed", String(button.dataset.steps === iterations.value));
    }
    showJob();
    showSample(busy);
  }

  function showSample(busy) {
    const sample = status?.sample;
    $("advanced-nav").hidden = !status || status.desktop;
    $("sample").hidden = !sample;
    $("sample-nav").hidden = !sample;
    if (!sample) return;
    const interactive = !!sample.renderUrl;
    if (busy || !interactive) sampleOpen = false;
    if ($("sample-image").getAttribute("src") !== sample.poster) $("sample-image").src = sample.poster;
    $("sample-image").hidden = sampleOpen;
    $("sample-view").hidden = !sampleOpen;
    $("sample-reset").hidden = !sampleOpen;
    $("sample-open").disabled = busy || !online || !interactive;
    $("sample-open").setAttribute("aria-pressed", String(sampleOpen));
    $("sample-open").textContent = sampleOpen ? "返回渲染图" : "体验自由视角";
    $("sample-tip").textContent = busy ? "训练进行时可先查看示例渲染图。"
      : interactive ? "按住滚轮拖动旋转，滚动缩放，右键拖动平移。" : "自由视角需要可用的 NVIDIA GPU。";
    sampleViewer.setEndpoint(sampleOpen ? sample.renderUrl : null);
    sampleViewer.setVisible(sampleOpen);
  }

  function showJob() {
    const job = status?.job?.kind === "training" ? status.job : null;
    const stageLabels = {
      calibrate: (job?.engine === "vggt" ? "VGGT" : "COLMAP") + " 正在标定相机",
      prepare: "准备 3DGS 训练",
      training: "正在进行 3DGS 训练",
      preview: "正在渲染预览",
      complete: "3DGS 模型已生成",
    };
    const progress = $("progress");
    $("job-progress").hidden = !sending && !job;
    $("log-panel").hidden = !job?.logs?.length;
    if (sending || job) {
      const steps = job?.progress;
      const training = !sending && job?.stage === "training" && steps;
      $("stage").textContent = sending ? "正在导入照片" : job.state === "failed" ? "生成失败" : stageLabels[job.stage] || "准备生成";
      $("progress-label").textContent = sending ? uploadPercent + "%" :
        training ? format.format(steps.iteration) + " / " + format.format(steps.total) + " 步" : job?.error || "";
      if (sending || training || job?.state === "succeeded" || job?.state === "failed") {
        progress.max = training ? steps.total : 100;
        progress.value = sending ? uploadPercent : training ? steps.iteration : job.state === "succeeded" ? 100 : 0;
      } else {
        progress.removeAttribute("value");
      }
      const logs = job?.logs?.slice() || [];
      if (job?.error) logs.push(job.error);
      const log = $("logs");
      const value = logs.join("\n");
      if (log.textContent !== value) {
        const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 35;
        log.textContent = value;
        if (atEnd) log.scrollTop = log.scrollHeight;
      }
      if (!result) {
        const failed = job?.state === "failed";
        $("empty-title").textContent = failed ? "生成未完成" : "正在生成";
        $("empty-detail").textContent = failed ? job.error || "请查看运行详情。" : "相机标定完成后会自动开始 3DGS 训练。";
      }
    }
    const next = status?.trainingResult;
    if (next?.downloadUrl && next.id !== result?.id) {
      result = next;
      viewIndex = 0;
      viewMode = result.views.length ? "render" : result.interactive ? "free" : "render";
      $("result-detail").textContent = (result.engine === "vggt" ? "VGGT" : result.engine === "colmap" ? "COLMAP" : "3DGS")
        + " · " + format.format(result.progress.total) + " 步";
      const download = $("download");
      download.href = result.downloadUrl;
      download.download = "3Dmap-" + result.id.slice(0, 8) + ".ply";
      download.setAttribute("aria-disabled", "false");
      showView();
    }
  }

  function showView() {
    const views = result?.views || [];
    const current = views[viewIndex];
    const free = viewMode === "free" && result?.interactive;
    const image = $("render");
    $("render-controls").hidden = !views.length && !result?.interactive;
    $("show-free").hidden = !result?.interactive;
    $("show-render").hidden = !views.length;
    $("show-original").hidden = !current?.referenceUrl;
    $("camera-controls").hidden = free || views.length <= 1;
    $("camera-index").textContent = (viewIndex + 1) + " / " + views.length;
    $("show-render").setAttribute("aria-pressed", String(viewMode === "render"));
    $("show-original").setAttribute("aria-pressed", String(viewMode === "original"));
    $("show-free").setAttribute("aria-pressed", String(free));
    $("empty-preview").hidden = free || !!current;
    if (!free && !current) {
      $("empty-title").textContent = "模型已生成";
      $("empty-detail").textContent = "可下载 PLY，渲染预览暂不可用。";
    }
    $("free-view").hidden = !free;
    $("orbit-hint").hidden = !free;
    image.hidden = free || !current;
    if (!image.hidden) {
      image.src = viewMode === "original" && current.referenceUrl ? current.referenceUrl : current.url;
      image.alt = viewMode === "original" ? "原照片" : "3DGS 渲染结果";
    }
    viewer.setEndpoint(free ? "/api/jobs/" + result.id + "/render" : null);
    viewer.setVisible(!!free);
  }

  async function poll() {
    const revision = statusRevision;
    try {
      const response = await fetch("/api/status", { signal: AbortSignal.timeout(4500) });
      if (!response.ok) throw new Error("连接失败");
      const next = await response.json();
      if (revision === statusRevision) {
        status = next;
        online = true;
      }
    } catch {
      if (revision === statusRevision) {
        online = false;
        status ||= {};
      }
    }
    renderStatus();
    setTimeout(poll, status?.job?.state === "running" ? 750 : 2000);
  }

  function submitUpload() {
    const files = photos.map(({ file }) => file);
    const header = new TextEncoder().encode(JSON.stringify({
      engine: engine(), iterations: Number($("iterations").value), sizes: files.map((file) => file.size),
    }));
    const size = new DataView(new ArrayBuffer(4));
    size.setUint32(0, header.byteLength);
    const body = new Blob([size.buffer, header, ...files], { type: "application/octet-stream" });
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/photo-jobs");
      request.responseType = "json";
      request.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
          uploadPercent = Math.round(100 * event.loaded / event.total);
          renderStatus();
        }
      });
      request.addEventListener("load", () => {
        if (request.status === 202) resolve(request.response);
        else reject(new Error(request.response?.error || "上传失败，请重试。"));
      });
      request.addEventListener("error", () => reject(new Error("无法连接本地服务。")));
      request.send(body);
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("start").disabled || !form.reportValidity()) return;
    sending = true;
    uploadPercent = 0;
    notice = "";
    statusRevision++;
    if (result?.views.length && viewMode === "free") { viewMode = "render"; showView(); }
    renderStatus();
    try {
      status = await submitUpload();
    } catch (error) {
      notice = error.message;
    } finally {
      statusRevision++;
      sending = false;
      renderStatus();
    }
  });
  form.addEventListener("input", () => { notice = ""; renderStatus(); });
  $("photo-picker").addEventListener("click", () => photoInput.click());
  photoInput.addEventListener("change", () => { addPhotos(photoInput.files); photoInput.value = ""; });
  const drop = $("photo-picker");
  drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.dataset.dragging = "true"; });
  drop.addEventListener("dragleave", () => { delete drop.dataset.dragging; });
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    delete drop.dataset.dragging;
    addPhotos(event.dataTransfer.files);
  });
  for (const button of form.querySelectorAll("[data-steps]")) {
    button.addEventListener("click", () => { $("iterations").value = button.dataset.steps; renderStatus(); });
  }
  for (const mode of ["render", "original", "free"]) {
    $("show-" + mode).addEventListener("click", () => {
      if (mode === "free" && sampleOpen) {
        sampleOpen = false;
        showSample(false);
      }
      viewMode = mode;
      showView();
    });
  }
  for (const [id, direction] of [["previous", -1], ["next", 1]]) {
    $(id).addEventListener("click", () => {
      const length = result.views.length;
      viewIndex = (viewIndex + direction + length) % length;
      showView();
    });
  }
  $("sample-open").addEventListener("click", () => {
    sampleOpen = !sampleOpen;
    if (sampleOpen && viewMode === "free") {
      viewMode = "render";
      showView();
    }
    showSample(false);
    if (sampleOpen) $("sample-view").focus({ preventScroll: true });
  });
  $("sample-reset").addEventListener("click", () => sampleViewer.reset());

  $("download").addEventListener("click", async (event) => {
    if (!result) { event.preventDefault(); return; }
    if (typeof window.showSaveFilePicker !== "function") return;
    event.preventDefault();
    const { href, download } = event.currentTarget;
    try {
      const destination = await window.showSaveFilePicker({
        suggestedName: download,
        types: [{ description: "3D Gaussian Splatting PLY", accept: { "application/octet-stream": [".ply"] } }],
      });
      const response = await fetch(href);
      if (!response.ok || !response.body) throw new Error("下载模型失败，请重试。");
      await response.body.pipeTo(await destination.createWritable());
    } catch (error) {
      if (error.name === "SecurityError" || error.name === "NotAllowedError") {
        const fallback = document.createElement("a");
        fallback.href = href;
        fallback.download = download;
        fallback.click();
      } else if (error.name !== "AbortError") {
        notice = error.message;
        renderStatus();
      }
    }
  });

  renderStatus();
  poll();
})();
