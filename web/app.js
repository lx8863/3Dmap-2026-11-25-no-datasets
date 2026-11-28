(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const modes = [...document.querySelectorAll("[data-mode]")];
  const sections = [...document.querySelectorAll("[data-section]")];
  const steps = [...document.querySelectorAll("#steps li")];
  const tabs = [$("logs-tab"), $("command-tab")];
  const reconstructionForm = $("reconstruction-form");
  const trainingForm = $("training-form");
  const gpuSelect = $("training-gpu");
  const viewer = new window.PointCloudViewer($("point-cloud"));
  const interactiveViewer = window.InteractiveGaussianView
    ? new window.InteractiveGaussianView($("interactive-view"), $("interactive-image"), $("interactive-status")) : null;
  const datasetTimeline = window.DatasetTimeline ? new window.DatasetTimeline($("dataset-timeline")) : null;
  const numbers = new Intl.NumberFormat("zh-CN");
  const jobByKind = {};
  const stepLabels = { waiting: "等待", active: "进行中", done: "完成", failed: "失败", skipped: "已关闭" };

  let mode = "reconstruction";
  let service = null;
  let linkState = location.protocol === "file:" ? "preview" : "pending";
  let submitting = false;
  let statusRevision = 0;
  let activeModelUrl = null;
  let activeViewKey = "";
  let activeViews = [];
  let activeViewIndex = 0;
  let previewMode = "points";
  let useReference = false;
  let interactiveEndpoint = null;
  let pointData = null;
  let focusedPoints = false;
  let panMode = false;
  let previewIsTraining = false;
  let previewEngine = null;
  let gpuOptions = "";
  let copyTimeout;

  function showDemo() {
    viewer.showDemo();
    $("model-badge").textContent = "示例";
    $("model-badge").dataset.live = "false";
    $("viewport-label-text").textContent = "石柱建筑 · 示例点云";
    $("point-count").textContent = numbers.format(viewer.pointCount);
    $("camera-count").textContent = "—";
    $("model-detail-note").textContent = "拖动或使用方向键调整视角";
  }

  function renderPreviewMode() {
    const view = activeViews[activeViewIndex];
    const showImage = previewMode === "image" && !!view;
    const showInteractive = previewMode === "interactive" && !!interactiveEndpoint;
    const showPoints = !showImage && !showInteractive;
    $("preview-switch").hidden = !activeViews.length && !interactiveEndpoint;
    $("preview-interactive").hidden = !interactiveEndpoint;
    $("preview-render").hidden = !activeViews.length;
    $("preview-interactive").setAttribute("aria-pressed", String(showInteractive));
    $("preview-render").setAttribute("aria-pressed", String(showImage));
    $("preview-points").setAttribute("aria-pressed", String(showPoints));
    $("reset-view").hidden = showImage;
    $("point-cloud").hidden = !showPoints;
    $("viewport-label").hidden = showImage;
    $("point-hint").hidden = showImage;
    $("render-view").hidden = !showImage;
    $("interactive-view").hidden = !showInteractive;
    interactiveViewer?.setVisible(showInteractive);
    $("camera-tools").hidden = showImage;
    $("model-title").textContent = showPoints ? "点云预览" : "模型预览";
    $("point-scope").hidden = !showPoints || !pointData?.focusPoints?.length;
    $("point-focus").setAttribute("aria-pressed", String(focusedPoints));
    $("point-entire").setAttribute("aria-pressed", String(!focusedPoints));
    if (showInteractive) {
      $("viewport-label-text").textContent = "3DGS 自由视角";
      $("model-detail-note").textContent = "GPU 渲染当前相机视角";
    } else if (showImage) {
      $("model-detail-note").textContent = "按拍摄相机的位姿渲染";
    } else if (pointData) {
      $("viewport-label-text").textContent = previewIsTraining ? "高斯中心采样"
        : previewEngine === "vggt" ? "VGGT · 点云" : "COLMAP · 稀疏点云";
      $("model-detail-note").textContent = `显示 ${numbers.format(viewer.pointCount)} 个点`;
    }
    if (!showImage) return;

    const image = $("render-image");
    const url = useReference ? view.referenceUrl : view.url;
    if (image.getAttribute("src") !== url) {
      $("render-error").hidden = true;
      image.hidden = false;
      image.src = url;
    }
    image.alt = `${useReference ? "原始照片" : "3DGS 渲染"} · ${view.name}`;
    $("render-position").textContent = `视角 ${activeViewIndex + 1} / ${activeViews.length}`;
    $("render-previous").disabled = activeViews.length < 2;
    $("render-next").disabled = activeViews.length < 2;
    const reference = $("render-reference");
    reference.setAttribute("aria-pressed", String(useReference));
    reference.textContent = useReference ? "返回渲染" : "对照原图";
  }

  function selectPointScope(focus) {
    if (!pointData) return;
    focusedPoints = focus && !!pointData.focusPoints?.length;
    viewer.setPoints(focusedPoints ? pointData.focusPoints : pointData.points, pointData.viewFrame, focusedPoints);
    renderPreviewMode();
  }

  function setPanMode(enabled) {
    panMode = enabled;
    viewer.setPanMode(enabled);
    interactiveViewer?.setPanMode(enabled);
    $("camera-pan").setAttribute("aria-pressed", String(enabled));
  }

  function formOptions(form) {
    const fields = new FormData(form);
    if (form === reconstructionForm) {
      return {
        engine: $("reconstruction-engine").value,
        imagePath: $("image-path").value.trim(),
        workspacePath: $("workspace-path").value.trim(),
        dataType: $("data-type").value,
        quality: $("quality").value,
        mapper: $("mapper").value,
        useGpu: $("use-gpu").checked,
        dense: $("dense").checked,
      };
    }
    return {
      sourceType: fields.get("sourceType"),
      sourcePath: $("training-source").value.trim(),
      imagePath: $("training-image-path").value.trim(),
      modelPath: $("training-model").value.trim(),
      iterations: Number(fields.get("iterations")),
      gpu: fields.get("gpu"),
    };
  }

  function shellQuote(value) {
    return `'${value.replaceAll("'", "'\"'\"'")}'`;
  }

  function renderCommand() {
    const options = formOptions(reconstructionForm);
    const isVggt = options.engine === "vggt";
    $("copy-command").hidden = isVggt || $("command-tab").getAttribute("aria-selected") !== "true";
    if (isVggt) {
      $("command-output").textContent = "选择照片目录和新工作区后，点击「开始 VGGT 重建」。";
      return;
    }
    const binary = service?.executable ? shellQuote(service.executable) : "colmap";
    const mapper = service?.colmapMappers?.length === 1 ? "" : ` --mapper ${options.mapper}`;
    $("command-output").textContent = [
      `${binary} automatic_reconstructor`,
      `  --image_path ${shellQuote(options.imagePath || "/data/scene/images")}`,
      `  --workspace_path ${shellQuote(options.workspacePath || "/data/scene/output")}`,
      `  --data_type ${options.dataType} --quality ${options.quality}${mapper}`,
      `  --use_gpu ${Number(options.useGpu)} --dense ${Number(options.dense)}`,
    ].join(" \\\n");
  }

  function selectTab(name, focus = false) {
    if (mode === "training") name = "logs";
    for (const tab of tabs) {
      const isActive = tab.id === `${name}-tab`;
      tab.setAttribute("aria-selected", String(isActive));
      tab.tabIndex = isActive ? 0 : -1;
      $(tab.getAttribute("aria-controls")).hidden = !isActive;
      if (focus && isActive) tab.focus();
    }
    $("copy-command").hidden = name !== "command" || $("reconstruction-engine").value === "vggt";
  }

  function rememberView(anchor) {
    if (location.protocol === "file:") return;
    const address = new URL(location.href);
    address.hash = anchor;
    history.replaceState(null, "", address);
  }

  function modeFromAddress() {
    if (location.hash === "#timeline") return "timeline";
    return ["#training", "#training-3d"].includes(location.hash) ? "training" : "reconstruction";
  }

  function setMode(nextMode) {
    mode = nextMode;
    let anchor = mode === "reconstruction" ? "" : mode;
    if (mode === "training" && location.hash === "#training-3d") anchor = "training-3d";
    rememberView(anchor);
    for (const button of modes) button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    for (const section of sections) section.hidden = section.dataset.section !== mode;
    const browsingData = mode === "timeline";
    $("dataset-timeline").hidden = !browsingData;
    $("workflow-workspace").hidden = browsingData;
    datasetTimeline?.activate(browsingData);
    $("command-tab").hidden = mode === "training";
    selectTab("logs");
    render();
  }

  function renderConnection() {
    const labels = {
      pending: "检查本地服务",
      online: "本地服务 · 已连接",
      missing: "本地服务 · 缺少依赖",
      preview: "离线预览",
    };
    $("connection").dataset.state = linkState;
    $("connection-label").textContent = labels[linkState];
    $("connection").title = service?.colmapVersion ? `COLMAP ${service.colmapVersion}` : "";
  }

  function renderGpuOptions() {
    const gpus = service?.gpus || [];
    const signature = JSON.stringify(gpus);
    if (signature !== gpuOptions) {
      const previous = gpuSelect.value;
      gpuOptions = signature;
      const options = gpus.map((gpu) => {
        const architecture = gpu.computeCapability ? " · 计算能力 " + gpu.computeCapability : "";
        return new Option(gpu.name + architecture, String(gpu.index));
      });
      if (options.length) options.unshift(new Option("自动选择", "auto"));
      else options.push(new Option("未检测到 NVIDIA GPU", ""));
      gpuSelect.replaceChildren(...options);
      if (options.some((option) => option.value === previous)) gpuSelect.value = previous;
    }
  }

  function renderTrainingSource() {
    const isColmap = $("source-type").value === "colmap";
    $("training-source-label").textContent = isColmap ? "COLMAP 输出工作区" : "数据集目录";
    $("training-source").placeholder = isColmap ? "例如 /data/scene/output" : "例如 /data/scene/dataset";
    $("training-source-note").textContent = isColmap
      ? "填写运行服务的电脑上的 COLMAP 工作区，系统会自动准备去畸变图像。"
      : "填写运行服务的电脑上包含 images/ 和 sparse/0/ 的实际目录。";
    $("training-image-field").hidden = !isColmap;
    $("training-image-path").required = isColmap;
    const previous = service?.lastReconstruction;
    $("use-last-reconstruction").hidden = !previous || (!isColmap && previous.sourceType !== "dataset");
  }

  function setFormStatus(element, message, error = false) {
    element.hidden = !message;
    element.dataset.error = String(error);
    if (element.textContent !== message) element.textContent = message;
  }

  function renderForms() {
    const activeKind = service?.job?.state === "running" ? service.job.kind : null;
    const busy = submitting || !!activeKind;
    const canReconstruct = !!service?.available;
    const isVggt = $("reconstruction-engine").value === "vggt";
    const colmapGpu = service?.colmapGpu !== false;
    const colmapDense = service?.colmapDense !== false;
    const requiresColmap = $("source-type").value === "colmap";
    const canTrain = !!service?.trainingAvailable && (!requiresColmap || canReconstruct);

    for (const field of reconstructionForm.querySelectorAll("input, select")) field.disabled = activeKind === "reconstruction";
    for (const field of trainingForm.querySelectorAll("input, select")) field.disabled = activeKind === "training";
    gpuSelect.disabled = activeKind === "training" || !service?.gpus?.length;
    for (const group of reconstructionForm.querySelectorAll("[data-colmap-field]")) group.hidden = isVggt;
    $("workspace-note").textContent = isVggt
      ? "使用尚未创建的输出目录。测试版支持 2–12 张照片。"
      : "输入本机目录的绝对路径。";
    const mapper = $("mapper");
    for (const option of mapper.options) option.disabled = !!service?.colmapMappers && !service.colmapMappers.includes(option.value);
    if (mapper.selectedOptions[0]?.disabled) mapper.value = service.colmapMappers[0];
    mapper.disabled ||= service?.colmapMappers?.length === 1;
    for (const [id, supported] of [["use-gpu", colmapGpu], ["dense", colmapDense]]) {
      const input = $(id);
      if (!supported) input.checked = false;
      input.disabled = !supported || activeKind === "reconstruction";
      input.closest("label").title = supported ? "" : "当前 COLMAP 构建不提供这个功能";
    }
    renderTrainingSource();
    $("use-last-reconstruction").disabled = activeKind === "training";
    $("run-button").disabled = busy || (isVggt && !service?.vggtAvailable);
    $("run-label").textContent = activeKind === "reconstruction" ? "正在重建"
      : isVggt ? "开始 VGGT 重建" : canReconstruct ? "开始重建" : "查看重建命令";
    $("train-button").disabled = busy || !canTrain;
    $("train-label").textContent = activeKind === "training" ? "正在训练" : "开始训练";

    let reconstructionNote = "";
    if (activeKind === "training") reconstructionNote = "GPU 训练进行中，可在「3DGS 训练」查看进度。";
    else if (isVggt) reconstructionNote = service?.vggtError || (!service ? "启动本地服务后可使用 VGGT。" : "");
    else if (service && !canReconstruct) reconstructionNote = "本地服务尚未找到 COLMAP。";
    else if (canReconstruct && !colmapGpu) reconstructionNote = "当前 COLMAP 使用 CPU，3DGS 训练可单独使用 GPU。";

    let trainingNote = "";
    if (activeKind === "reconstruction") trainingNote = "等待当前重建任务结束。";
    else if (linkState === "preview") trainingNote = "启动本地服务后即可使用 NVIDIA GPU 训练。";
    else if (linkState === "pending" || (service && !canTrain && !service.trainingError && !requiresColmap)) trainingNote = "正在验证 CUDA 训练环境。";
    else if (service?.trainingError) trainingNote = service.trainingError;
    else if (requiresColmap && !canReconstruct) trainingNote = "此来源需要本地 COLMAP 来准备训练数据。";

    // Keep errors from a rejected task visible until the form is edited.
    if (!$("form-status").dataset.rejected) setFormStatus($("form-status"), reconstructionNote);
    if (!$("training-status").dataset.rejected) setFormStatus($("training-status"), trainingNote);
  }

  function renderProgress() {
    const reconstruction = jobByKind.reconstruction;
    const vggtProgress = reconstruction ? reconstruction.engine === "vggt" : $("reconstruction-engine").value === "vggt";
    $("steps").hidden = vggtProgress;
    $("vggt-progress").hidden = !vggtProgress;
    if (vggtProgress) {
      $("vggt-progress").textContent = reconstruction?.state === "succeeded" ? "VGGT 重建完成"
        : reconstruction?.state === "failed" ? "VGGT 重建失败，请查看运行日志。"
        : reconstruction?.state === "running" ? "VGGT 正在重建，可查看运行日志。" : "等待开始 VGGT 重建";
    }
    const useDense = reconstruction?.dense ?? $("dense").checked;
    const current = steps.findIndex((step) => step.dataset.stage === reconstruction?.stage);
    steps.forEach((step, index) => {
      let state = "waiting";
      if (step.dataset.stage === "dense" && !useDense) state = "skipped";
      else if (reconstruction?.state === "succeeded" || index < current) state = "done";
      else if (index === current || (current < 0 && index === 0 && reconstruction)) {
        state = reconstruction.state === "failed" ? "failed" : "active";
      }
      step.dataset.state = state;
      step.querySelector(".step-status").textContent = stepLabels[state];
    });

    const training = jobByKind.training;
    const progress = training?.progress;
    const progressLabel = $("training-progress-status");
    progressLabel.dataset.state = training?.state || "idle";
    const stageLabels = { prepare: "准备训练数据", training: "GPU 训练中", preview: "生成渲染预览" };
    progressLabel.textContent = training?.state === "succeeded" ? "训练完成"
      : training?.state === "failed" ? "训练失败"
      : training ? stageLabels[training.stage] || "正在启动" : "等待开始";
    const bar = $("training-progress");
    bar.max = progress?.total || 1;
    bar.value = progress?.iteration || 0;
    $("training-progress-detail").textContent = progress
      ? `${numbers.format(progress.iteration)} / ${numbers.format(progress.total)} 次迭代` : "— / — 次迭代";
    $("training-result").hidden = !(training?.state === "succeeded" && training.outputPath);
    $("training-result").textContent = training?.outputPath ? `模型目录 ${training.outputPath}` : "";
  }

  function renderLogs(job) {
    const log = $("log-output");
    const lines = job?.logs?.slice() || [];
    if (job?.error && !lines.includes(job.error)) lines.push(job.error);
    log.hidden = lines.length === 0;
    $("empty-log").hidden = lines.length > 0;
    if (!lines.length) $("empty-log-text").textContent = job?.state === "running" ? "正在启动任务" : "等待任务开始";
    const next = lines.join("\n");
    if (log.textContent !== next) {
      const nearEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
      log.textContent = next;
      if (nearEnd) log.scrollTop = log.scrollHeight;
    }
  }

  async function renderModel(job, previous = false) {
    const url = job?.model?.url || null;
    const views = job?.views || [];
    const endpoint = job?.interactive && interactiveViewer ? `/api/jobs/${job.id}/render` : null;
    const viewKey = [...views.map((view) => view.url), endpoint || ""].join("\n");
    const trainingLabel = previous ? "上次结果" : "训练结果";
    if (activeModelUrl === url && activeViewKey === viewKey) {
      if (url && job?.kind === "training") $("model-badge").textContent = trainingLabel;
      return;
    }
    activeModelUrl = url;
    activeViewKey = viewKey;
    interactiveEndpoint = endpoint;
    interactiveViewer?.setEndpoint(interactiveEndpoint);
    activeViews = views;
    activeViewIndex = 0;
    if (endpoint && (location.hash === "#training-3d" || !views.length)) previewMode = "interactive";
    else previewMode = views.length ? "image" : "points";
    useReference = false;
    pointData = null;
    focusedPoints = false;
    previewIsTraining = job?.kind === "training";
    previewEngine = job?.engine;
    setPanMode(false);
    renderPreviewMode();
    if (!url) {
      if (!views.length && !endpoint) showDemo();
      return;
    }
    $("model-detail-note").textContent = "正在读取模型";
    try {
      const data = await requestJson(url);
      if (activeModelUrl !== url) return;
      pointData = data;
      selectPointScope(true);
      $("model-badge").textContent = previewIsTraining ? trainingLabel : "重建结果";
      $("model-badge").dataset.live = "true";
      $("point-count").textContent = numbers.format(data.count);
      $("camera-count").textContent = data.cameras == null ? "—" : numbers.format(data.cameras);
    } catch {
      if (activeModelUrl === url) {
        if (!activeViews.length && !interactiveEndpoint) showDemo();
        else {
          $("model-badge").textContent = trainingLabel;
          $("model-badge").dataset.live = "true";
          $("point-count").textContent = numbers.format(job.model.points);
          $("camera-count").textContent = job.model.cameras == null ? "—" : numbers.format(job.model.cameras);
        }
        $("model-detail-note").textContent = "暂时无法读取采样点云";
      }
    }
  }

  function render() {
    renderConnection();
    renderGpuOptions();
    renderForms();
    renderProgress();
    renderCommand();
    const job = jobByKind[mode];
    renderLogs(job);
    const previous = mode === "training" && !job?.views?.length && service?.trainingResult?.id !== job?.id
      ? service?.trainingResult : null;
    void renderModel(previous || job, !!previous);
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `本地服务返回 ${response.status}`);
    return data;
  }

  function receiveStatus(next) {
    service = next;
    if (next.jobs) Object.assign(jobByKind, next.jobs);
    if (next.job) jobByKind[next.job.kind || "reconstruction"] = next.job;
    linkState = next.available || next.trainingAvailable || next.vggtAvailable ? "online" : "missing";
    render();
  }

  async function poll() {
    const revision = statusRevision;
    try {
      const next = await requestJson("/api/status", { signal: AbortSignal.timeout(4500) });
      if (revision === statusRevision && !submitting) receiveStatus(next);
    } catch {
      if (revision === statusRevision && !submitting) {
        service = null;
        linkState = "preview";
        render();
      }
    } finally {
      setTimeout(poll, service?.job?.state === "running" ? 850 : 2300);
    }
  }

  async function submit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const isTraining = form === trainingForm;
    const notice = isTraining ? $("training-status") : $("form-status");
    const payload = formOptions(form);
    if (payload.engine === "vggt" && !service?.vggtAvailable) return;
    if (payload.engine === "colmap" && !service?.available) {
      selectTab("command", true);
      $("command-panel").scrollIntoView({ block: "nearest" });
      return;
    }
    if (isTraining && !service?.trainingAvailable) return;
    if (!form.reportValidity()) return;

    delete notice.dataset.rejected;
    submitting = true;
    statusRevision++;
    selectTab("logs");
    render();
    try {
      receiveStatus(await requestJson(isTraining ? "/api/training" : "/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }));
    } catch (error) {
      notice.dataset.rejected = "true";
      setFormStatus(notice, error.message, true);
    } finally {
      submitting = false;
      render();
    }
  }

  for (const button of modes) button.addEventListener("click", () => setMode(button.dataset.mode));
  window.addEventListener("hashchange", () => {
    const nextMode = modeFromAddress();
    if (nextMode !== mode) setMode(nextMode);
    else if (mode === "training" && interactiveEndpoint && location.hash === "#training-3d") {
      previewMode = "interactive";
      renderPreviewMode();
    }
  });
  for (const tab of tabs) {
    tab.addEventListener("click", () => selectTab(tab.id === "logs-tab" ? "logs" : "command"));
    tab.addEventListener("keydown", (event) => {
      if (mode === "training" || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const useLogs = event.key === "Home" || (event.key !== "End" && tab.id === "command-tab");
      selectTab(useLogs ? "logs" : "command", true);
    });
  }

  for (const [form, notice] of [[reconstructionForm, $("form-status")], [trainingForm, $("training-status")]]) {
    form.addEventListener("submit", submit);
    form.addEventListener("input", () => {
      delete notice.dataset.rejected;
      renderForms();
      renderCommand();
      renderProgress();
    });
  }

  $("use-last-reconstruction").addEventListener("click", () => {
    const previous = service?.lastReconstruction;
    if (!previous) return;
    $("source-type").value = previous.sourceType || "colmap";
    $("training-source").value = previous.workspacePath;
    $("training-image-path").value = previous.imagePath;
    delete $("training-status").dataset.rejected;
    renderForms();
  });

  $("copy-command").addEventListener("click", async () => {
    clearTimeout(copyTimeout);
    try {
      await navigator.clipboard.writeText($("command-output").textContent);
      $("copy-label").textContent = "已复制";
    } catch {
      const selection = getSelection();
      const range = document.createRange();
      range.selectNodeContents($("command-output"));
      selection.removeAllRanges();
      selection.addRange(range);
      $("copy-label").textContent = "已选中，请复制";
    }
    copyTimeout = setTimeout(() => { $("copy-label").textContent = "复制命令"; }, 2200);
  });

  function activeCamera() {
    return previewMode === "interactive" && interactiveViewer ? interactiveViewer : viewer;
  }

  $("reset-view").addEventListener("click", () => activeCamera().reset());
  $("camera-zoom-in").addEventListener("click", () => activeCamera().zoomBy(1.16));
  $("camera-zoom-out").addEventListener("click", () => activeCamera().zoomBy(1 / 1.16));
  $("camera-pan").addEventListener("click", () => setPanMode(!panMode));
  $("point-focus").addEventListener("click", () => selectPointScope(true));
  $("point-entire").addEventListener("click", () => selectPointScope(false));
  $("preview-interactive").addEventListener("click", () => {
    previewMode = "interactive";
    if (mode === "training") rememberView("training-3d");
    renderPreviewMode();
  });
  $("preview-render").addEventListener("click", () => {
    previewMode = "image";
    if (mode === "training") rememberView("training");
    renderPreviewMode();
  });
  $("preview-points").addEventListener("click", () => {
    previewMode = "points";
    if (mode === "training") rememberView("training");
    renderPreviewMode();
  });
  for (const [button, delta] of [["render-previous", -1], ["render-next", 1]]) {
    $(button).addEventListener("click", () => {
      if (activeViews.length < 2) return;
      activeViewIndex = (activeViewIndex + delta + activeViews.length) % activeViews.length;
      renderPreviewMode();
    });
  }
  $("render-reference").addEventListener("click", () => {
    useReference = !useReference;
    renderPreviewMode();
  });
  $("render-image").addEventListener("error", () => {
    $("render-image").hidden = true;
    $("render-error").hidden = false;
  });
  $("fullscreen").disabled = !document.fullscreenEnabled;
  $("fullscreen").addEventListener("click", () => $("viewport").requestFullscreen().catch(() => {}));
  $("exit-fullscreen").addEventListener("click", () => document.exitFullscreen().catch(() => {}));
  document.addEventListener("fullscreenchange", () => {
    $("fullscreen").setAttribute("aria-pressed", String(!!document.fullscreenElement));
  });

  showDemo();
  setMode(modeFromAddress());
  if (linkState !== "preview") void poll();
})();
