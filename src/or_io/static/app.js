(() => {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const videoView = $("#videoView");
  const canvas = $("#roiCanvas");
  const ctx = canvas.getContext("2d");
  const hint = $("#viewportHint");
  const statusPill = $("#statusPill");

  const state = {
    connected: false,
    monitoring: false,
    hasSnapshot: false,
    width: 0,
    height: 0,
    layer: "outside",
    outside: [],
    inside: [],
    previewTimer: null,
    eventTimer: null,
    frozen: false,
  };

  function setStep(n) {
    $$(".step").forEach((b) => b.classList.toggle("active", b.dataset.step === String(n)));
    $$(".step-pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === String(n)));
  }

  $$(".step").forEach((btn) => {
    btn.addEventListener("click", () => setStep(btn.dataset.step));
  });

  $$('input[name="roiLayer"]').forEach((el) => {
    el.addEventListener("change", () => {
      if (el.checked) state.layer = el.value;
    });
  });

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : detail || data.message || res.statusText;
      throw new Error(msg);
    }
    return data;
  }

  function applyState(s) {
    state.connected = !!s.connected;
    state.monitoring = !!s.monitoring;
    state.hasSnapshot = !!s.has_snapshot;
    state.width = s.width || state.width;
    state.height = s.height || state.height;
    statusPill.textContent = s.status || "—";
    $("#sourceMeta").textContent = `源：${s.source || "—"}`;
    $("#sizeMeta").textContent = s.width ? `${s.width}×${s.height}` : "—";
    $("#modeMeta").textContent = s.monitoring ? "模式：实时监测" : state.frozen ? "模式：截图标注" : "模式：预览";
    $("#statEnter").textContent = String(s.enters ?? s.stats?.enters ?? 0);
    $("#statExit").textContent = String(s.exits ?? s.stats?.exits ?? 0);
    $("#statFrames").textContent = String(s.stats?.frames ?? 0);
    hint.classList.toggle("hidden", state.connected);
  }

  function contentRect() {
    const rect = videoView.getBoundingClientRect();
    const nw = state.width || videoView.naturalWidth || 1;
    const nh = state.height || videoView.naturalHeight || 1;
    const scale = Math.min(rect.width / nw, rect.height / nh);
    const dw = nw * scale;
    const dh = nh * scale;
    const left = rect.left + (rect.width - dw) / 2;
    const top = rect.top + (rect.height - dh) / 2;
    return { left, top, width: dw, height: dh, scale, nw, nh };
  }

  function resizeCanvas() {
    const wrap = $("#canvasWrap").getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(wrap.width * devicePixelRatio));
    canvas.height = Math.max(1, Math.floor(wrap.height * devicePixelRatio));
    canvas.style.width = `${wrap.width}px`;
    canvas.style.height = `${wrap.height}px`;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    drawRoi();
  }

  function drawPoly(points, color, label) {
    if (!points.length) return;
    const cr = contentRect();
    const wrap = $("#canvasWrap").getBoundingClientRect();
    const toXY = (p) => [
      cr.left - wrap.left + p[0] * cr.width,
      cr.top - wrap.top + p[1] * cr.height,
    ];
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = toXY(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (points.length >= 3) {
      ctx.closePath();
      ctx.save();
      ctx.globalAlpha = 0.18;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    points.forEach((p) => {
      const [x, y] = toXY(p);
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fill();
    });
    if (points[0]) {
      const [x, y] = toXY(points[0]);
      ctx.fillStyle = "#e8f2ef";
      ctx.font = "600 12px Instrument Sans, sans-serif";
      ctx.fillText(label, x + 8, y - 8);
    }
  }

  function drawRoi() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawPoly(state.outside, "#3db8ff", "门外");
    drawPoly(state.inside, "#3ddea2", "门内");
    $("#roiInfo").textContent = `门外 ${state.outside.length} 点 · 门内 ${state.inside.length} 点`;
  }

  function eventToNorm(evt) {
    const cr = contentRect();
    const wrap = $("#canvasWrap").getBoundingClientRect();
    const x = evt.clientX;
    const y = evt.clientY;
    if (x < cr.left || y < cr.top || x > cr.left + cr.width || y > cr.top + cr.height) return null;
    return [
      Math.min(1, Math.max(0, (x - cr.left) / cr.width)),
      Math.min(1, Math.max(0, (y - cr.top) / cr.height)),
    ];
  }

  canvas.addEventListener("click", (evt) => {
    if (!state.frozen && !state.hasSnapshot) {
      statusPill.textContent = "请先截图再标注";
      return;
    }
    const p = eventToNorm(evt);
    if (!p) return;
    state[state.layer].push(p);
    drawRoi();
  });

  function refreshFrame(which = "auto") {
    const w = which === "auto" ? (state.monitoring ? "monitor" : state.frozen ? "snapshot" : "live") : which;
    const bust = Date.now();
    videoView.src = `/api/frame.jpg?which=${w}&q=78&t=${bust}`;
  }

  function startPreviewLoop() {
    stopPreviewLoop();
    state.previewTimer = setInterval(() => {
      if (state.frozen && !state.monitoring) return;
      refreshFrame("auto");
    }, state.monitoring ? 450 : 350);
  }

  function stopPreviewLoop() {
    if (state.previewTimer) clearInterval(state.previewTimer);
    state.previewTimer = null;
  }

  async function refreshEvents() {
    try {
      const data = await api("/api/events?limit=50");
      const list = $("#eventList");
      if (!data.events.length) {
        list.innerHTML = `<li class="empty">尚无入/出室事件</li>`;
        return;
      }
      list.innerHTML = data.events
        .map((e) => {
          const when = e.local_time || e.wall_time_iso || "";
          const label = e.event === "enter" ? "入室" : "出室";
          return `<li>
            <span class="tag ${e.event}">${label}</span>
            <span class="when">${when}</span>
            <span class="meta">track ${e.track_id} · frame ${e.frame_idx} · conf ${(e.confidence || 0).toFixed(2)}</span>
          </li>`;
        })
        .join("");
      $("#statEnter").textContent = String(data.enters);
      $("#statExit").textContent = String(data.exits);
    } catch (_) {
      /* ignore */
    }
  }

  function startEventLoop() {
    stopEventLoop();
    state.eventTimer = setInterval(async () => {
      const s = await api("/api/state");
      applyState(s);
      if (s.monitoring || s.event_count) refreshEvents();
    }, 1200);
  }

  function stopEventLoop() {
    if (state.eventTimer) clearInterval(state.eventTimer);
    state.eventTimer = null;
  }

  $("#btnConnect").addEventListener("click", async () => {
    const source = $("#sourceInput").value.trim();
    if (!source) return alert("请填写视频源");
    try {
      statusPill.textContent = "连接中…";
      const data = await api("/api/connect", { method: "POST", body: JSON.stringify({ source }) });
      applyState(data);
      state.frozen = false;
      refreshFrame("live");
      startPreviewLoop();
      startEventLoop();
      setStep(2);
    } catch (e) {
      alert(e.message);
      statusPill.textContent = e.message;
    }
  });

  $("#btnDisconnect").addEventListener("click", async () => {
    await api("/api/disconnect", { method: "POST", body: "{}" });
    stopPreviewLoop();
    stopEventLoop();
    state.frozen = false;
    videoView.removeAttribute("src");
    applyState({ connected: false, status: "未连接" });
    hint.classList.remove("hidden");
  });

  $("#btnSnapshot").addEventListener("click", async () => {
    try {
      const data = await api("/api/snapshot", { method: "POST", body: "{}" });
      applyState(data);
      state.frozen = true;
      refreshFrame("snapshot");
      statusPill.textContent = "已截图：请标注门外/门内 ROI";
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btnClearRoi").addEventListener("click", () => {
    state.outside = [];
    state.inside = [];
    drawRoi();
  });

  $("#btnSaveRoi").addEventListener("click", async () => {
    if (state.outside.length < 3 || state.inside.length < 3) {
      alert("门外与门内均需至少 3 个点");
      return;
    }
    try {
      const data = await api("/api/roi", {
        method: "POST",
        body: JSON.stringify({
          outside: state.outside,
          inside: state.inside,
          site_name: $("#siteName").value.trim() || "or_door",
          already_normalized: true,
          save: true,
        }),
      });
      applyState(data);
      statusPill.textContent = `ROI 已保存 ${data.roi_path || ""}`;
      setStep(3);
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btnStart").addEventListener("click", async () => {
    try {
      statusPill.textContent = "启动监测（首次加载模型可能较慢）…";
      const data = await api("/api/monitor/start", {
        method: "POST",
        body: JSON.stringify({ device: $("#deviceSelect").value }),
      });
      applyState(data);
      state.frozen = false;
      startPreviewLoop();
      refreshEvents();
    } catch (e) {
      alert(e.message);
      statusPill.textContent = e.message;
    }
  });

  $("#btnStop").addEventListener("click", async () => {
    const data = await api("/api/monitor/stop", { method: "POST", body: "{}" });
    applyState(data);
    refreshEvents();
  });

  $("#btnRefreshEvents").addEventListener("click", refreshEvents);

  window.addEventListener("resize", resizeCanvas);
  videoView.addEventListener("load", resizeCanvas);
  resizeCanvas();

  // restore ROI points if server already has them
  api("/api/state")
    .then(async (s) => {
      applyState(s);
      if (s.connected) {
        startPreviewLoop();
        startEventLoop();
        refreshFrame("auto");
      }
      if (s.has_roi) {
        try {
          const roi = await api("/api/roi");
          state.outside = roi.roi.zone.rois.outside.polygon;
          state.inside = roi.roi.zone.rois.inside.polygon;
          drawRoi();
        } catch (_) {}
      }
    })
    .catch(() => {});
})();
