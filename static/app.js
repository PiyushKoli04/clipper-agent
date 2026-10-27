/* ShortsAgent frontend - vanilla JS, no external dependencies. */
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const res = await fetch(path, {
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch (_) { /* non-JSON error */ }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
};

/** Tiny DOM builder: h("div", {class:"x", onclick:fn}, child, "text") */
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
}

const ICONS = {
  bolt: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>`,
  download: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>`,
  refresh: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>`,
  film: `<svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line></svg>`,
};

let OPTS = null;               // /api/options
let jobTimer = null;
let clipTimer = null;
let currentVideo = null;
const cards = new Map();       // clip id -> {el, refs, clip}
let libraryVideos = [];
let selectedLibraryVideoId = null;
let currentExistingVideo = null;
let checkTimeout = null;

/* ------------------------------------------------------------------ options */
const FIELDS = ["num_clips", "layout", "overlay_mode", "hook_style", "caption_style", "whisper_model",
  "llm_provider", "hook_duration", "music"];
const CHECKS = ["loudnorm", "zoom", "silence_trim", "progress_bar"];

function fillSelect(sel, map, value) {
  sel.replaceChildren(...Object.entries(map).map(([k, label]) => h("option", { value: k }, label)));
  if (value) sel.value = value;
}

async function loadOptions() {
  OPTS = await api("/api/options");
  fillSelect($("hook_style"), OPTS.hook_styles, OPTS.defaults.hook_style);
  fillSelect($("caption_style"), OPTS.caption_styles, OPTS.defaults.caption_style);
  refreshMusic();
  renderSwatches();

  try { // restore last settings
    const saved = JSON.parse(localStorage.getItem("sa_settings") || "{}");
    for (const f of FIELDS) {
      const el = $(f);
      if (!el || saved[f] == null) continue;
      if (el.tagName === "SELECT") {
        if ([...el.options].some((o) => o.value === String(saved[f]))) el.value = saved[f];
      } else el.value = saved[f];
    }
    for (const c of CHECKS) if ($(c) && saved[c] != null) $(c).checked = !!saved[c];
  } catch (_) { /* ignore */ }

  if (document.fonts?.load) document.fonts.load('100px "HookFont"').catch(() => {});
  updateHardcodedPreview();
}

function refreshMusic(selected) {
  const sel = $("music");
  sel.replaceChildren(h("option", { value: "" }, "None"), ...OPTS.music.map((m) => h("option", { value: m }, m)));
  sel.value = selected || "";
}

$("music_file")?.addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const res = await fetch("/api/music", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    OPTS = await api("/api/options");
    refreshMusic(data.name);
  } catch (err) { showJobError(err.message); }
});

function formParams() {
  const p = { url: $("url").value.trim() };
  p.num_clips = parseInt($("num_clips").value, 10) || 3;
  for (const f of ["layout", "overlay_mode", "hook_style", "caption_style", "whisper_model", "llm_provider"]) p[f] = $(f).value;
  p.hook_duration = parseFloat($("hook_duration").value) || 3;
  for (const c of CHECKS) p[c] = $(c).checked;
  p.music = $("music").value || null;
  return p;
}

/* ------------------------------------------------------------------ Existing Video Check */
$("url")?.addEventListener("input", (e) => {
  clearTimeout(checkTimeout);
  const val = e.target.value.trim();
  if (!val || val.length < 15) {
    hideExistingBanner();
    return;
  }
  checkTimeout = setTimeout(() => checkUrlForExisting(val), 400);
});

async function checkUrlForExisting(url) {
  try {
    const res = await api(`/api/videos/check?url=${encodeURIComponent(url)}`);
    if (res.exists && res.clip_count > 0) {
      currentExistingVideo = res;
      showExistingBanner(res);
    } else {
      hideExistingBanner();
    }
  } catch (_) {
    hideExistingBanner();
  }
}

function showExistingBanner(info) {
  const banner = $("existing-alert");
  if (!banner) return;
  banner.hidden = false;
  $("existing-title").textContent = `This video already has ${info.clip_count} clips generated!`;
  $("existing-count-msg").textContent = `${info.done_count || info.clip_count} rendered clips ready for "${info.title}". View them or generate fresh non-duplicate clips.`;

  $("btn-view-existing").onclick = () => {
    currentVideo = info.video_id;
    localStorage.setItem("sa_video", info.video_id);
    refreshClips(info.video_id);
    $("grid").scrollIntoView({ behavior: "smooth" });
  };

  $("btn-force-new").onclick = () => {
    startGeneration(true);
  };
}

function hideExistingBanner() {
  const banner = $("existing-alert");
  if (banner) banner.hidden = true;
  currentExistingVideo = null;
}

/* ------------------------------------------------------------------ Hardcoded Live Style Preview */
function renderSwatches() {
  const capContainer = $("caption-swatches");
  if (capContainer && OPTS?.caption_styles) {
    capContainer.replaceChildren();
    const dots = {
      white_yellow: "#ffe11a",
      white_clean: "#ffffff",
      white_green: "#2cff6b",
      white_pink: "#ff4fa3",
    };
    for (const [key, label] of Object.entries(OPTS.caption_styles)) {
      const pill = h("button", {
        type: "button",
        class: "swatch-pill",
        "data-for": "caption",
        "data-val": key,
        onclick: () => {
          $("caption_style").value = key;
          updateHardcodedPreview();
        }
      }, h("span", { class: "swatch-dot", style: `background:${dots[key] || "#fff"}` }), label);
      capContainer.append(pill);
    }
  }

  const hookContainer = $("hook-swatches");
  if (hookContainer && OPTS?.hook_styles) {
    hookContainer.replaceChildren();
    for (const [key, label] of Object.entries(OPTS.hook_styles)) {
      const pill = h("button", {
        type: "button",
        class: "swatch-pill",
        "data-for": "hook",
        "data-val": key,
        onclick: () => {
          $("hook_style").value = key;
          updateHardcodedPreview();
        }
      }, label);
      hookContainer.append(pill);
    }
  }
}

function updateHardcodedPreview() {
  const layout = $("layout")?.value || "blur";
  const overlayMode = $("overlay_mode")?.value || "both";
  const hookStyle = $("hook_style")?.value || "bold_outline";
  const captionStyle = $("caption_style")?.value || "white_yellow";
  const progressBar = $("progress_bar")?.checked;
  const customHook = $("preview-test-hook")?.value.trim() || "WAIT FOR THE ENDING!";

  // 1. Stage layout (Blur background vs Smart crop)
  const stage = $("hardcoded-stage");
  if (stage) {
    stage.className = `stage-container layout-${layout}`;
  }

  // 2. Hardcoded Hook preview
  const hookWrap = $("pv-hook-wrap");
  if (hookWrap) {
    hookWrap.hidden = (overlayMode === "captions");
    hookWrap.className = `pv-hook-box hk-${hookStyle}`;
    const words = customHook.toUpperCase().split(/\s+/).filter(Boolean);
    const mid = Math.ceil(words.length / 2);
    const line1 = words.slice(0, mid).join(" ");
    const line2 = words.slice(mid).join(" ");
    hookWrap.replaceChildren(
      h("div", { class: "pv-hook-content" },
        line1 ? h("span", { class: "pv-hook-line" }, line1) : null,
        line2 ? h("span", { class: "pv-hook-line" }, line2) : null
      )
    );
  }

  // 3. Hardcoded Caption preview
  const capWrap = $("pv-caption-wrap");
  if (capWrap) {
    capWrap.hidden = (overlayMode === "hook");
    capWrap.className = `pv-caption-box cap-${captionStyle}`;
  }

  // 4. Progress bar preview
  const pb = $("pv-progress-bar");
  if (pb) {
    pb.hidden = !progressBar;
  }

  // 5. Active state on swatch pills
  document.querySelectorAll(".swatch-pill[data-for='caption']").forEach((el) => {
    el.classList.toggle("active", el.dataset.val === captionStyle);
  });
  document.querySelectorAll(".swatch-pill[data-for='hook']").forEach((el) => {
    el.classList.toggle("active", el.dataset.val === hookStyle);
  });
}

// Hook up preview event listeners
for (const id of ["layout", "overlay_mode", "hook_style", "caption_style", "progress_bar"]) {
  $(id)?.addEventListener("change", updateHardcodedPreview);
}
$("preview-test-hook")?.addEventListener("input", updateHardcodedPreview);

/* ------------------------------------------------------------------ job flow */
function showJobError(msg) {
  const el = $("job-error");
  el.hidden = !msg;
  el.textContent = msg || "";
  if (msg) $("progress").hidden = false;
}

function setProgress(stage, pct) {
  $("progress").hidden = false;
  $("stage").textContent = stage;
  $("pct").textContent = `${Math.round(pct)}%`;
  $("bar-fill").style.width = `${pct}%`;
  document.querySelector(".bar").setAttribute("aria-valuenow", Math.round(pct));
}

$("gen-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  // If the video already has clips, ask to generate fresh clips
  const shouldForceNew = Boolean(currentExistingVideo && currentExistingVideo.clip_count > 0);
  startGeneration(shouldForceNew);
});

async function startGeneration(forceNew = false) {
  const params = formParams();
  if (forceNew) params.force_new = true;
  localStorage.setItem("sa_settings", JSON.stringify({ ...params, music: params.music }));
  showJobError("");
  $("go").disabled = true;
  hideExistingBanner();
  try {
    const { job_id, video_id } = await api("/api/jobs", { method: "POST", body: params });
    if (!forceNew) clearGrid();
    currentVideo = video_id;
    localStorage.setItem("sa_video", video_id);
    setProgress(forceNew ? "Generating fresh highlights..." : "Queued", 0);
    pollJob(job_id);
    updateLibraryCount();
  } catch (err) {
    showJobError(err.message);
    $("go").disabled = false;
  }
}

function pollJob(jobId) {
  clearInterval(jobTimer);
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${jobId}`);
      setProgress(j.stage || j.status, j.progress || 0);
      if (j.video_id) await refreshClips(j.video_id);
      if (j.status === "done" || j.status === "failed") {
        clearInterval(jobTimer);
        $("go").disabled = false;
        if (j.status === "failed") {
          showJobError(j.error || "The job failed.");
          showRetryButton(jobId);
        } else {
          updateLibraryCount();
        }
      }
    } catch (err) {
      clearInterval(jobTimer);
      $("go").disabled = false;
      showJobError(err.message);
    }
  };
  tick();
  jobTimer = setInterval(tick, 2000);
}

function showRetryButton(jobId) {
  const existing = document.getElementById("retry-btn");
  if (existing) existing.remove();
  const btn = h("button", {
    id: "retry-btn",
    class: "primary",
    type: "button",
    onclick: () => retryJob(jobId),
    style: "margin-top:10px; display:inline-flex; align-items:center; gap:6px;"
  });
  btn.innerHTML = `${ICONS.refresh} <span>Retry (resumes from cache)</span>`;
  $("job-error").insertAdjacentElement("afterend", btn);
}

async function retryJob(jobId) {
  const retryBtn = document.getElementById("retry-btn");
  if (retryBtn) retryBtn.disabled = true;
  showJobError("");
  if (retryBtn) retryBtn.remove();
  $("go").disabled = true;
  try {
    const { job_id, video_id } = await api(`/api/jobs/${jobId}/retry`, { method: "POST" });
    if (video_id) {
      currentVideo = video_id;
      localStorage.setItem("sa_video", video_id);
    }
    setProgress("Retrying (resuming from cache)...", 0);
    pollJob(job_id);
  } catch (err) {
    showJobError(err.message);
    $("go").disabled = false;
  }
}

/* ------------------------------------------------------------------ clips */
function clearGrid() {
  $("grid").replaceChildren();
  cards.clear();
  $("video-title").hidden = true;
}

async function refreshClips(videoId) {
  const data = await api(`/api/videos/${videoId}/clips`);
  $("video-title").hidden = false;
  $("video-title").textContent = data.video.title || videoId;
  for (const clip of data.clips) upsertCard(clip);
  const busy = data.clips.some((c) => c.status === "rendering" || c.status === "pending");
  if (busy && !clipTimer && !jobTimer) clipTimer = setInterval(() => refreshClips(videoId).catch(() => {}), 2000);
  if (!busy && clipTimer) { clearInterval(clipTimer); clipTimer = null; }
}

function setStatus(ref, text, kind = "") {
  ref.className = `status ${kind}`;
  ref.textContent = text;
}

function upsertCard(clip, targetGrid = null) {
  const gridEl = targetGrid || $("grid");
  let card = cards.get(clip.id);
  if (!card) {
    card = buildCard(clip);
    cards.set(clip.id, card);
    gridEl.append(card.el);
  }
  card.clip = clip;
  const r = card.refs;
  r.score.textContent = clip.score ? clip.score.toFixed(0) : "-";
  r.title.textContent = clip.title || `Clip ${clip.id}`;
  r.meta.textContent = `${fmt(clip.start)} - ${fmt(clip.end)} (${(clip.end - clip.start).toFixed(0)}s)` + (clip.reason ? ` · ${clip.reason}` : "");
  if (clip.url && r.video.dataset.src !== clip.url) {
    r.video.dataset.src = clip.url;
    r.video.src = clip.url;
    r.ph.hidden = true;
    r.video.hidden = false;
  }
  r.dl.classList.toggle("off", !clip.download_url);
  r.dl.href = clip.download_url || "#";
  if (clip.status === "rendering" || clip.status === "pending") {
    if (!clip.url) { r.ph.hidden = false; r.ph.textContent = clip.status === "pending" ? "Queued..." : "Rendering..."; }
    setStatus(r.status, "Rendering...", "busy");
    r.rerender.disabled = true;
  } else if (clip.status === "failed") {
    setStatus(r.status, `Failed: ${clip.error || "unknown error"}`, "bad");
    if (!clip.url) r.ph.textContent = "Render failed";
    r.rerender.disabled = false;
  } else {
    if (r.status.classList.contains("busy")) setStatus(r.status, "Done", "ok");
    r.rerender.disabled = false;
  }
  updateCardHookPreview(card);
}

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function buildCard(clip) {
  const refs = {};
  const style = clip.style || {};
  refs.score = h("b");
  refs.ph = h("div", { class: "ph" }, "Queued...");
  refs.video = h("video", { controls: true, playsinline: true, preload: "metadata", hidden: true });
  refs.preview = h("div", { class: "hook-pv", "aria-hidden": "true" });
  refs.title = h("h3", { class: "card-title" });
  refs.meta = h("div", { class: "meta" });

  refs.dl = h("a", { class: "btn off", href: "#", download: "" });
  refs.dl.innerHTML = `${ICONS.download} <span>Download MP4</span>`;

  refs.status = h("div", { class: "status" });

  // ---- editor
  const variants = clip.hooks || [];
  refs.hookSel = h("select", { "aria-label": "Hook variant" },
    ...variants.map((v, i) => h("option", { value: v }, `${i + 1}. ${v}`)), h("option", { value: "__custom" }, "Custom..."));
  refs.hookCustom = h("input", { type: "text", maxlength: "90", placeholder: "Type custom hook", "aria-label": "Custom hook" });
  if (variants.includes(clip.hook_text)) refs.hookSel.value = clip.hook_text;
  else { refs.hookSel.value = "__custom"; refs.hookCustom.value = clip.hook_text || ""; }
  refs.mode = h("select", { "aria-label": "Overlay mode" },
    h("option", { value: "hook" }, "Hook only"), h("option", { value: "captions" }, "Captions"), h("option", { value: "both" }, "Both"));
  refs.mode.value = clip.overlay_mode;
  refs.hookStyle = h("select", { "aria-label": "Hook style" });
  fillSelect(refs.hookStyle, OPTS.hook_styles, style.hook_style || "bold_outline");
  refs.capStyle = h("select", { "aria-label": "Caption style" });
  fillSelect(refs.capStyle, OPTS.caption_styles, style.caption_style || "white_yellow");
  refs.layout = h("select", { "aria-label": "Layout" },
    h("option", { value: "blur" }, "Blur background"), h("option", { value: "smart" }, "Smart crop"));
  refs.layout.value = style.layout || "blur";
  refs.start = h("input", { type: "number", step: "0.1", min: "0", value: clip.start.toFixed(1), "aria-label": "Start seconds" });
  refs.end = h("input", { type: "number", step: "0.1", min: "0", value: clip.end.toFixed(1), "aria-label": "End seconds" });

  refs.rerender = h("button", { class: "rerender", type: "button" });
  refs.rerender.innerHTML = `${ICONS.bolt} <span>Re-render</span>`;

  const lab = (t, el, wide) => h("label", { class: wide ? "wide" : "" }, t, el);
  const editor = h("details", { class: "editor" },
    h("summary", {}, "Edit Clip Settings"),
    h("div", { class: "fields" },
      lab("Hook variant", refs.hookSel, true), lab("Custom hook", refs.hookCustom, true),
      lab("Overlay", refs.mode), lab("Layout", refs.layout),
      lab("Hook style", refs.hookStyle), lab("Caption style", refs.capStyle),
      lab("Start (s)", refs.start), lab("End (s)", refs.end),
      h("div", { class: "wide row" }, refs.rerender)));

  const el = h("article", { class: "card" },
    h("div", { class: "player" }, h("div", { class: "badge" }, "Score ", refs.score), refs.ph, refs.video, refs.preview),
    h("div", { class: "card-body" }, refs.title, refs.meta, h("div", { class: "row" }, refs.dl), refs.status, editor));
  const card = { el, refs, clip };

  const live = () => updateCardHookPreview(card);
  refs.hookSel.addEventListener("change", () => {
    refs.hookCustom.hidden = refs.hookSel.value !== "__custom";
    live();
  });
  refs.hookCustom.hidden = refs.hookSel.value !== "__custom";
  for (const x of [refs.hookCustom, refs.mode, refs.hookStyle]) x.addEventListener("input", live);
  refs.video.addEventListener("timeupdate", () => toggleCardPreview(card));
  refs.video.addEventListener("seeked", () => toggleCardPreview(card));
  refs.video.addEventListener("pause", () => toggleCardPreview(card));
  new ResizeObserver(live).observe(el.querySelector(".player"));
  refs.rerender.addEventListener("click", () => rerender(card));
  return card;
}

function currentHook(card) {
  const r = card.refs;
  return r.hookSel.value === "__custom" ? r.hookCustom.value.trim() : r.hookSel.value;
}

async function rerender(card) {
  const r = card.refs, id = card.clip.id;
  const start = parseFloat(r.start.value), end = parseFloat(r.end.value);
  if (!(end > start)) return setStatus(r.status, "End must be after start", "bad");
  r.rerender.disabled = true;
  setStatus(r.status, "Saving...", "busy");
  try {
    await api(`/api/clips/${id}`, {
      method: "PATCH",
      body: { start, end, hook_text: currentHook(card), overlay_mode: r.mode.value,
        style: { hook_style: r.hookStyle.value, caption_style: r.capStyle.value, layout: r.layout.value } },
    });
    await api(`/api/clips/${id}/render`, { method: "POST" });
    setStatus(r.status, "Rendering...", "busy");
    if (!clipTimer) clipTimer = setInterval(() => refreshClips(card.clip.video_id).catch(() => {}), 2000);
  } catch (err) {
    setStatus(r.status, err.message, "bad");
    r.rerender.disabled = false;
  }
}

/* ------------------------------------------------------------------ card video hook preview */
const cv = document.createElement("canvas").getContext("2d");
function textW(text, fs) {
  const em = fs * (OPTS?.em_per_fs || 0.83);
  cv.font = `${em}px HookFont, Anton, Impact, sans-serif`;
  return cv.measureText(text).width;
}

function wrapBalanced(words, fs, maxW) {
  const greedy = (limit) => {
    const lines = []; let cur = "";
    for (const w of words) {
      const t = cur ? `${cur} ${w}` : w;
      if (cur && textW(t, fs) > limit) { lines.push(cur); cur = w; } else cur = t;
    }
    if (cur) lines.push(cur);
    return lines;
  };
  const n = greedy(maxW).length;
  if (n <= 1) return greedy(maxW);
  let lo = Math.max(...words.map((w) => textW(w, fs))), hi = maxW;
  for (let i = 0; i < 14; i++) { const mid = (lo + hi) / 2; if (greedy(mid).length <= n) hi = mid; else lo = mid; }
  return greedy(hi);
}

function fitHook(text, padX) {
  const words = text.toUpperCase().split(/\s+/).filter(Boolean);
  const avail = 0.85 * 1080 - 2 * padX;
  for (let fs = 150; fs >= 44; fs -= 4) {
    const lines = wrapBalanced(words, fs, avail);
    if (lines.length <= 3 && Math.max(...lines.map((l) => textW(l, fs))) <= avail) return { lines, fs };
  }
  return { lines: wrapBalanced(words, 44, avail).slice(0, 3), fs: 44 };
}

const PAD_X = { bold_outline: 0, yellow_box: 30, rounded_dark: 48, gradient_pill: 56 };

function updateCardHookPreview(card) {
  const { refs } = card;
  const mode = refs.mode.value, text = currentHook(card);
  const box = refs.video.parentElement;
  const k = box.clientWidth / 1080;
  if (!text || mode === "captions" || !OPTS) { refs.preview.replaceChildren(); return; }
  const styleKey = refs.hookStyle.value;
  const { lines, fs } = fitHook(text, PAD_X[styleKey] ?? 0);
  box.style.setProperty("--k", k);
  refs.preview.className = `hook-pv hk-${styleKey}`;
  refs.preview.style.setProperty("--fs", fs);
  refs.preview.style.fontSize = `${fs * (OPTS.em_per_fs || 0.83) * k}px`;
  refs.preview.style.lineHeight = `${fs * k}px`;
  const lineEls = lines.map((l) => h("span", { class: "ln" }, l));
  const block = styleKey === "yellow_box" || styleKey === "bold_outline" ? lineEls : [h("div", { class: "blk" }, lineEls)];
  refs.preview.replaceChildren(...block);
  toggleCardPreview(card);
}

function toggleCardPreview(card) {
  const v = card.refs.video;
  const dur = parseFloat(card.clip.style?.hook_duration) || 3;
  card.refs.preview.classList.toggle("off", v.currentTime > dur);
}

/* ------------------------------------------------------------------ Persistent Clips Library */
function setupTabs() {
  $("tab-gen")?.addEventListener("click", () => switchTab("generator"));
  $("tab-library")?.addEventListener("click", () => {
    switchTab("library");
    loadLibrary();
  });
  $("refresh-library-btn")?.addEventListener("click", loadLibrary);
}

function switchTab(name) {
  $("tab-gen")?.classList.toggle("active", name === "generator");
  $("tab-library")?.classList.toggle("active", name === "library");
  const viewGen = $("view-generator");
  const viewLib = $("view-library");
  if (viewGen) {
    viewGen.classList.toggle("active", name === "generator");
    viewGen.hidden = (name !== "generator");
    viewGen.style.display = (name === "generator" ? "block" : "none");
  }
  if (viewLib) {
    viewLib.classList.toggle("active", name === "library");
    viewLib.hidden = (name !== "library");
    viewLib.style.display = (name === "library" ? "block" : "none");
  }
}

async function updateLibraryCount() {
  try {
    const { videos } = await api("/api/history");
    const count = videos ? videos.reduce((acc, v) => acc + (v.clip_count || 0), 0) : 0;
    if ($("library-count")) $("library-count").textContent = count;
  } catch (_) { /* ignore */ }
}

async function loadLibrary() {
  const container = $("library-grid");
  const chipsContainer = $("library-videos");
  if (!container) return;

  container.innerHTML = `<div class="library-empty"><p>Loading saved clips...</p></div>`;
  try {
    const { videos } = await api("/api/history");
    libraryVideos = videos || [];
    updateLibraryCount();

    if (!chipsContainer) return;
    chipsContainer.replaceChildren();

    if (libraryVideos.length === 0) {
      container.innerHTML = `
        <div class="library-empty">
          <p>No clips generated yet. Paste a YouTube URL in the Generator to create your first Short!</p>
        </div>
      `;
      return;
    }

    const activeId = selectedLibraryVideoId || (currentVideo || libraryVideos[0].id);
    for (const v of libraryVideos) {
      const chip = h("div", {
        class: `video-chip ${v.id === activeId ? "active" : ""}`,
        onclick: () => selectLibraryVideo(v.id),
      },
        h("span", { class: "video-chip-icon" }),
        h("div", { class: "video-chip-info" },
          h("span", { class: "video-chip-title" }, v.title || v.id),
          h("span", { class: "video-chip-count" }, `${v.done_count || 0}/${v.clip_count || 0} clips ready`)
        )
      );
      chip.querySelector(".video-chip-icon").innerHTML = ICONS.film;
      chipsContainer.append(chip);
    }

    selectLibraryVideo(activeId);
  } catch (err) {
    container.innerHTML = `<div class="library-empty"><p class="error">Failed to load library: ${err.message}</p></div>`;
  }
}

async function selectLibraryVideo(videoId) {
  selectedLibraryVideoId = videoId;
  document.querySelectorAll(".video-chip").forEach((c, i) => {
    c.classList.toggle("active", libraryVideos[i]?.id === videoId);
  });

  const container = $("library-grid");
  if (!container) return;
  container.replaceChildren();

  try {
    const data = await api(`/api/videos/${videoId}/clips`);
    if (!data.clips || data.clips.length === 0) {
      container.innerHTML = `<div class="library-empty"><p>No clips found for this video.</p></div>`;
      return;
    }
    for (const clip of data.clips) {
      upsertCard(clip, container);
    }
  } catch (err) {
    container.innerHTML = `<div class="library-empty"><p class="error">Error: ${err.message}</p></div>`;
  }
}

/* ------------------------------------------------------------------ boot */
(async function init() {
  try {
    setupTabs();
    await loadOptions();
    await updateLibraryCount();
    const last = localStorage.getItem("sa_video");
    if (last) {
      currentVideo = last;
      await refreshClips(last).catch(() => localStorage.removeItem("sa_video"));
    }
  } catch (err) {
    showJobError(`Could not initialize: ${err.message}`);
  }
})();
