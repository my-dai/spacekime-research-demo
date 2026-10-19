#!/usr/bin/env python3
"""Local browser viewer for Spacekime-inspired fMRI phase experiments."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spacekime_core import (
    build_phase_distribution_surface,
    complex_phase_metrics,
    kime_distance_L2,
    kime_js_distance,
    kime_surface_summary,
    make_phase_noisy_complex_slice,
    metrics_psnr_ssim,
    phase_scope_mask,
    roi_center_box,
    snr_mean_over_std,
)
from spacekime_loader import bigim_metadata, load_bigim
from spacekime_paths import resolve_mat_path


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spacekime Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6f80;
      --line: #cfd8df;
      --panel: #f7f9fb;
      --accent: #1b6ca8;
      --accent-strong: #10527f;
      --warn: #9b3412;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #eef3f6;
      font-family: Segoe UI, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 14px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: calc(100vh - 57px);
    }
    aside {
      padding: 14px;
      background: #ffffff;
      border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    section {
      padding: 14px;
      min-width: 0;
    }
    .group {
      border-top: 1px solid var(--line);
      padding: 13px 0;
    }
    .group:first-child { border-top: 0; padding-top: 0; }
    .group h2 {
      margin: 0 0 10px;
      font-size: 13px;
      font-weight: 650;
      color: #263848;
    }
    label {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      margin: 10px 0 4px;
      color: #32485a;
      font-weight: 600;
    }
    input[type="range"] { width: 100%; }
    input[type="number"], select {
      width: 100%;
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 4px 7px;
      background: #ffffff;
      color: var(--ink);
    }
    button {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--accent);
      border-radius: 4px;
      background: var(--accent);
      color: #ffffff;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button.secondary {
      background: #ffffff;
      color: var(--accent);
    }
    button.secondary:hover { background: #eef7fd; }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr);
      gap: 14px;
      align-items: start;
    }
    .panel {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 650;
    }
    .plot {
      width: 100%;
      min-height: 260px;
      border: 1px solid #d8e0e6;
      background: var(--panel);
      object-fit: contain;
    }
    .plot.tall { min-height: 360px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      min-width: 0;
    }
    .metric .name {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric .value {
      margin-top: 2px;
      font-size: 17px;
      font-weight: 650;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .error {
      color: var(--warn);
      font-weight: 650;
      margin-top: 8px;
      white-space: pre-wrap;
    }
    code {
      background: #eef3f6;
      border: 1px solid #d8e0e6;
      border-radius: 4px;
      padding: 1px 4px;
    }
    @media (max-width: 980px) {
      main, .workspace { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Spacekime fMRI Viewer</h1>
    <div class="status" id="status">Loading</div>
  </header>
  <main>
    <aside>
      <div class="group">
        <h2>Slice</h2>
        <label for="t">t <span id="tLabel">0</span></label>
        <input id="t" type="range" min="0" max="0" value="0">
        <label for="z">z <span id="zLabel">10</span></label>
        <input id="z" type="range" min="0" max="0" value="10">
        <label for="sigma">sigma <span id="sigmaLabel">2.00</span></label>
        <input id="sigma" type="range" min="0" max="4" step="0.05" value="2">
        <label for="noise">noise</label>
        <select id="noise">
          <option value="gaussian">gaussian</option>
          <option value="vm">von Mises</option>
        </select>
        <label for="seed">seed</label>
        <input id="seed" type="number" value="42" step="1">
        <label for="roiHalf">ROI half</label>
        <input id="roiHalf" type="number" value="8" min="1" step="1">
        <div class="actions">
          <button id="sliceBtn">Render slice</button>
          <button id="resetBtn" class="secondary">Reset</button>
        </div>
      </div>
      <div class="group">
        <h2>Phase-distribution proxy</h2>
        <label for="scope">scope</label>
        <select id="scope">
          <option value="slice">selected slice</option>
          <option value="roi">central ROI</option>
          <option value="volume">whole volume</option>
        </select>
        <label for="phaseMode">phase mode</label>
        <select id="phaseMode">
          <option value="raw">raw observed phase</option>
          <option value="relative">relative to selected t</option>
        </select>
        <label for="maxT">max t</label>
        <input id="maxT" type="number" value="50" min="2" step="1">
        <label for="maskPercentile">mask percentile</label>
        <input id="maskPercentile" type="number" value="50" min="0" max="99" step="1">
        <label for="nTheta">theta bins</label>
        <input id="nTheta" type="number" value="128" min="16" step="16">
        <label for="normalizeH">normalize H</label>
        <select id="normalizeH">
          <option value="1">yes</option>
          <option value="0">no</option>
        </select>
        <button id="kimeBtn">Analyze phase proxy</button>
        <div class="error" id="error"></div>
      </div>
    </aside>
    <section>
      <div class="workspace">
        <div class="panel">
          <h2>Slice comparison</h2>
          <img id="slicePlot" class="plot" alt="Slice comparison">
        </div>
        <div class="panel">
          <h2>Slice metrics</h2>
          <div class="metrics" id="sliceMetrics"></div>
        </div>
      </div>
      <div class="workspace" style="margin-top:14px;">
        <div class="panel">
          <h2>Spatial phase-distribution comparison</h2>
          <img id="kimePlot" class="plot tall" alt="Spatial phase-distribution comparison">
        </div>
        <div class="panel">
          <h2>Phase-proxy metrics</h2>
          <div class="metrics" id="kimeMetrics"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const state = { shape: [0, 0, 0, 0] };

    function setStatus(text) { $("status").textContent = text; }
    function setError(text) { $("error").textContent = text || ""; }
    function value(id) { return $(id).value; }
    function setLabels() {
      $("tLabel").textContent = value("t");
      $("zLabel").textContent = value("z");
      $("sigmaLabel").textContent = Number(value("sigma")).toFixed(2);
    }
    function metrics(target, rows) {
      target.innerHTML = rows.map(([name, val]) => {
        const text = typeof val === "number" ? formatNumber(val) : val;
        return `<div class="metric"><div class="name">${name}</div><div class="value">${text}</div></div>`;
      }).join("");
    }
    function formatNumber(x) {
      if (!Number.isFinite(x)) return "nan";
      if (Math.abs(x) >= 100) return x.toFixed(1);
      if (Math.abs(x) >= 10) return x.toFixed(2);
      if (Math.abs(x) >= 1) return x.toFixed(3);
      return x.toFixed(5);
    }
    function params(extra = {}) {
      const p = new URLSearchParams({
        t: value("t"),
        z: value("z"),
        sigma: value("sigma"),
        noise: value("noise"),
        seed: value("seed"),
        roi_half: value("roiHalf"),
        max_t: value("maxT"),
        mask_percentile: value("maskPercentile"),
        n_theta: value("nTheta"),
        normalize_h: value("normalizeH"),
        scope: value("scope"),
        phase_mode: value("phaseMode"),
        ...extra
      });
      return p.toString();
    }
    async function getJson(url) {
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || response.statusText);
      return data;
    }
    async function loadState() {
      const data = await getJson("/api/state");
      state.shape = data.shape;
      $("t").max = data.shape[3] - 1;
      $("z").max = data.shape[2] - 1;
      $("z").value = Math.min(10, data.shape[2] - 1);
      setLabels();
      setStatus(`${data.shape.join(" x ")} | ${data.dtype}`);
    }
    async function renderSlice() {
      setError("");
      setLabels();
      setStatus("Rendering slice");
      try {
        const data = await getJson(`/api/slice?${params()}`);
        $("slicePlot").src = data.image;
        metrics($("sliceMetrics"), [
          ["PSNR", data.metrics.psnr],
          ["SSIM", data.metrics.ssim],
          ["ROI SNR orig", data.metrics.roi_snr_orig],
          ["ROI SNR noisy", data.metrics.roi_snr_noisy],
          ["ROI delta", data.metrics.roi_snr_delta],
          ["complex NRMSE", data.metrics.complex_nrmse],
          ["magnitude NRMSE", data.metrics.magnitude_nrmse],
          ["phase MAE rad", data.metrics.phase_mae_rad],
          ["phase RMSE rad", data.metrics.phase_rmse_rad],
          ["phase coherence", data.metrics.phase_coherence],
          ["elapsed ms", data.elapsed_ms]
        ]);
        setStatus("Slice ready");
      } catch (err) {
        setError(err.message);
        setStatus("Slice failed");
      }
    }
    async function runKime() {
      setError("");
        setStatus("Analyzing phase distribution");
      try {
        const data = await getJson(`/api/kime?${params()}`);
        $("kimePlot").src = data.image;
        metrics($("kimeMetrics"), [
          ["L2", data.metrics.l2],
          ["L2 row norm", data.metrics.l2_row_normalized],
          ["JS distance", data.metrics.js_distance],
          ["mean R orig", data.metrics.mean_R_orig],
          ["mean R noisy", data.metrics.mean_R_noisy],
          ["mean entropy orig", data.metrics.mean_entropy_orig],
          ["mean entropy noisy", data.metrics.mean_entropy_noisy],
          ["normalized entropy orig", data.metrics.mean_entropy_normalized_orig],
          ["normalized entropy noisy", data.metrics.mean_entropy_normalized_noisy],
          ["paired phase coherence", data.metrics.mean_phase_coherence],
          ["paired complex NRMSE", data.metrics.mean_complex_nrmse],
          ["mask voxels", data.metrics.mask_voxels],
          ["elapsed ms", data.elapsed_ms]
        ]);
        setStatus("Phase proxy ready");
      } catch (err) {
        setError(err.message);
        setStatus("Kime failed");
      }
    }
    function resetControls() {
      $("t").value = 0;
      $("z").value = Math.min(10, state.shape[2] - 1);
      $("sigma").value = 2;
      $("noise").value = "gaussian";
      $("seed").value = 42;
      $("roiHalf").value = 8;
      $("maxT").value = Math.min(50, state.shape[3]);
      $("maskPercentile").value = 50;
      $("nTheta").value = 128;
      $("normalizeH").value = 1;
      $("scope").value = "slice";
      $("phaseMode").value = "raw";
      setLabels();
      renderSlice();
    }
    ["t", "z", "sigma"].forEach(id => $(id).addEventListener("input", setLabels));
    $("sliceBtn").addEventListener("click", renderSlice);
    $("kimeBtn").addEventListener("click", runKime);
    $("resetBtn").addEventListener("click", resetControls);
    loadState().then(renderSlice).catch(err => {
      setError(err.message);
      setStatus("Load failed");
    });
  </script>
</body>
</html>
"""


class ViewerState:
    def __init__(self, mat_path: Path) -> None:
        self.mat_path = mat_path
        self.bigim = load_bigim(mat_path)
        self.meta = bigim_metadata(self.bigim)


class SpacekimeHandler(BaseHTTPRequestHandler):
    state: ViewerState

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
            elif parsed.path == "/api/state":
                self._send_json(self._state_payload())
            elif parsed.path == "/api/slice":
                self._send_json(self._slice_payload(parse_qs(parsed.query)))
            elif parsed.path == "/api/kime":
                self._send_json(self._kime_payload(parse_qs(parsed.query)))
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - server protection
            self._send_json(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _state_payload(self) -> dict:
        return {
            "mat_file": str(self.state.mat_path),
            "shape": list(self.state.bigim.shape),
            "dtype": self.state.meta["dtype"],
            "mag_min": self.state.meta["mag_min"],
            "mag_max": self.state.meta["mag_max"],
            "mag_mean": self.state.meta["mag_mean"],
        }

    def _slice_payload(self, query: dict[str, list[str]]) -> dict:
        started = time.perf_counter()
        bigim = self.state.bigim
        _, _, Z, T = bigim.shape
        t_idx = _clip_int(_param(query, "t", 0), 0, T - 1)
        z_idx = _clip_int(_param(query, "z", 10), 0, Z - 1)
        sigma = max(0.0, _param_float(query, "sigma", 2.0))
        noise = _param_text(query, "noise", "gaussian")
        seed = _param_int(query, "seed", 42)
        roi_half = max(1, _param_int(query, "roi_half", 8))

        orig_complex, noisy_complex = make_phase_noisy_complex_slice(
            bigim[:, :, :, t_idx],
            z_idx=z_idx,
            sigma=sigma,
            noise=noise,
            rng=np.random.default_rng(seed),
        )
        orig = np.real(orig_complex).astype(np.float32)
        noisy = np.real(noisy_complex).astype(np.float32)
        psnr_value, ssim_value = metrics_psnr_ssim(orig, noisy)
        roi_orig = roi_center_box(orig, half=roi_half)
        roi_noisy = roi_center_box(noisy, half=roi_half)
        snr_orig = snr_mean_over_std(roi_orig)
        snr_noisy = snr_mean_over_std(roi_noisy)
        paired = complex_phase_metrics(orig_complex, noisy_complex)

        return {
            "image": _slice_figure(orig, noisy, t_idx=t_idx, z_idx=z_idx, sigma=sigma),
            "metrics": _json_ready(
                {
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "roi_snr_orig": snr_orig,
                    "roi_snr_noisy": snr_noisy,
                    "roi_snr_delta": snr_noisy - snr_orig,
                    **paired,
                }
            ),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def _kime_payload(self, query: dict[str, list[str]]) -> dict:
        started = time.perf_counter()
        bigim = self.state.bigim
        _, _, Z, T = bigim.shape
        t_idx = _clip_int(_param(query, "t", 0), 0, T - 1)
        z_idx = _clip_int(_param(query, "z", 10), 0, Z - 1)
        sigma = max(0.0, _param_float(query, "sigma", 2.0))
        noise = _param_text(query, "noise", "gaussian")
        seed = _param_int(query, "seed", 42)
        max_t = _clip_int(_param(query, "max_t", 50), 2, T)
        n_theta = _clip_int(_param(query, "n_theta", 128), 16, 512)
        mask_percentile = _clip_float(_param_float(query, "mask_percentile", 50.0), 0, 99)
        normalize_h = _param_int(query, "normalize_h", 1) == 1
        scope = _param_text(query, "scope", "slice")
        phase_mode = _param_text(query, "phase_mode", "raw")
        roi_half = max(1, _param_int(query, "roi_half", 8))

        t_indices = list(range(max_t))
        mask = phase_scope_mask(
            bigim,
            t_idx=t_idx,
            percentile=mask_percentile,
            scope=scope,
            z_idx=z_idx,
            roi_half=roi_half,
        )

        stats_orig = build_phase_distribution_surface(
            bigim,
            mask,
            t_list=t_indices,
            n_theta=n_theta,
            normalize_h=normalize_h,
            phase_mode=phase_mode,
            reference_t=t_idx,
        )
        stats_noisy = build_phase_distribution_surface(
            bigim,
            mask,
            t_list=t_indices,
            n_theta=n_theta,
            normalize_h=normalize_h,
            phase_noise_sigma=sigma,
            noise=noise,
            rng=np.random.default_rng(seed),
            phase_mode=phase_mode,
            reference_t=t_idx,
        )

        summary_orig = kime_surface_summary(stats_orig)
        summary_noisy = kime_surface_summary(stats_noisy)
        metrics = {
            "l2": kime_distance_L2(stats_orig, stats_noisy),
            "l2_row_normalized": kime_distance_L2(
                stats_orig,
                stats_noisy,
                normalize_rows=True,
            ),
            "js_distance": kime_js_distance(stats_orig, stats_noisy),
            "mean_R_orig": summary_orig["mean_R"],
            "mean_R_noisy": summary_noisy["mean_R"],
            "mean_kappa_orig": summary_orig["mean_kappa"],
            "mean_kappa_noisy": summary_noisy["mean_kappa"],
            "mean_entropy_orig": summary_orig["mean_entropy"],
            "mean_entropy_noisy": summary_noisy["mean_entropy"],
            "mean_entropy_normalized_orig": summary_orig["mean_entropy_normalized"],
            "mean_entropy_normalized_noisy": summary_noisy["mean_entropy_normalized"],
            "mean_phase_coherence": summary_noisy["mean_phase_coherence"],
            "mean_complex_nrmse": summary_noisy["mean_complex_nrmse"],
            "mask_voxels": int(mask.sum()),
            "scope": scope,
            "phase_mode": phase_mode,
            "z": z_idx,
        }

        return {
            "image": _kime_figure(stats_orig, stats_noisy, sigma=sigma),
            "metrics": _json_ready(metrics),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        payload: dict,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(_json_ready(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _slice_figure(
    orig: np.ndarray,
    noisy: np.ndarray,
    t_idx: int,
    z_idx: int,
    sigma: float,
) -> str:
    diff = noisy - orig
    vmin = float(np.nanpercentile(np.concatenate([orig.ravel(), noisy.ravel()]), 1))
    vmax = float(np.nanpercentile(np.concatenate([orig.ravel(), noisy.ravel()]), 99))
    dmax = float(max(abs(np.nanpercentile(diff, 1)), abs(np.nanpercentile(diff, 99)), 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    for ax, img, title, cmap in (
        (axes[0], orig, f"Original t={t_idx} z={z_idx}", "gray"),
        (axes[1], noisy, f"Noisy sigma={sigma:g}", "gray"),
        (axes[2], diff, "Noisy - original", "coolwarm"),
    ):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if title == "Noisy - original":
            im = ax.imshow(img, cmap=cmap, vmin=-dmax, vmax=dmax)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.78)
    return _figure_to_data_url(fig)


def _kime_figure(stats_orig: dict, stats_noisy: dict, sigma: float) -> str:
    theta = np.asarray(stats_orig["theta"], dtype=np.float32)
    t_axis = np.asarray(stats_orig["t_axis"], dtype=int)
    H_orig = np.asarray(stats_orig["H"], dtype=np.float32)
    H_noisy = np.asarray(stats_noisy["H"], dtype=np.float32)
    R_orig = np.asarray(stats_orig["R_t"], dtype=np.float32)
    R_noisy = np.asarray(stats_noisy["R_t"], dtype=np.float32)
    entropy_orig = np.asarray(stats_orig["entropy_t"], dtype=np.float32)
    entropy_noisy = np.asarray(stats_noisy["entropy_t"], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)
    extent = (int(t_axis[0]), int(t_axis[-1]), float(theta[0]), float(theta[-1]))
    for ax, H, title in (
        (axes[0], H_orig, "Spatial phase H(t, theta) original"),
        (axes[1], H_noisy, f"Spatial phase H(t, theta) noisy sigma={sigma:g}"),
    ):
        im = ax.imshow(H.T, aspect="auto", origin="lower", extent=extent, cmap="viridis")
        ax.set_xlabel("t")
        ax.set_ylabel("theta")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.78)

    axes[2].plot(t_axis, R_orig, label="R orig")
    axes[2].plot(t_axis, R_noisy, label="R noisy")
    axes[2].plot(t_axis, entropy_orig, "--", label="entropy orig")
    axes[2].plot(t_axis, entropy_noisy, "--", label="entropy noisy")
    axes[2].set_xlabel("t")
    axes[2].set_title("Spatial phase response")
    axes[2].grid(True)
    axes[2].legend(fontsize=8)
    return _figure_to_data_url(fig)


def _figure_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _param(query: dict[str, list[str]], key: str, default: int) -> int:
    return _param_int(query, key, default)


def _param_int(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(float(query.get(key, [default])[0]))
    except (TypeError, ValueError):
        return int(default)


def _param_float(query: dict[str, list[str]], key: str, default: float) -> float:
    try:
        return float(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return float(default)


def _param_text(query: dict[str, list[str]], key: str, default: str) -> str:
    value = str(query.get(key, [default])[0]).strip().lower()
    return value or default


def _clip_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _clip_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local Spacekime browser viewer.")
    parser.add_argument(
        "mat_file",
        nargs="?",
        default=None,
        help="Path to .mat (default: $SPACEKIME_MAT_FILE, data/, or project root)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    try:
        mat_path = resolve_mat_path(args.mat_file)
        print(f"Loading: {mat_path}")
        state = ViewerState(mat_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    handler = SpacekimeHandler
    handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Spacekime viewer: {url}")
    print("Press Ctrl+C to stop.")
    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
