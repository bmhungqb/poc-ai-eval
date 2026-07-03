"""Score-matrix heatmap (PNG) and HTML timeline visualization."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

STATUS_COLORS = {
    "MATCHED": "#4caf50",
    "EXTRA": "#e53935",
    "LOW_CONFIDENCE": "#ff9800",
    "UNMATCHED": "#795548",
    "MISSING": "#9e9e9e",
    "WRONG_ORDER": "#8e24aa",
    "UNKNOWN": "#607d8b",
}

AUX_VERDICT_COLORS = {"present": "#4caf50", "absent": "#e53935", "uncertain": "#ff9800"}


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def build_frame_labels(worker_df, report: dict):
    """One row per worker frame: which decoded segment (if any) covers its
    timestamp, carrying the segment's matched scene/operations/status/
    confidence. Frames outside every segment's [start, end) (small gaps at
    segment boundaries) get status UNKNOWN. `report["segments"]` is already
    sorted by worker_start (segments are built by walking the path in
    order), so a single searchsorted over segment starts locates the
    covering segment per frame in O(log n) instead of a per-frame scan."""
    import pandas as pd

    segments = report["segments"]
    times = worker_df["time"].to_numpy()
    frames = worker_df["frame"].to_numpy() if "frame" in worker_df.columns else np.arange(len(worker_df))

    if not segments:
        starts = ends = np.array([])
    else:
        starts = np.array([s["worker_start"] for s in segments])
        ends = np.array([s["worker_end"] for s in segments])

    rows = []
    idxs = np.searchsorted(starts, times, side="right") - 1 if len(starts) else np.full(len(times), -1)
    for i, t in enumerate(times):
        si = int(idxs[i])
        seg = segments[si] if 0 <= si < len(segments) and starts[si] <= t < ends[si] + 1e-6 else None
        rows.append({
            "frame": int(frames[i]),
            "time": round(float(t), 3),
            "expert_scene_index": seg["matched_expert_scene_index"] if seg else None,
            "operations": " + ".join(seg["operations"]) if seg else "",
            "status": seg["status"] if seg else "UNKNOWN",
            "timing_status": seg["timing_status"] if seg else "",
            "confidence": seg["confidence"] if seg else 0.0,
        })
    return pd.DataFrame(rows)


def save_worker_frame_labels(labels_df, worker_video_path: str, out_dir: str | Path,
                             video_name: str = "worker_labeled.mp4", log=lambda *_a: None):
    """Write per-frame labels to CSV, plus (if the source video is
    reachable) an annotated copy of the worker video with the current
    scene/operation/status burned into each frame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(out_dir / "worker_frame_labels.csv", index=False)

    if not worker_video_path or not Path(worker_video_path).exists():
        log(f"skipping labeled video -- worker video not found ({worker_video_path!r})")
        return

    import cv2

    cap = cv2.VideoCapture(str(worker_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / video_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    labels = labels_df.to_dict("records")
    n = len(labels)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= n:
            break
        row = labels[i]
        color = _hex_to_bgr(STATUS_COLORS.get(row["status"], STATUS_COLORS["UNKNOWN"]))
        scene = f"E{row['expert_scene_index']}" if row["expert_scene_index"] is not None else "-"
        line1 = f"t={row['time']:.2f}s  scene={scene}  {row['status']}"
        line2 = row["operations"] or "(none)"
        line3 = f"conf={row['confidence']:.2f} {row['timing_status']}".strip()
        cv2.rectangle(frame, (0, 0), (width, 70), (0, 0, 0), -1)
        cv2.putText(frame, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, line3, (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    log(f"wrote labeled video ({i} frames) to {out_dir / video_name}")


def save_score_matrix(score_matrix: np.ndarray, out_dir: str | Path, scene_labels: list[str],
                      name: str = "score_matrix", title: str = "Worker steps vs expert scenes"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    np.save(out_dir / f"{name}.npy", score_matrix)
    fig, ax = plt.subplots(figsize=(10, max(6, score_matrix.shape[0] / 40)))
    im = ax.imshow(score_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("expert scene index")
    ax.set_ylabel("worker observation step")
    ax.set_xticks(range(len(scene_labels)))
    ax.set_xticklabels([str(i) for i in range(len(scene_labels))])
    fig.colorbar(im, label="similarity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def save_debug_signal(values: np.ndarray, out_dir: str | Path, name: str, ylabel: str,
                      step_times: list[float] | None = None):
    """Save a 1-D per-step signal (e.g. extra_emission, motion) as .npy + line plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    np.save(out_dir / f"{name}.npy", values)
    x = step_times if step_times is not None else range(len(values))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, values, linewidth=1)
    ax.set_xlabel("time (s)" if step_times is not None else "worker observation step")
    ax.set_ylabel(ylabel)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def save_frame_correspondence_plot(df, out_dir: str | Path, name: str = "frame_correspondence"):
    """Scatter of worker time vs. the nearest-neighbor-matched expert frame's
    time, one point per worker frame, colored by assigned scene. `df` has
    columns worker_time, expert_time_nn, scene (see
    src.matching.similarity.frame_correspondence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(9, 6))
    scenes = sorted(df["scene"].unique())
    cmap = plt.get_cmap("tab20")
    for i, sc in enumerate(scenes):
        sub = df[df["scene"] == sc]
        ax.scatter(sub["worker_time"], sub["expert_time_nn"], s=6, color=cmap(i % 20), label=f"E{sc}")
    ax.set_xlabel("worker time (s)")
    ax.set_ylabel("nearest-neighbor-matched expert frame time (s)")
    ax.set_title("Per-frame correspondence (pose/flow nearest-neighbor query)")
    ax.legend(fontsize=7, ncol=2, markerscale=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def save_frame_overlays(pairs: list[dict], expert_video_path: str, worker_video_path: str,
                        out_dir: str | Path, log=lambda *_a: None):
    """For each {worker_frame, worker_time, expert_frame, expert_time, scene}
    pair, read both raw frames and write a side-by-side PNG so a match can be
    eyeballed directly. Silently skipped if either video file is unavailable
    (e.g. re-run from a machine that doesn't have the source videos)."""
    import cv2

    if not expert_video_path or not worker_video_path or \
       not Path(expert_video_path).exists() or not Path(worker_video_path).exists():
        log(f"skipping frame overlays -- video file(s) not found "
            f"(expert={expert_video_path!r}, worker={worker_video_path!r})")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ecap = cv2.VideoCapture(str(expert_video_path))
    wcap = cv2.VideoCapture(str(worker_video_path))

    def _read(cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = cap.read()
        return frame if ok else None

    written = 0
    for i, p in enumerate(pairs):
        wf, ef = _read(wcap, p["worker_frame"]), _read(ecap, p["expert_frame"])
        if wf is None or ef is None:
            continue
        target_h = 240
        wf = cv2.resize(wf, (int(wf.shape[1] * target_h / wf.shape[0]), target_h))
        ef = cv2.resize(ef, (int(ef.shape[1] * target_h / ef.shape[0]), target_h))
        for img, label in ((wf, f"worker t={p['worker_time']:.2f}s"),
                           (ef, f"expert E{p['scene']} t={p['expert_time']:.2f}s")):
            cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        combo = cv2.hconcat([wf, ef])
        cv2.imwrite(str(out_dir / f"{i:03d}_seg_E{p['scene']}_wf{p['worker_frame']}.png"), combo)
        written += 1

    ecap.release()
    wcap.release()
    log(f"wrote {written} frame overlay images to {out_dir}")


def save_debug_path(path: list[int], step_times: list[float], out_dir: str | Path,
                    name: str = "viterbi_path"):
    """Save the decoded Viterbi state path (scene index per step, -1 = EXTRA)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    arr = np.array(path)
    np.save(out_dir / f"{name}.npy", arr)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(step_times, arr, where="post", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("expert scene index (-1 = EXTRA)")
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def _bar(items, total_dur, row_label):
    cells = []
    for it in items:
        left = 100 * it["start"] / total_dur
        w = max(0.3, 100 * (it["end"] - it["start"]) / total_dur)
        color = it["color"]
        tip = html.escape(it["tip"])
        label = html.escape(it["label"])
        cells.append(
            f'<div class="seg" style="left:{left:.2f}%;width:{w:.2f}%;background:{color}" title="{tip}">{label}</div>')
    return (f'<div class="rowlabel">{html.escape(row_label)}</div>'
            f'<div class="bar">{"".join(cells)}</div>')


def save_timeline_html(report: dict, scenes, out_dir: str | Path):
    out_dir = Path(out_dir)
    expert_total = max(sc.end for sc in scenes)
    worker_total = max((s["worker_end"] for s in report["segments"]), default=1.0)
    missing_idx = {e["expert_scene_index"] for e in report["errors"] if e["type"] == "MISSING"}
    wrong_idx = {e.get("expert_scene_index") for e in report["errors"] if e["type"] == "WRONG_ORDER"}

    expert_items = []
    for sc in scenes:
        color = STATUS_COLORS["MISSING"] if sc.scene_index in missing_idx else "#2196f3"
        expert_items.append({
            "start": sc.start, "end": sc.end, "color": color,
            "label": str(sc.scene_index),
            "tip": f"E{sc.scene_index} {sc.start:.1f}-{sc.end:.1f}s: {sc.label}"
                   + (" [MISSING in worker]" if sc.scene_index in missing_idx else "")})

    worker_items = []
    for s in report["segments"]:
        idx = s["matched_expert_scene_index"]
        if idx is None:
            color, lbl = STATUS_COLORS["EXTRA"], "X"
        elif idx in wrong_idx:
            color, lbl = STATUS_COLORS["WRONG_ORDER"], str(idx)
        else:
            color, lbl = STATUS_COLORS[s["status"]], str(idx)
        worker_items.append({
            "start": s["worker_start"], "end": s["worker_end"], "color": color, "label": lbl,
            "tip": (f"{s['worker_start']:.1f}-{s['worker_end']:.1f}s -> "
                    f"{'E' + str(idx) if idx is not None else 'EXTRA'} "
                    f"{' + '.join(s['operations'])} conf={s['confidence']} "
                    f"{s['status']} {s['timing_status']}")})

    summary = report["summary"]
    err_rows = "".join(
        f"<tr><td>{html.escape(e['type'])}</td><td>{html.escape(e['message'])}</td></tr>"
        for e in report["errors"]) or "<tr><td colspan=2>No errors detected</td></tr>"

    aux = report.get("aux_checklist") or []
    aux_rows = "".join(
        f"<tr><td>E{a['scene_index']}</td><td>{html.escape(a['scene_label'])}</td>"
        f"<td>{html.escape(a['operation'])}</td>"
        f"<td style='color:{AUX_VERDICT_COLORS.get(a['verdict'], '#607d8b')};font-weight:600'>"
        f"{html.escape(a['verdict'])}</td>"
        f"<td>{a['confidence']:.2f}</td>"
        f"<td>{('%.1f-%.1fs' % tuple(a['worker_time'])) if a.get('worker_time') else '—'}</td>"
        f"<td>{html.escape(a.get('evidence') or '')}</td></tr>"
        for a in aux)
    aux_section = ("" if not aux else f"""
<h2 style="font-size:16px">Aux-operation checklist</h2>
<table><tr><th>Scene</th><th>Scene label</th><th>Aux operation</th><th>Verdict</th>
<th>Conf</th><th>Worker time</th><th>Evidence</th></tr>{aux_rows}</table>""")

    legend = "".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>{n.lower()}</span>'
        for n, c in {**STATUS_COLORS, "expert scene": "#2196f3"}.items())

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Alignment: {html.escape(report['task_name'])}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:90px 1fr;gap:6px;align-items:center}}
.bar{{position:relative;height:38px;background:#eee;border-radius:4px;overflow:hidden}}
.seg{{position:absolute;top:0;height:100%;color:#fff;font-size:11px;display:flex;align-items:center;
justify-content:center;border-right:1px solid rgba(255,255,255,.6);box-sizing:border-box;overflow:hidden}}
.rowlabel{{font-size:13px;text-align:right;padding-right:6px}}
.lg{{margin-right:14px;font-size:12px}} .sw{{display:inline-block;width:12px;height:12px;border-radius:2px;
margin-right:4px;vertical-align:-2px}}
table{{border-collapse:collapse;margin-top:16px}} td,th{{border:1px solid #ccc;padding:6px 10px;font-size:13px}}
.kpis span{{display:inline-block;margin-right:18px;font-size:14px}}
</style></head><body>
<h1>Alignment report — {html.escape(report['task_name'])}</h1>
<div class="kpis">
<span><b>Scenes expected:</b> {summary['num_expected_scenes']}</span>
<span><b>Segments detected:</b> {summary['num_detected_segments']}</span>
<span><b>Missing:</b> {summary['missing_count']}</span>
<span><b>Extra:</b> {summary['extra_count']}</span>
<span><b>Wrong order:</b> {summary['wrong_order_count']}</span>
<span><b>Duplicated:</b> {summary['duplicated_count']}</span>
<span><b>Unmatched:</b> {summary.get('unmatched_count', 0)}</span>
<span><b>Overall score:</b> {summary['overall_score']}</span>
<span><b>Emission:</b> {html.escape(report.get('emission_source', 'pose_flow'))}</span>
</div>
<p>{legend}</p>
<div class="grid">
{_bar(expert_items, expert_total, f"Expert ({expert_total:.0f}s)")}
{_bar(worker_items, worker_total, f"Worker ({worker_total:.0f}s)")}
</div>
<p style="font-size:12px;color:#666">Hover segments for details. Numbers are expert scene indices; X = extra action.</p>
<h2 style="font-size:16px">Errors</h2>
<table><tr><th>Type</th><th>Detail</th></tr>{err_rows}</table>
{aux_section}
<h2 style="font-size:16px">Score matrix</h2>
<img src="score_matrix.png" style="max-width:100%">
</body></html>"""
    (out_dir / "alignment_timeline.html").write_text(page, encoding="utf-8")
