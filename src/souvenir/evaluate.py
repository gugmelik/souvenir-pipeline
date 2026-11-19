"""Batch evaluation: run several modes over a folder of real photos and compare them by group size.

Writes results.csv (one row per photo x mode), summary.md (pass rates by number of people),
and index.html (side-by-side contact sheet for human review).
"""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path
from statistics import mean

from souvenir.pipeline import Pipeline
from souvenir.schemas import RunReport

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(src: str) -> list[str]:
    p = Path(src)
    if p.is_dir():
        return [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in IMAGE_EXT]
    if p.suffix.lower() == ".txt":  # one path or URL per line
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [src]


def _row(r: RunReport) -> dict:
    final = next((v for v in r.verify if v.target in ("hero", "baseline")), None)
    checks = {c.name: c for c in (final.checks if final else [])}
    ids = [c.score for n, c in checks.items() if n.startswith("identity_rider_") and c.score is not None]
    count = checks.get("rider_count")
    return {
        "run_id": r.run_id,
        "mode": r.mode,
        "source": r.source,
        "n_people": r.n_people,
        "status": r.status,
        "passed": bool(final and final.passed),
        "count_ok": None if not count else count.passed,
        "identity_min": round(min(ids), 3) if ids else None,
        "identity_mean": round(mean(ids), 3) if ids else None,
        "checklist": checks.get("vlm_checklist").score if checks.get("vlm_checklist") else None,
        "calls": r.backend_calls,
        "seconds": r.seconds_total,
        "output": r.outputs.get("hero") or r.outputs.get("baseline") or "",
        "warnings": " | ".join(r.warnings)[:500],
        "error": r.error,
    }


def run_eval(pipeline: Pipeline, images: list[str], modes: list[str], out_root: Path, brief: str | None, seed: int = 0) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, src in enumerate(images):
        for mode in modes:
            r = pipeline.run(src, out_root / "runs", mode=mode, seed=seed + i, brief=brief)
            rows.append(_row(r))
            print(f"[{i + 1}/{len(images)}] {mode:<14} n={r.n_people} status={r.status} calls={r.backend_calls} {r.seconds_total}s")

    csv_path = out_root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        w.writeheader()
        w.writerows(rows)
    _summary(rows, out_root / "summary.md")
    _contact_sheet(rows, images, modes, out_root / "index.html")
    return csv_path


def _summary(rows: list[dict], path: Path) -> None:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["n_people"])].append(r)
    lines = ["# Evaluation summary", "", "| mode | people | photos | pass rate | count ok | identity min (mean) | calls/photo | sec/photo |",
             "|---|---|---|---|---|---|---|---|"]
    for (mode, n), rs in sorted(groups.items()):
        passed = sum(1 for r in rs if r["passed"]) / len(rs)
        count_ok = [r["count_ok"] for r in rs if r["count_ok"] is not None]
        idm = [r["identity_min"] for r in rs if r["identity_min"] is not None]
        idmean = [r["identity_mean"] for r in rs if r["identity_mean"] is not None]
        count_txt = f"{sum(count_ok) / len(count_ok):.0%}" if count_ok else "n/a"
        id_txt = f"{mean(idm):.3f} ({mean(idmean):.3f})" if idm else "n/a"
        calls = mean(r["calls"] for r in rs)
        secs = mean(r["seconds"] for r in rs)
        lines.append(f"| {mode} | {n} | {len(rs)} | {passed:.0%} | {count_txt} | {id_txt} | {calls:.1f} | {secs:.1f} |")
    lines += ["", "Pass = every check on the final image passed (count, identity per rider, checklist, OCR).",
              "Undecidable checks (e.g. no readable face on stylized art) do not fail a photo; review them in index.html."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _contact_sheet(rows: list[dict], images: list[str], modes: list[str], path: Path) -> None:
    by_src: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_src[r["source"]][r["mode"]] = r
    head = "".join(f"<th>{html.escape(m)}</th>" for m in modes)
    body = []
    for src in images:
        cells = [f'<td><img src="{html.escape(_uri(src))}"><br>source</td>']
        for m in modes:
            r = by_src[src].get(m)
            if not r:
                cells.append("<td>-</td>")
                continue
            img = f'<img src="{html.escape(_uri(r["output"]))}">' if r["output"] else "(no output)"
            mark = "PASS" if r["passed"] else r["status"]
            cells.append(f"<td>{img}<br><b>{mark}</b> n={r['n_people']} id_min={r['identity_min']}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>souvenir eval</title>"
        "<style>body{font-family:sans-serif}td{vertical-align:top;padding:6px;font-size:12px}img{width:260px}</style>"
        f"<table><tr><th>source</th>{head}</tr>{''.join(body)}</table>",
        encoding="utf-8",
    )


def _uri(p: str) -> str:
    if p.startswith("http"):
        return p
    return Path(p).resolve().as_uri() if p else ""
