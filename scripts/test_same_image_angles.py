#!/usr/bin/env python3
"""
Same source image → many synthetic viewpoints (rotate / flip), POST /face/get-face-embedding each time.
Requires: fiass running (default http://127.0.0.1:18089), enrolled face for that user.
Usage:
  ./scripts/test_same_image_angles.py [path/to/image.png]
"""
from __future__ import annotations

import io
import json
import os
import sys

import requests
from PIL import Image

BASE = os.environ.get("FACE_TEST_BASE", "http://127.0.0.1:18089").rstrip("/")
DEFAULT_IMG = os.path.join(
    os.path.dirname(__file__), "..", "..", "test-face-e2e.png"
)


def post_embedding(pil_img: Image.Image, label: str) -> dict:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    url = f"{BASE}/face/get-face-embedding?fast=1"
    r = requests.post(
        url,
        files={"image": ("probe.png", buf.getvalue(), "image/png")},
        timeout=120,
    )
    try:
        data = r.json()
    except Exception:
        return {
            "label": label,
            "http": r.status_code,
            "error": r.text[:200],
        }
    row = {
        "label": label,
        "http": r.status_code,
        "status": data.get("status"),
        "message": data.get("message"),
    }
    m = data.get("match") or {}
    if m:
        row["user_id"] = m.get("user_id")
        row["similarity"] = m.get("similarity")
        row["confidence"] = m.get("confidence")
    row["appearance_variation"] = data.get("appearance_variation")
    return row


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    base = Image.open(path).convert("RGB")
    w, h = base.size

    cases: list[tuple[str, Image.Image]] = []

    def add(name: str, im: Image.Image) -> None:
        cases.append((name, im))

    add("original (0°)", base)
    for deg in (-35, -25, -15, -10, 10, 15, 25, 35):
        add(f"rotate {deg:+.0f}° (expand)", base.rotate(-deg, expand=True, fillcolor=(128, 128, 128)))
    for deg in (90, 180, 270):
        add(f"rotate {deg}°", base.rotate(-deg, expand=True))
    add("flip horizontal", base.transpose(Image.FLIP_LEFT_RIGHT))
    add("flip vertical", base.transpose(Image.FLIP_TOP_BOTTOM))
    add("flip H + V (=180° portrait crop style)", base.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM))

    rows = []
    for label, im in cases:
        rows.append(post_embedding(im, label))

    # Pretty table
    print(f"FACE_TEST_BASE={BASE}")
    print(f"source_image={path}")
    print()
    hdr = f"{'viewpoint / transform':<42} | {'HTTP':>4} | {'match?':>6} | {'user_id':<28} | {'sim':>7} | notes"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        uid = row.get("user_id") or "—"
        sim = row.get("similarity")
        sims = f"{sim:.4f}" if isinstance(sim, (int, float)) else "—"
        ok = "yes" if row.get("status") is True else "no"
        msg = (row.get("message") or "")[:40]
        print(
            f"{row['label']:<42} | {row.get('http', '—'):>4} | {ok:>6} | {str(uid):<28} | {sims:>7} | {msg}"
        )

    out_json = os.path.join(os.path.dirname(path), "angle-test-results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print()
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
