# ArcFace / InsightFace 512-d Upgrade — Migration Runbook (P2)

**Status:** Design complete. Code hooks prepared. **Not yet enabled in production.**

## Why this upgrade

Today the service uses `face_recognition` (dlib ResNet, 128-d). On clean lab data
(LFW) it scores ~99.38% accuracy. On real Indian-demographic production traffic
with imperfect lighting, masks, glasses, partial occlusion, our observed
false-accept rate at cosine >= 0.93 is still meaningfully non-zero — which is
why we had to stack:

1. Pose gates (yaw/pitch/roll)
2. Blur/brightness/contrast/landmark gates
3. Margin check between top-1 and top-2
4. Automatic `fast=1 → fast=0` re-verification
5. Strict employee-scope check
6. Claude Vision secondary verification

To cross **99% production accuracy** reliably, we need a stronger encoder.
ArcFace/InsightFace `buffalo_l` (ResNet-100 + ArcFace loss, 512-d) is the
current open-source state of the art:

| Model                        | Dim | LFW   | IJB-C   | ~FAR @ 0.1% FRR |
|------------------------------|-----|-------|---------|-----------------|
| dlib `face_recognition`      | 128 | 99.38 | ~92     | ~1%             |
| InsightFace `buffalo_s`      | 512 | 99.60 | ~95     | ~0.3%           |
| **InsightFace `buffalo_l`**  | 512 | 99.80 | **~97** | **~0.05%**      |

## Why it is **not** a one-line change

ArcFace cosine similarity scales **differently** from dlib. A 0.93 threshold
that works for dlib is nonsense for ArcFace (there, a strong match sits around
0.45–0.55, and thresholds are typically around 0.40). So the entire scoring /
margin / high-confidence stack has to be recalibrated. Additionally:

* The FAISS index has to be rebuilt from 128-d to 512-d.
* **Every existing face descriptor** has to be re-encoded from the original
  source image (cannot be converted in-place).
* Model download (~200–300 MB) and ONNX runtime dependencies need to be
  baked into the Docker image.
* Inference per image goes from ~40ms to ~90ms on CPU — still fine for 50 RPS
  with 2–4 worker instances, but the per-instance budget shifts.

## Migration plan (phases)

### Phase 0 — Prep (no production impact)

1. Add deps to `fiass-face-matching/requirements.txt`:

   ```
   insightface==0.7.3
   onnxruntime==1.17.1       # or onnxruntime-gpu on GPU boxes
   ```

2. Add to Docker build: pre-pull `buffalo_l` ZIP into `/root/.insightface/models/`
   so container cold-start doesn't re-download.

3. In `main.py`, behind `EMBED_BACKEND=arcface`, wire an alternative encoder:

   ```python
   from insightface.app import FaceAnalysis
   _arc_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
   _arc_app.prepare(ctx_id=0, det_size=(640, 640))

   def arcface_encode(img_np, face_location):
       top, right, bottom, left = face_location
       faces = _arc_app.get(img_np)
       # Pick face whose bbox best overlaps face_location.
       ...
       return faces[0].normed_embedding  # 512-d, L2-normalized
   ```

4. Add `ARCFACE_DIM=512`, `ARCFACE_MATCH_THRESHOLD=0.42`,
   `ARCFACE_HIGH_CONFIDENCE=0.50`, `ARCFACE_MARGIN=0.06`.

### Phase 1 — Shadow mode (zero risk)

Run ArcFace **alongside** dlib for every search call. Log both scores to
Redis. Do NOT change the match decision. After 7–14 days we have real
production data to pick thresholds that hit our 99% target on our user base,
not on LFW.

### Phase 2 — Offline re-encode

1. Take a read-only snapshot of the `faces` table.
2. For every row, fetch the **original image** from R2
   (`faceVerification/<imageURL>`).
3. Encode with ArcFace → 512-d vector.
4. Insert into a new `faces_v2` table (`VECTOR(512)` column).
5. Build a fresh FAISS `IndexFlatIP` with dim=512.
6. Verify: for each user, top-1 in `faces_v2` should be themselves.

Rows whose R2 object returned 404 (we already saw this in logs) get **flagged
for re-enrollment** rather than silently losing them.

### Phase 3 — Cutover

1. Deploy a build where `EMBED_BACKEND=arcface` is the default.
2. `/face/search`, `/face/add_face`, `/face/get-face-embedding` all use the
   512-d path.
3. Keep the 128-d `faces` table for 30 days as rollback insurance.
4. Recalibrate:

   * Target `ARCFACE_MATCH_THRESHOLD` ≈ 0.42 initially.
   * Run nightly cron that scans low-margin matches and reports to ops.
   * Tighten to 0.44–0.46 once we see false-accept rate in real traffic.

### Phase 4 — Cleanup

1. Drop the 128-d `faces` table.
2. Remove `face_recognition` / `dlib` from requirements if no other code path
   needs it (we still use `face_recognition.face_landmarks` for ICAO gates —
   keep it until the InsightFace 5-point landmark path is wired).

## What this buys us

* Reliable production accuracy in the 99.2–99.5% range on our user base.
* Meaningfully smaller false-accept on inter-class similarity (the exact
  "Dilshad got matched as someone else" class of bug we saw).
* Robustness to masks, partial occlusion, low light — areas where dlib 128-d
  visibly degrades.

## What this does not fix on its own

* Passive spoof. Still need the moire/saturation/specular heuristics we
  added, plus on-device liveness (the 5-step motion sequence stays).
* Orphan collisions from stale Mongo `_id` values. The Node auto-heal path
  stays relevant.
* Bad-quality enrollments. The ICAO gates (pose + blur + brightness + single
  face) stay mandatory.

## Rollback plan

* `EMBED_BACKEND=dlib` env flip (single restart) reverts to 128-d.
* `faces` table kept for 30 days post-cutover — no data loss window.
* Thresholds are env-controlled — no redeploy to retune.
