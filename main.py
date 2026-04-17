import _pkg_resources_compat  # noqa: F401 — must load before face_recognition on Py3.12+

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
import faiss
import numpy as np
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import threading
from pydantic import BaseModel
import logging
from typing import List, Dict, Optional, Tuple
import redis
import pickle
import hashlib
import os
import time
import json
import base64
import face_recognition
from PIL import Image, ExifTags, ImageOps
import io
from fastapi.staticfiles import StaticFiles
import uuid
import math
import cv2
from skimage import exposure

# Load .env so DB_HOST / DB_NAME / REDIS_URL etc. apply (same keys as docker-compose)
def _load_env_local():
    try:
        from dotenv import load_dotenv

        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.isfile(env_path):
            load_dotenv(env_path)
    except ImportError:
        pass


_load_env_local()

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Constants
# Change these constants at the top of your code
DIM = 128
HIGH_CONFIDENCE_THRESHOLD = 0.95  # 95% minimum for positive match
BASE_THRESHOLD = 0.85             # Regular threshold
APPEARANCE_VARIATION_THRESHOLD = 0.80  # For minor variations
MIN_FACE_SIZE = 100  # Minimum face size in pixels
# Laplacian variance on face ROI — below this = reject as too blurry (tune via env)
FACE_MIN_BLUR_VARIANCE = float(os.getenv("FACE_MIN_BLUR_VARIANCE", "55"))
CACHE_TTL = 300  # 5 minutes cache TTL
MAX_CACHE_SIZE = 1000  # Maximum cache size for descriptors

# Environment Variables (DB_* preferred; POSTGRES_* fallback for legacy .env)
POSTGRES_HOST = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB", "postgres")
POSTGRES_USER = os.getenv("DB_USER") or os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASS = os.getenv("DB_PASS") or os.getenv("POSTGRES_PASSWORD", "postgres")
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("KV_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
USE_FAKE_REDIS = os.getenv("REDIS_FAKE", "").lower() in ("1", "true", "yes")

# ------------------------------------------------------------------------------
# Database and Redis Setup
# ------------------------------------------------------------------------------

pool = ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASS,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
)

def get_db_conn():
    conn = pool.getconn()
    conn.autocommit = True
    return conn

def put_db_conn(conn):
    pool.putconn(conn)

def _build_redis_client():
    if USE_FAKE_REDIS:
        import fakeredis
        return fakeredis.FakeRedis(), True

    if REDIS_URL:
        try:
            client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
            return client, False
        except Exception as e:
            print(f"[redis] Invalid REDIS_URL/KV_URL, falling back to REDIS_HOST/REDIS_PORT: {e}")

    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0), False

r, _redis_is_fake = _build_redis_client()

def cache_json_event(key: str, payload: Dict, ttl_sec: int = 3600) -> None:
    """Best-effort Redis writer for observability; never breaks API flow."""
    try:
        r.setex(key, ttl_sec, json.dumps(payload, default=str))
    except Exception as e:
        logger.warning(f"Redis cache write failed for key={key}: {e}")

def cache_face_details(
    key_prefix: str,
    descriptor: Optional[np.ndarray] = None,
    image_bytes: Optional[bytes] = None,
    ttl_sec: int = 86400,
    meta: Optional[Dict] = None,
) -> None:
    """Store descriptor/image details in Redis for debugging and traceability."""
    payload: Dict = dict(meta or {})
    if descriptor is not None:
        arr = np.array(descriptor, dtype=np.float32)
        payload["descriptor_dim"] = int(arr.shape[0])
        payload["descriptor"] = [round(float(x), 8) for x in arr.tolist()]
        payload["descriptor_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    if image_bytes is not None:
        payload["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        payload["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
        payload["image_bytes"] = len(image_bytes)
    cache_json_event(key_prefix, payload, ttl_sec=ttl_sec)

# ------------------------------------------------------------------------------
# FAISS Index Setup
# ------------------------------------------------------------------------------

index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
id_to_user_id: Dict[int, str] = {}
faiss_lock = threading.Lock()

# In-memory cache for faster duplicate checking
face_cache = {}
cache_order = []

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def face_hash(vector: np.ndarray) -> str:
    """Create a unique hash for the face vector."""
    return hashlib.md5(vector.tobytes()).hexdigest()

def enhance_face_features(img_np: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing for better matching in poor conditions"""
    try:
        # Convert to LAB color space for better lighting normalization
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    except Exception as e:
        logger.warning(f"Enhancement failed, using original: {str(e)}")
        return img_np

def get_robust_face_encoding(img_np: np.ndarray, face_location, fast_mode: bool = False) -> np.ndarray:
    """Get face encoding with multiple strategies for robustness."""
    try:
        if len(face_location) != 4:
            logger.error("Invalid face location format")
            return None

        encodings = []
        jitters = 2 if fast_mode else 3

        enc_large = face_recognition.face_encodings(
            img_np,
            known_face_locations=[face_location],
            num_jitters=jitters,
            model="large"
        )
        if enc_large:
            encodings.extend(enc_large)

        if not fast_mode:
            try:
                enhanced = enhance_face_features(img_np)
                if enhanced is not None and enhanced.shape == img_np.shape:
                    enc_enhanced = face_recognition.face_encodings(
                        enhanced,
                        known_face_locations=[face_location],
                        num_jitters=1,
                        model="large"
                    )
                    if enc_enhanced:
                        encodings.extend(enc_enhanced)
            except Exception:
                pass

        if not encodings:
            enc_small = face_recognition.face_encodings(
                img_np,
                known_face_locations=[face_location],
                num_jitters=1,
                model="small"
            )
            if enc_small:
                encodings.extend(enc_small)

        if not encodings:
            logger.warning("No face encodings generated")
            return None

        avg_encoding = np.mean(encodings, axis=0)
        normalized_encoding = avg_encoding / np.linalg.norm(avg_encoding)

        return normalized_encoding

    except Exception as e:
        logger.error(f"Robust encoding failed: {str(e)}", exc_info=True)
        return None

def preprocess_image(image: Image.Image) -> Image.Image:
    """Enhanced image preprocessing for better face detection."""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Auto-orient based on EXIF
        image = ImageOps.exif_transpose(image)
        
        # Enhance contrast slightly
        image = ImageOps.autocontrast(image, cutoff=2)
        
        return image
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise HTTPException(400, "Image processing failed")

def face_crop_laplacian_variance(img_np: np.ndarray, face_location) -> float:
    """Sharpness proxy: higher = sharper. face_location is (top, right, bottom, left)."""
    try:
        top, right, bottom, left = face_location
        h, w = img_np.shape[:2]
        top = max(0, int(top))
        left = max(0, int(left))
        bottom = min(h, int(bottom))
        right = min(w, int(right))
        if bottom <= top + 1 or right <= left + 1:
            return 0.0
        crop = img_np[top:bottom, left:right]
        if crop.size == 0:
            return 0.0
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        logger.warning(f"Laplacian variance failed: {e}")
        return 0.0


def blur_rejection_payload(img_np: np.ndarray, face_location, min_var: float) -> Optional[Dict]:
    """If face ROI is too blurry, return API error dict; else None."""
    v = face_crop_laplacian_variance(img_np, face_location)
    if v < min_var:
        logger.warning(
            f"[blur] Rejecting: laplacian_var={v:.2f} < min={min_var:.2f}"
        )
        return {
            "status": False,
            "message": "Image too blurry — hold steady, use good light, and retake",
            "blur_score": round(v, 2),
            "min_blur_required": min_var,
        }
    return None


def validate_face(face_location, image_size) -> bool:
    """Validate face size and position."""
    top, right, bottom, left = face_location
    width = right - left
    height = bottom - top

    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return False

    img_width, img_height = image_size
    if left < 0 or right > img_width or top < 0 or bottom > img_height:
        return False

    return True

def get_most_centered_face(face_locations, image_size):
    """Select the most centered face when multiple are detected."""
    img_center_x = image_size[0] / 2
    img_center_y = image_size[1] / 2
    min_distance = float('inf')
    best_index = 0
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        if not validate_face((top, right, bottom, left), image_size):
            continue
            
        face_center_x = (left + right) / 2
        face_center_y = (top + bottom) / 2
        distance = ((face_center_x - img_center_x)**2 + 
                    (face_center_y - img_center_y)**2)**0.5
        
        if distance < min_distance:
            min_distance = distance
            best_index = i
    
    return best_index if min_distance != float('inf') else None

def update_cache(descriptor: np.ndarray, user_id: str):
    """Update in-memory cache with LRU eviction policy."""
    descriptor_hash = face_hash(descriptor)
    
    if descriptor_hash in face_cache:
        # Move to end of cache_order (most recently used)
        cache_order.remove(descriptor_hash)
        cache_order.append(descriptor_hash)
    else:
        if len(face_cache) >= MAX_CACHE_SIZE:
            # Remove least recently used
            oldest = cache_order.pop(0)
            del face_cache[oldest]
        
        face_cache[descriptor_hash] = user_id
        cache_order.append(descriptor_hash)

def robust_face_detection(img_np: np.ndarray, label: str = "") -> Tuple[list, np.ndarray]:
    """Multi-strategy face detection with automatic fallbacks.
    Returns (face_locations, detection_image) so the caller can encode on
    the same image the detector actually found the face in.

    Strategy order (fast → slow):
      Phase 1  HOG on 0°/90°/270°/180°              ~400 ms total
      Phase 2  HOG-2x-upsample on all orientations   ~800 ms
      Phase 3  Upscale small images + HOG on all      ~600 ms
      Phase 4  CLAHE + HOG on 0° and 90°              ~300 ms
      Phase 5  CNN on 0° and 90° (slow fallback)      ~6 s
    """
    tag = f"[{label}] " if label else ""

    candidates = [(img_np, "0°")]
    for angle, name in [(90, "90°"), (270, "270°"), (180, "180°")]:
        rot = np.array(Image.fromarray(img_np).rotate(angle, expand=True))
        candidates.append((rot, name))

    # Phase 1 — HOG on every orientation (covers EXIF-stripped phone images)
    for test_img, name in candidates:
        locs = face_recognition.face_locations(test_img, model="hog")
        if locs:
            if name != "0°":
                logger.info(f"{tag}Phase-1 face found at {name} (HOG)")
            return locs, test_img

    # Phase 2 — HOG with 2× upsampling on every orientation
    for test_img, name in candidates:
        locs = face_recognition.face_locations(
            test_img, number_of_times_to_upsample=2, model="hog"
        )
        if locs:
            logger.info(f"{tag}Phase-2 face found at {name} (HOG 2x)")
            return locs, test_img

    # Phase 3 — Upscale small images (< 600 px longest edge) then retry HOG
    h, w = img_np.shape[:2]
    if max(h, w) < 600:
        scale = 600.0 / max(h, w)
        upscaled = cv2.resize(img_np, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        up_candidates = [(upscaled, "0° upscaled")]
        for angle, name in [(90, "90° upscaled"), (270, "270° upscaled")]:
            rot = np.array(Image.fromarray(upscaled).rotate(angle, expand=True))
            up_candidates.append((rot, name))
        for test_img, name in up_candidates:
            locs = face_recognition.face_locations(test_img, model="hog")
            if locs:
                logger.info(f"{tag}Phase-3 face found at {name} (HOG)")
                return locs, test_img

    # Phase 4 — CLAHE enhancement + HOG on 0° and 90°
    for test_img, name in candidates[:2]:
        try:
            enhanced = enhance_face_features(test_img)
            locs = face_recognition.face_locations(enhanced, model="hog")
            if locs:
                logger.info(f"{tag}Phase-4 face found at {name} (CLAHE+HOG)")
                return locs, enhanced
        except Exception:
            pass

    # Phase 5 — CNN on 0° and 90° (slow but most robust)
    for test_img, name in candidates[:2]:
        try:
            locs = face_recognition.face_locations(test_img, model="cnn")
            if locs:
                logger.info(f"{tag}Phase-5 face found at {name} (CNN)")
                return locs, test_img
        except Exception as e:
            logger.warning(f"{tag}CNN failed at {name}: {e}")

    logger.warning(f"{tag}All detection strategies exhausted "
                   f"(img {w}x{h}, {img_np.nbytes//1024}KB)")
    return [], img_np


async def process_uploaded_image(image: UploadFile) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """Process uploaded image and return face location"""
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img_np = np.array(img)

        face_locations, det_img = robust_face_detection(img_np, label="process_uploaded")
        if not face_locations:
            return None, None

        face_location = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[3]-loc[1]))
        return det_img, face_location

    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}", exc_info=True)
        return None, None

# ------------------------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------------------------

def initialize_faiss_index():
    """Load face descriptors from Postgres into FAISS index on startup."""
    conn = get_db_conn()
    try:
        cur = conn.cursor()

        with faiss_lock:
            index.reset()
        id_to_user_id.clear()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id SERIAL PRIMARY KEY,
                user_id TEXT UNIQUE,
                face_descriptor FLOAT4[]
            );
        """)
        conn.commit()

        cur.execute("SELECT id, user_id, face_descriptor FROM faces;")
        rows = cur.fetchall()

        if rows:
            vectors = np.array([row[2] for row in rows], dtype=np.float32)
            ids = np.array([row[0] for row in rows], dtype='int64')
            with faiss_lock:
                index.add_with_ids(vectors, ids)
            id_to_user_id.update({row[0]: row[1] for row in rows})

            for row in rows:
                vector = np.array(row[2], dtype=np.float32)
                update_cache(vector, row[1])

            logger.info(f"Loaded {len(rows)} face descriptors into FAISS.")
        else:
            logger.info("No face descriptors in database.")

        cur.close()
    finally:
        put_db_conn(conn)

# ------------------------------------------------------------------------------
# Matching / Registration thresholds
# Production tuning (2026-04):
#   * Raised search threshold to 0.93 to cut demographic false-positives.
#   * Registration threshold aligned to 0.93 so duplicate detection and match
#     operate on the same surface.
#   * Introduced FACE_MATCH_MIN_MARGIN: top-1 minus top-2 similarity must be
#     >= margin to accept a match (rejects noisy near-ties).
# ------------------------------------------------------------------------------
ADD_FACE_REJECT_THRESHOLD = float(os.getenv("ADD_FACE_REJECT_THRESHOLD", "0.93"))
SEARCH_FACE_MATCH_THRESHOLD = float(os.getenv("SEARCH_FACE_MATCH_THRESHOLD", "0.90"))
# `/face/get-face-embedding` only accepts matches at or above this similarity (default 93%).
SEARCH_HIGH_CONFIDENCE = float(os.getenv("SEARCH_HIGH_CONFIDENCE", "0.93"))
# Used by other flows (e.g. legacy); get-face-embedding no longer relaxes to this threshold.
SEARCH_APPEARANCE_VARIATION = float(os.getenv("SEARCH_APPEARANCE_VARIATION", "0.90"))
# Minimum margin (top-1 similarity minus top-2 similarity) to accept a match.
FACE_MATCH_MIN_MARGIN = float(os.getenv("FACE_MATCH_MIN_MARGIN", "0.04"))

# ------------------------------------------------------------------------------
# Face-registration quality gates (ICAO-inspired defaults)
# Pose envelope tuned for "near-frontal" registration:
#   Yaw   (left-right turn) : -30 .. +30 deg
#   Pitch (up-down tilt)    : -15 .. +15 deg
#   Roll  (head-side tilt)  : -10 .. +10 deg
# ------------------------------------------------------------------------------
REG_MIN_IMAGE_DIM = int(os.getenv("REG_MIN_IMAGE_DIM", "320"))            # full image min dimension
REG_MIN_FACE_SIZE = int(os.getenv("REG_MIN_FACE_SIZE", "160"))            # face bbox min width/height
REG_MIN_BLUR_VARIANCE = float(os.getenv("REG_MIN_BLUR_VARIANCE", "80"))   # sharper than auth capture
# Mobile-selfie realistic envelope. Strict enough to keep enrollment frontal
# without producing false rejects when the phone is held slightly low/high
# (which adds apparent pitch) or when the front camera is off-axis.
REG_MAX_YAW_DEGREES = float(os.getenv("REG_MAX_YAW_DEGREES", "35"))       # head turn (left-right)
REG_MAX_PITCH_DEGREES = float(os.getenv("REG_MAX_PITCH_DEGREES", "25"))   # head tilt (up-down)
REG_MAX_ROLL_DEGREES = float(os.getenv("REG_MAX_ROLL_DEGREES", "15"))     # head rotation sideways
REG_MIN_BRIGHTNESS = float(os.getenv("REG_MIN_BRIGHTNESS", "55"))         # mean pixel brightness
REG_MAX_BRIGHTNESS = float(os.getenv("REG_MAX_BRIGHTNESS", "210"))
REG_MIN_CONTRAST = float(os.getenv("REG_MIN_CONTRAST", "20"))             # std-dev of brightness
REG_MIN_FACE_AREA_RATIO = float(os.getenv("REG_MIN_FACE_AREA_RATIO", "0.08"))  # face must occupy >= 8%
REG_MAX_FACE_AREA_RATIO = float(os.getenv("REG_MAX_FACE_AREA_RATIO", "0.80"))  # face must occupy <= 80%
REG_REQUIRED_LANDMARK_GROUPS = (
    "left_eye", "right_eye", "nose_bridge", "top_lip", "bottom_lip"
)

# Passive anti-spoof heuristic thresholds (image-level, no liveness prompt).
# These are conservative: they block the most common screen/photo replays but
# can't replace a dedicated liveness model.
REG_SPOOF_ENABLE = str(os.getenv("REG_SPOOF_ENABLE", "true")).lower() in ("1", "true", "yes")
REG_SPOOF_MAX_MOIRE_RATIO = float(os.getenv("REG_SPOOF_MAX_MOIRE_RATIO", "0.58"))   # FFT high-freq share — screens show strong moire
REG_SPOOF_MIN_COLOR_STD = float(os.getenv("REG_SPOOF_MIN_COLOR_STD", "8.0"))         # saturation std-dev — printed/low-gamut photos are flat
REG_SPOOF_MAX_SPECULAR_RATIO = float(os.getenv("REG_SPOOF_MAX_SPECULAR_RATIO", "0.18"))  # over-bright pixel share — glass/screens show big specular patches


def estimate_face_pose(
    img_np: np.ndarray,
    face_location: Tuple[int, int, int, int],
    landmarks: Optional[Dict],
) -> Dict[str, Optional[float]]:
    """Estimate yaw / pitch / roll in degrees using OpenCV solvePnP against
    a canonical 3D face model. Returns dict with yaw, pitch, roll or Nones
    if estimation fails.
    """
    result = {"yaw": None, "pitch": None, "roll": None}
    try:
        if not landmarks:
            return result

        # 2D image points (pixel coords) — mean of each landmark group.
        def mean_point(key):
            pts = landmarks.get(key)
            if not pts:
                return None
            arr = np.array(pts, dtype=np.float32)
            return tuple(arr.mean(axis=0))

        left_eye = mean_point("left_eye")
        right_eye = mean_point("right_eye")
        nose_pts = landmarks.get("nose_tip") or landmarks.get("nose_bridge")
        top_lip = mean_point("top_lip")
        bottom_lip = mean_point("bottom_lip")
        chin_pts = landmarks.get("chin")

        if not (left_eye and right_eye and nose_pts and top_lip and chin_pts):
            return result

        nose_tip = nose_pts[len(nose_pts) // 2]          # ~tip of nose
        mouth_center = (
            (top_lip[0] + bottom_lip[0]) / 2.0,
            (top_lip[1] + bottom_lip[1]) / 2.0,
        )
        # Chin lowest point (max y).
        chin_point = max(chin_pts, key=lambda p: p[1])

        image_points = np.array([
            nose_tip,
            chin_point,
            left_eye,
            right_eye,
            top_lip[:2],
            bottom_lip[:2],
        ], dtype=np.float64)

        # Canonical 3D points (mm) for a generic human face.
        model_points = np.array([
            (0.0,    0.0,    0.0),      # nose tip
            (0.0,  -63.6,  -12.5),      # chin
            (-43.3, 32.7,  -26.0),      # left eye corner
            (43.3,  32.7,  -26.0),      # right eye corner
            (-0.0,  -28.0, -15.0),      # upper lip
            (-0.0,  -45.0, -20.0),      # lower lip
        ], dtype=np.float64)

        h, w = img_np.shape[:2]
        focal = float(w)
        cam_matrix = np.array([
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _tvec = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return result

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose to Euler (degrees). Convention: (pitch, yaw, roll).
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = 0.0

        # OpenCV Euler extraction can sometimes produce equivalent angles near
        # +/-180 for near-frontal faces. Normalize to human-friendly ranges to
        # avoid false rejects like pitch=163 or roll=-166 on a straight face.
        def _normalize_pose_angle(deg: float) -> float:
            # First fold to [-180, 180].
            a = ((float(deg) + 180.0) % 360.0) - 180.0
            # Then fold equivalent near-180 solutions into [-90, 90].
            if a > 90.0:
                a = 180.0 - a
            elif a < -90.0:
                a = -180.0 - a
            return a

        # Pitch positive = looking down in this decomposition; invert so
        # positive means looking up (more intuitive for thresholds).
        pitch = -pitch
        yaw = _normalize_pose_angle(yaw)
        pitch = _normalize_pose_angle(pitch)
        roll = _normalize_pose_angle(roll)

        result["yaw"] = float(round(yaw, 2))
        result["pitch"] = float(round(pitch, 2))
        result["roll"] = float(round(roll, 2))
        return result
    except Exception as e:
        logger.warning(f"pose estimation failed: {e}")
        return result


def passive_spoof_report(img_np: np.ndarray, face_location: Tuple[int, int, int, int]) -> Dict:
    """Heuristic passive-spoof check. Returns dict with:
       is_suspicious (bool), reasons (list), moire_ratio, saturation_std, specular_ratio.
    Lightweight; intended as a pre-filter only. Real liveness should be used
    on top of this (captured on device using motion/yaw variations).
    """
    report: Dict = {
        "is_suspicious": False,
        "reasons": [],
        "moire_ratio": None,
        "saturation_std": None,
        "specular_ratio": None,
    }
    try:
        top, right, bottom, left = face_location
        top = max(0, int(top)); left = max(0, int(left))
        bottom = min(img_np.shape[0], int(bottom)); right = min(img_np.shape[1], int(right))
        roi = img_np[top:bottom, left:right]
        if roi.size == 0 or roi.shape[0] < 48 or roi.shape[1] < 48:
            return report

        # --- Moire / screen-replay check via FFT high-frequency energy ---
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
        g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        g -= g.mean()
        f = np.fft.fft2(g)
        mag = np.abs(np.fft.fftshift(f))
        total = mag.sum() + 1e-6
        yy, xx = np.indices(mag.shape)
        center = np.array([128, 128])
        dist = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
        high_mask = dist > 40
        high_energy = mag[high_mask].sum()
        moire_ratio = float(high_energy / total)
        report["moire_ratio"] = round(moire_ratio, 4)

        # --- Saturation variance (printed photos / low-gamut screens are flat) ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV) if roi.ndim == 3 else None
        if hsv is not None:
            sat_std = float(hsv[..., 1].std())
            report["saturation_std"] = round(sat_std, 2)
        else:
            sat_std = 100.0

        # --- Specular highlight ratio (screens reflect big specular blobs) ---
        _, spec = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        specular_ratio = float((spec > 0).mean())
        report["specular_ratio"] = round(specular_ratio, 4)

        reasons: List[str] = []
        if moire_ratio > REG_SPOOF_MAX_MOIRE_RATIO:
            reasons.append("moire_pattern_detected")
        if sat_std < REG_SPOOF_MIN_COLOR_STD:
            reasons.append("flat_color_distribution")
        if specular_ratio > REG_SPOOF_MAX_SPECULAR_RATIO:
            reasons.append("excessive_specular_highlights")

        report["reasons"] = reasons
        report["is_suspicious"] = len(reasons) > 0
        return report
    except Exception as e:
        logger.warning(f"spoof check failed: {e}")
        return report

def face_exists(vector: np.ndarray) -> Tuple[bool, Optional[str], Optional[float]]:
    """Check if face exists above the rejection threshold"""
    results = search_face(vector, k=5)  # Check multiple potential matches
    if not results:
        return False, None, None
        
    # Return the highest similarity match that's above threshold
    for match in results:
        if match["similarity"] >= ADD_FACE_REJECT_THRESHOLD:
            return True, match["user_id"], match["similarity"]
    
    return False, None, None

def search_face(vector: np.ndarray, k: int = 5, threshold_override: float = None) -> List[Dict[str, float]]:
    """Search for faces. Only returns matches at or above the effective threshold."""
    assert vector.shape == (DIM,), f"Must be {DIM}-dim vector."

    effective_threshold = threshold_override if threshold_override is not None else SEARCH_FACE_MATCH_THRESHOLD

    with faiss_lock:
        if index.ntotal == 0:
            return []
        distances, ids = index.search(vector[None, :], k * 3)

    results = []
    for rank, (idx, dist) in enumerate(zip(ids[0], distances[0])):
        if idx == -1:
            continue

        similarity = 1 - dist
        user_id = id_to_user_id.get(int(idx), None)

        if rank < 3:
            logger.info(
                f"[FAISS] rank={rank} user={user_id} dist={dist:.6f} sim={similarity:.4f} "
                f"threshold={effective_threshold}"
            )

        if similarity >= effective_threshold and user_id:
            if similarity >= SEARCH_HIGH_CONFIDENCE:
                confidence = "high"
            elif similarity >= SEARCH_FACE_MATCH_THRESHOLD:
                confidence = "medium"
            else:
                confidence = "low"
            results.append({
                "user_id": user_id,
                "similarity": float(similarity),
                "confidence": confidence
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:k]


def search_face_with_margin(
    vector: np.ndarray,
    threshold_override: float = None,
    observe_top_k: int = 5,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Like search_face but also returns observability stats for the best and
    second-best neighbours in the index (regardless of threshold).

    Returns (matches_passing_threshold, stats) where stats contains:
      - top1_similarity / top1_user
      - top2_similarity / top2_user
      - margin (top1 - top2, or top1 if only one match in index)
      - index_total (total descriptors indexed)
    Matches list already filters to effective_threshold.
    """
    assert vector.shape == (DIM,), f"Must be {DIM}-dim vector."
    effective_threshold = (
        threshold_override if threshold_override is not None else SEARCH_FACE_MATCH_THRESHOLD
    )

    stats: Dict[str, float] = {
        "top1_similarity": None,
        "top1_user": None,
        "top2_similarity": None,
        "top2_user": None,
        "margin": None,
        "index_total": 0,
        "threshold": float(effective_threshold),
    }

    with faiss_lock:
        stats["index_total"] = int(index.ntotal)
        if index.ntotal == 0:
            return [], stats
        distances, ids = index.search(vector[None, :], max(observe_top_k, 2))

    raw = []
    for dist, idx in zip(distances[0], ids[0]):
        if idx == -1:
            continue
        uid = id_to_user_id.get(int(idx), None)
        if uid is None:
            continue
        raw.append((uid, 1.0 - float(dist)))

    raw.sort(key=lambda x: x[1], reverse=True)
    if raw:
        stats["top1_user"] = raw[0][0]
        stats["top1_similarity"] = float(raw[0][1])
    if len(raw) >= 2:
        stats["top2_user"] = raw[1][0]
        stats["top2_similarity"] = float(raw[1][1])
        stats["margin"] = float(raw[0][1] - raw[1][1])
    elif len(raw) == 1:
        # Only one face registered: margin is conceptually the full similarity.
        stats["margin"] = float(raw[0][1])

    passing: List[Dict[str, float]] = []
    for uid, sim in raw:
        if sim < effective_threshold:
            continue
        if sim >= SEARCH_HIGH_CONFIDENCE:
            confidence = "high"
        elif sim >= SEARCH_FACE_MATCH_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "low"
        passing.append({
            "user_id": uid,
            "similarity": sim,
            "confidence": confidence,
        })

    return passing, stats


def face_quality_report(
    img_np: np.ndarray,
    face_location,
    img_size_wh: Tuple[int, int],
    num_faces_detected: int,
) -> Tuple[bool, Dict]:
    """Return (is_ok, report). Applies ICAO-style gates for registration quality.
    img_size_wh is (width, height). face_location is (top, right, bottom, left).
    """
    try:
        top, right, bottom, left = face_location
        img_w, img_h = img_size_wh
        face_w = max(0, int(right - left))
        face_h = max(0, int(bottom - top))
        face_area = face_w * face_h
        img_area = max(1, img_w * img_h)
        area_ratio = face_area / img_area

        report: Dict = {
            "image_width": int(img_w),
            "image_height": int(img_h),
            "face_width": face_w,
            "face_height": face_h,
            "face_area_ratio": round(float(area_ratio), 4),
            "num_faces": int(num_faces_detected),
        }

        reasons: List[str] = []

        if img_w < REG_MIN_IMAGE_DIM or img_h < REG_MIN_IMAGE_DIM:
            reasons.append(
                f"image_resolution_too_low (min {REG_MIN_IMAGE_DIM}px)"
            )

        if num_faces_detected > 1:
            reasons.append("multiple_faces_detected")

        if face_w < REG_MIN_FACE_SIZE or face_h < REG_MIN_FACE_SIZE:
            reasons.append(
                f"face_too_small (min {REG_MIN_FACE_SIZE}px; got {face_w}x{face_h})"
            )

        if area_ratio < REG_MIN_FACE_AREA_RATIO:
            reasons.append(
                f"face_too_far (needs >= {int(REG_MIN_FACE_AREA_RATIO*100)}% of frame)"
            )
        elif area_ratio > REG_MAX_FACE_AREA_RATIO:
            reasons.append(
                f"face_too_close (needs <= {int(REG_MAX_FACE_AREA_RATIO*100)}% of frame)"
            )

        # Blur on face ROI
        blur_var = face_crop_laplacian_variance(img_np, face_location)
        report["blur_variance"] = round(blur_var, 2)
        if blur_var < REG_MIN_BLUR_VARIANCE:
            reasons.append(
                f"image_too_blurry (min {REG_MIN_BLUR_VARIANCE})"
            )

        # Brightness & contrast on face ROI
        try:
            t, r_, b, l_ = face_location
            t = max(0, int(t)); l_ = max(0, int(l_))
            b = min(img_np.shape[0], int(b)); r_ = min(img_np.shape[1], int(r_))
            roi = img_np[t:b, l_:r_]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
                mean_val = float(gray.mean())
                std_val = float(gray.std())
                report["brightness_mean"] = round(mean_val, 2)
                report["contrast_std"] = round(std_val, 2)
                if mean_val < REG_MIN_BRIGHTNESS:
                    reasons.append("image_too_dark")
                elif mean_val > REG_MAX_BRIGHTNESS:
                    reasons.append("image_overexposed")
                if std_val < REG_MIN_CONTRAST:
                    reasons.append("low_contrast")
        except Exception as e:
            logger.warning(f"brightness check failed: {e}")

        # Landmarks: require all critical groups; then compute full 3D pose
        # (yaw/pitch/roll) via solvePnP and gate on the registration envelope.
        try:
            landmarks_list = face_recognition.face_landmarks(
                img_np, face_locations=[face_location]
            )
            if not landmarks_list:
                reasons.append("facial_landmarks_not_detected")
            else:
                lm = landmarks_list[0]
                missing = [g for g in REG_REQUIRED_LANDMARK_GROUPS if g not in lm or not lm[g]]
                if missing:
                    reasons.append(
                        f"occluded_or_partial_face (missing: {','.join(missing)})"
                    )
                else:
                    # Quick eye-line roll (fallback if solvePnP fails)
                    left_eye_c = np.mean(lm["left_eye"], axis=0)
                    right_eye_c = np.mean(lm["right_eye"], axis=0)
                    dx = float(right_eye_c[0] - left_eye_c[0])
                    dy = float(right_eye_c[1] - left_eye_c[1])
                    eye_roll = math.degrees(math.atan2(dy, dx))

                    pose = estimate_face_pose(img_np, face_location, lm)
                    yaw = pose.get("yaw")
                    pitch = pose.get("pitch")
                    roll = pose.get("roll")
                    if roll is None:
                        roll = eye_roll

                    report["yaw_degrees"] = yaw
                    report["pitch_degrees"] = pitch
                    report["roll_degrees"] = round(roll, 2) if roll is not None else None

                    if yaw is not None and abs(yaw) > REG_MAX_YAW_DEGREES:
                        reasons.append(
                            f"head_turned (yaw must be within +/-{REG_MAX_YAW_DEGREES} deg; got {round(yaw,1)} deg). Please look straight."
                        )
                    if pitch is not None and abs(pitch) > REG_MAX_PITCH_DEGREES:
                        reasons.append(
                            f"head_tilted_updown (pitch must be within +/-{REG_MAX_PITCH_DEGREES} deg; got {round(pitch,1)} deg). Please look straight."
                        )
                    if roll is not None and abs(roll) > REG_MAX_ROLL_DEGREES:
                        reasons.append(
                            f"head_tilted_sideways (roll must be within +/-{REG_MAX_ROLL_DEGREES} deg; got {round(roll,1)} deg). Please look straight."
                        )
        except Exception as e:
            logger.warning(f"landmark/pose check failed: {e}")
            reasons.append("landmark_check_failed")

        # Passive spoof / replay heuristics (screen, printed photo, glossy paper)
        if REG_SPOOF_ENABLE:
            try:
                spoof = passive_spoof_report(img_np, face_location)
                report["moire_ratio"] = spoof.get("moire_ratio")
                report["saturation_std"] = spoof.get("saturation_std")
                report["specular_ratio"] = spoof.get("specular_ratio")
                if spoof.get("is_suspicious"):
                    reasons.append(
                        "possible_spoof_detected (" + ",".join(spoof.get("reasons", [])) + ")"
                    )
            except Exception as e:
                logger.warning(f"spoof check wrapper failed: {e}")

        report["reasons"] = reasons
        return (len(reasons) == 0), report

    except Exception as e:
        logger.error(f"face_quality_report failed: {e}", exc_info=True)
        return False, {"reasons": ["quality_check_exception"], "error": str(e)}

def add_face(user_id: str, vector: np.ndarray) -> Tuple[Optional[int], str]:
    """Add face with strict duplicate checking"""
    assert vector.shape == (DIM,), f"Must be {DIM}-dim vector."

    conn = get_db_conn()
    try:
        cur = conn.cursor()

        cur.execute("SELECT id FROM faces WHERE user_id = %s", (user_id,))
        if cur.fetchone():
            cur.close()
            return None, f"Face already exists for user_id {user_id}"

        exists, existing_user, similarity = face_exists(vector)
        if exists:
            cur.close()
            return None, f"Similar face already exists for user {existing_user} (similarity={similarity:.2f})"

        cur.execute(
            "INSERT INTO faces (user_id, face_descriptor) VALUES (%s, %s) RETURNING id",
            (user_id, vector.tolist())
        )
        face_id = int(cur.fetchone()[0])

        with faiss_lock:
            index.add_with_ids(vector[None, :], np.array([face_id]))
        id_to_user_id[face_id] = user_id
        update_cache(vector, user_id)

        cur.close()
        return face_id, f"Face added for {user_id}"

    except psycopg2.Error as e:
        logger.error(f"DB error while adding face: {str(e)}")
        return None, f"Database error: {str(e)}"
    finally:
        put_db_conn(conn)

def delete_face_descriptor(user_id: str) -> Tuple[bool, str]:
    """Delete face descriptor from all storage layers."""
    conn = get_db_conn()
    try:
        cur = conn.cursor()

        cur.execute("SELECT id, face_descriptor FROM faces WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return False, f"No face found for user_id {user_id}"

        face_id, descriptor = row
        descriptor = np.array(descriptor, dtype=np.float32)

        with faiss_lock:
            index.remove_ids(np.array([face_id], dtype=np.int64))
        id_to_user_id.pop(face_id, None)

        face_cache.pop(face_hash(descriptor), None)
        for key in r.scan_iter(f"face:*:{user_id}"):
            r.delete(key)

        cur.execute("DELETE FROM faces WHERE id = %s", (face_id,))
        cur.close()
        return True, f"Deleted face for user_id {user_id}"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        put_db_conn(conn)

def correct_exif_orientation(img: Image.Image) -> Image.Image:
    """Correct image orientation based on EXIF data."""
    try:
        exif = img._getexif()
        if not exif:
            return img

        for tag, value in ExifTags.TAGS.items():
            if value == 'Orientation':
                orientation_tag = tag
                break

        orientation = exif.get(orientation_tag, 1)

        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)

    except Exception:
        pass

    return img

def rotate_image_if_face_sideways(img: Image.Image) -> Image.Image:
    """Rotate image if face is detected at an angle.
    Uses border-fill to avoid black regions that confuse HOG."""
    try:
        img_np = np.array(img)
        face_landmarks_list = face_recognition.face_landmarks(img_np)

        if not face_landmarks_list:
            return img

        landmarks = face_landmarks_list[0]
        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return img

        left_eye = np.mean(landmarks["left_eye"], axis=0)
        right_eye = np.mean(landmarks["right_eye"], axis=0)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))

        if abs(angle) > 15:
            edge_color = tuple(np.array(img).mean(axis=(0, 1)).astype(int))
            img = img.rotate(-angle, expand=True, fillcolor=edge_color)

        return img
    except Exception as e:
        logger.warning(f"Rotation correction failed, using original: {e}")
        return img

def extract_and_crop_face(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """Extract and crop the most prominent face from an image"""
    try:
        # Find all face locations
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            logger.warning("No faces found in image")
            return None, None

        # Select the largest face
        face_location = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[3]-loc[1]))
        top, right, bottom, left = face_location

        # Add 20% padding around the face
        height, width = image.shape[:2]
        padding = int(0.2 * (right - left))
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)

        # Crop the face region
        cropped_face = image[top:bottom, left:right]
        
        # Verify the cropped face meets minimum size requirements
        if cropped_face.shape[0] < MIN_FACE_SIZE or cropped_face.shape[1] < MIN_FACE_SIZE:
            logger.warning(f"Cropped face too small: {cropped_face.shape}")
            return None, None

        return cropped_face, (top, right, bottom, left)

    except Exception as e:
        logger.error(f"Face extraction failed: {str(e)}", exc_info=True)
        return None, None

def get_face_encoding(image: np.ndarray, face_location: tuple) -> Optional[np.ndarray]:
    """Get face encoding from an image and face location"""
    try:
        # Get encodings for the face location
        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=[face_location],  # (top, right, bottom, left)
            num_jitters=3,
            model="large"
        )
        
        if not encodings:
            return None

        # Normalize the encoding
        encoding = encodings[0]
        return encoding / np.linalg.norm(encoding)

    except Exception as e:
        logger.error(f"Encoding failed: {str(e)}", exc_info=True)
        return None

# ------------------------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------------------------

app = FastAPI()
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()

class FaceData(BaseModel):
    faceData: List[float]

# ------------------------------------------------------------------------------
# API Endpoints (maintaining your exact endpoints)
# ------------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming: {request.method} {request.url}")
    return await call_next(request)

@app.delete("/face/delete/{user_id}")
def delete_face(user_id: str):
    success, message = delete_face_descriptor(user_id)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return {"msg": message}

@app.post("/face/get-face-embedding")
async def get_face_embedding(request: Request, image: UploadFile = File(...)):
    t0 = time.time()
    rotated_image_url = None
    req_id = str(uuid.uuid4())
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = ImageOps.exif_transpose(img)

        fast = request.query_params.get("fast", "0") == "1"

        if not fast:
            img = rotate_image_if_face_sideways(img)
            file_ext = image.filename.split('.')[-1].lower()
            file_id = f"{uuid.uuid4()}.{file_ext}"
            saved_path = os.path.join(UPLOAD_FOLDER, file_id)
            img.save(saved_path)
            rotated_image_url = f"{request.base_url}static/uploads/{file_id}"

        img_np = np.array(img)
        face_locations, det_img = robust_face_detection(img_np, label="get-face-embedding")
        if not face_locations:
            resp = {
                "status": False,
                "message": "No face detected in file",
                "rotated_image_url": rotated_image_url
            }
            cache_json_event(
                f"fiass:face:get-face-embedding:req:{req_id}",
                {"endpoint": "get-face-embedding", "status": False, "reason": "no_face", "response": resp},
                ttl_sec=3600,
            )
            return resp

        det_pil = Image.fromarray(det_img)
        best_index = get_most_centered_face(face_locations, det_pil.size)
        if best_index is None:
            best_index = 0

        best_face = face_locations[best_index]
        blur_err = blur_rejection_payload(det_img, best_face, FACE_MIN_BLUR_VARIANCE)
        if blur_err:
            blur_err["rotated_image_url"] = rotated_image_url
            cache_json_event(
                f"fiass:face:get-face-embedding:req:{req_id}",
                {"endpoint": "get-face-embedding", "status": False, "reason": "blur", "response": blur_err},
                ttl_sec=3600,
            )
            return blur_err

        descriptor = get_robust_face_encoding(det_img, best_face, fast_mode=fast)
        if descriptor is None:
            resp = {
                "status": False,
                "message": "Failed to extract face features",
                "rotated_image_url": rotated_image_url
            }
            cache_json_event(
                f"fiass:face:get-face-embedding:req:{req_id}",
                {"endpoint": "get-face-embedding", "status": False, "reason": "encoding_failed", "response": resp},
                ttl_sec=3600,
            )
            return resp
        
        arr = np.array(descriptor, dtype=np.float32)
        cache_face_details(
            f"fiass:face:get-face-embedding:req:{req_id}:details",
            descriptor=arr,
            image_bytes=contents,
            ttl_sec=86400,
            meta={"endpoint": "get-face-embedding", "request_id": req_id},
        )
        
        if arr.shape[0] != DIM:
            raise HTTPException(400, f"Invalid face vector length. Expected {DIM}, got {arr.shape[0]}.")

        # High-confidence only: no relaxed / appearance-variation second pass for this endpoint.
        results, stats = search_face_with_margin(arr, threshold_override=SEARCH_HIGH_CONFIDENCE)

        # Anti-false-positive: if fast=1 produced a noisy match, automatically
        # re-embed with fast=0 (more jitters + CLAHE) and require the SAME user
        # to re-match above threshold; else demote to no-match.
        reverified = False
        reverify_similarity = None
        if fast and results:
            try:
                arr2 = get_robust_face_encoding(det_img, best_face, fast_mode=False)
                if arr2 is not None:
                    arr2 = np.array(arr2, dtype=np.float32)
                    results2, stats2 = search_face_with_margin(
                        arr2, threshold_override=SEARCH_HIGH_CONFIDENCE
                    )
                    top_user = results[0]["user_id"]
                    confirm = next(
                        (m for m in results2 if m["user_id"] == top_user), None
                    )
                    if confirm is None or confirm["similarity"] < SEARCH_HIGH_CONFIDENCE:
                        logger.warning(
                            f"[reverify] fast=1 match {top_user} @ {results[0]['similarity']:.4f} "
                            f"FAILED fast=0 reverification (top2={stats2.get('top1_user')}@"
                            f"{stats2.get('top1_similarity')}) — rejecting"
                        )
                        results = []
                    else:
                        reverified = True
                        reverify_similarity = float(confirm["similarity"])
            except Exception as reverify_err:
                logger.warning(f"[reverify] exception ignored: {reverify_err}")

        # Margin safety: top-1 must beat top-2 by >= FACE_MATCH_MIN_MARGIN
        if results and stats.get("margin") is not None and stats.get("top2_similarity") is not None:
            if stats["margin"] < FACE_MATCH_MIN_MARGIN:
                logger.warning(
                    f"[margin] too close: top1={stats.get('top1_user')}@{stats.get('top1_similarity')} "
                    f"top2={stats.get('top2_user')}@{stats.get('top2_similarity')} margin={stats['margin']:.4f} "
                    f"< {FACE_MATCH_MIN_MARGIN} — rejecting"
                )
                results = []

        elapsed_ms = int((time.time() - t0) * 1000)

        if not results:
            resp = {
                "status": False,
                "matches": [],
                "message": "No matching face found or confidence below minimum",
                "threshold": SEARCH_HIGH_CONFIDENCE,
                "margin_required": FACE_MATCH_MIN_MARGIN,
                "top1_similarity": stats.get("top1_similarity"),
                "top2_similarity": stats.get("top2_similarity"),
                "margin_observed": stats.get("margin"),
                "appearance_variation": False,
                "elapsed_ms": elapsed_ms,
                "rotated_image_url": rotated_image_url
            }
            cache_json_event(
                f"fiass:face:get-face-embedding:req:{req_id}",
                {"endpoint": "get-face-embedding", "status": False, "reason": "no_match", "response": resp},
                ttl_sec=3600,
            )
            cache_json_event(
                "fiass:face:get-face-embedding:last",
                {"endpoint": "get-face-embedding", "status": False, "reason": "no_match", "request_id": req_id, "response": resp},
                ttl_sec=86400,
            )
            return resp

        best_match = results[0]
        resp = {
            "status": True,
            "match": {
                "user_id": best_match["user_id"],
                "similarity": round(best_match["similarity"], 4),
                "confidence": best_match.get("confidence", "high"),
                "margin": round(stats["margin"], 4) if stats.get("margin") is not None else None,
                "top2_similarity": round(stats["top2_similarity"], 4) if stats.get("top2_similarity") is not None else None,
                "reverified_fast0": reverified,
                "reverified_similarity": round(reverify_similarity, 4) if reverify_similarity is not None else None,
            },
            "message": f"Face matched with {best_match['similarity']*100:.1f}% confidence",
            "appearance_variation": False,
            "threshold": SEARCH_HIGH_CONFIDENCE,
            "margin_required": FACE_MATCH_MIN_MARGIN,
            "elapsed_ms": elapsed_ms,
            "rotated_image_url": rotated_image_url
        }
        cache_payload = {
            "endpoint": "get-face-embedding",
            "status": True,
            "request_id": req_id,
            "user_id": best_match["user_id"],
            "similarity": round(best_match["similarity"], 4),
            "margin": stats.get("margin"),
            "top2_similarity": stats.get("top2_similarity"),
            "top2_user": stats.get("top2_user"),
            "reverified_fast0": reverified,
            "appearance_variation": False,
            "threshold": SEARCH_HIGH_CONFIDENCE,
            "elapsed_ms": elapsed_ms,
        }
        cache_json_event(f"fiass:face:get-face-embedding:req:{req_id}", cache_payload, ttl_sec=3600)
        cache_json_event("fiass:face:get-face-embedding:last", cache_payload, ttl_sec=86400)
        cache_json_event(f"fiass:face:user:{best_match['user_id']}:last_match", cache_payload, ttl_sec=6400)
        cache_face_details(
            f"fiass:face:user:{best_match['user_id']}:details:last",
            descriptor=arr,
            image_bytes=contents,
            ttl_sec=6400,
            meta={"endpoint": "get-face-embedding", "request_id": req_id, "user_id": best_match["user_id"]},
        )
        return resp

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get-face-embedding: {str(e)}", exc_info=True)
        resp = {
            "status": False,
            "message": str(e),
            "rotated_image_url": rotated_image_url
        }
        cache_json_event(
            f"fiass:face:get-face-embedding:req:{req_id}",
            {"endpoint": "get-face-embedding", "status": False, "reason": "exception", "message": str(e)},
            ttl_sec=3600,
        )
        return resp

@app.post("/face/add_face/{user_id}")
async def add_face_endpoint(user_id: str, image: UploadFile = File(...)):
    """Register a face for a user. Enforces ICAO-style production gates:
      * image resolution, face size & framing
      * blur, brightness, contrast
      * frontal pose (yaw/pitch/roll envelope)
      * single face only
      * landmark completeness (no occlusion)
      * duplicate reject above ADD_FACE_REJECT_THRESHOLD
    """
    try:
        await image.seek(0)
        raw_image_bytes = await image.read()
        await image.seek(0)

        # 1) Load & orient
        try:
            pil_img = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB")
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            raise HTTPException(400, "Invalid image payload")

        img_np = np.array(pil_img)
        img_w, img_h = pil_img.size

        # 2) Robust detection (rotation + upsample + CLAHE + CNN fallback).
        # This avoids false "no_face_detected" rejects on real device captures
        # where EXIF/camera orientation or contrast can vary.
        all_faces, det_img = robust_face_detection(img_np, label="add_face")
        if all_faces:
            img_np = det_img
            img_h, img_w = img_np.shape[:2]

        if not all_faces:
            # Helpful telemetry for debugging in production logs.
            try:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                blur_v = cv2.Laplacian(gray, cv2.CV_64F).var()
                mean_v = gray.mean()
                logger.warning(
                    f"[add_face] no_face_detected user={user_id} "
                    f"img={img_w}x{img_h} blur={blur_v:.2f} brightness={mean_v:.2f}"
                )
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "no_face_detected",
                    "message": "No face detected. Please keep your full face inside the frame, look straight, and use good front lighting.",
                },
            )

        # Pick the most centered face (used for encoding) but keep full count
        best_index = get_most_centered_face(all_faces, (img_w, img_h))
        if best_index is None:
            best_index = 0
        face_location = all_faces[best_index]

        # 3) Production quality gates
        ok, report = face_quality_report(
            img_np=img_np,
            face_location=face_location,
            img_size_wh=(img_w, img_h),
            num_faces_detected=len(all_faces),
        )
        if not ok:
            logger.warning(
                f"[add_face] quality gate failed user={user_id} reasons={report.get('reasons')}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "face_quality_rejected",
                    "message": "Face image does not meet registration quality. "
                               "Please retake: look straight at the camera, ensure good lighting, remove glasses/mask, and keep only one face in frame.",
                    "quality": report,
                },
            )

        # 4) Encode with full-quality path (fast_mode=False)
        encoding = get_robust_face_encoding(img_np, face_location, fast_mode=False)
        if encoding is None:
            raise HTTPException(400, "Could not extract face features. Please retake.")

        arr = np.array(encoding, dtype=np.float32)

        # 5) Duplicate / hijack reject — aligned with SEARCH_HIGH_CONFIDENCE
        dup_hits, dup_stats = search_face_with_margin(
            arr, threshold_override=ADD_FACE_REJECT_THRESHOLD, observe_top_k=3
        )
        if dup_hits:
            top = dup_hits[0]
            if top["user_id"] != user_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "similar_face_exists",
                        "message": "This face already appears to be registered to another account.",
                        "existing_user": top["user_id"],
                        "similarity": round(top["similarity"], 3),
                        "threshold": ADD_FACE_REJECT_THRESHOLD,
                        "quality": report,
                    },
                )
            # Same user re-registering: delete existing, proceed to re-add cleanly.
            logger.info(f"[add_face] user={user_id} re-registering face; replacing old descriptor")
            try:
                delete_face_descriptor(user_id)
            except Exception as e:
                logger.warning(f"[add_face] prior descriptor delete failed: {e}")

        # 6) Insert
        face_id, message = add_face(user_id, arr)
        if face_id is None:
            raise HTTPException(status_code=409, detail={"error": "add_face_failed", "message": message})

        cache_face_details(
            f"fiass:face:user:{user_id}:details:registered",
            descriptor=arr,
            image_bytes=raw_image_bytes,
            ttl_sec=7 * 86400,
            meta={
                "endpoint": "add_face",
                "user_id": user_id,
                "face_id": face_id,
                "quality": report,
            },
        )
        return {
            "status": True,
            "id": face_id,
            "message": message,
            "quality": report,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Add face error: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/face/add_face_descriptor/{user_id}")
async def add_face_descriptor_endpoint(user_id: str, data: FaceData):
    """Accept a raw 128-dim face descriptor array (from ERP/client-side face-api.js)."""
    try:
        arr = np.array(data.faceData, dtype=np.float32)
        if arr.shape[0] != DIM:
            raise HTTPException(400, f"Invalid vector length. Expected {DIM}, got {arr.shape[0]}.")

        normalized = arr / np.linalg.norm(arr)

        dup_hits, _dup_stats = search_face_with_margin(
            normalized, threshold_override=ADD_FACE_REJECT_THRESHOLD, observe_top_k=3
        )
        if dup_hits:
            top = dup_hits[0]
            if top["user_id"] != user_id:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "similar_face_exists",
                        "message": "This face already appears to be registered to another account.",
                        "existing_user": top["user_id"],
                        "similarity": round(top["similarity"], 3),
                        "threshold": ADD_FACE_REJECT_THRESHOLD,
                    },
                )
            # Same user re-registering: remove existing before inserting.
            try:
                delete_face_descriptor(user_id)
            except Exception as e:
                logger.warning(f"[add_face_descriptor] prior descriptor delete failed: {e}")

        face_id, message = add_face(user_id, normalized)
        cache_face_details(
            f"fiass:face:user:{user_id}:details:descriptor_registered",
            descriptor=normalized,
            image_bytes=None,
            ttl_sec=7 * 86400,
            meta={"endpoint": "add_face_descriptor", "user_id": user_id, "face_id": face_id},
        )
        return {
            "status": True,
            "id": face_id,
            "message": message
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Add face descriptor error: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/face/search")
async def search_endpoint(image: UploadFile = File(...)):
    req_id = str(uuid.uuid4())
    try:
        img_np, face_location = await process_uploaded_image(image)
        if img_np is None or face_location is None:
            resp = {
                "status": False,
                "message": "No face detected"
            }
            cache_json_event(f"fiass:face:search:req:{req_id}", {"status": False, "reason": "no_face", "response": resp}, 3600)
            return resp

        blur_err = blur_rejection_payload(img_np, face_location, FACE_MIN_BLUR_VARIANCE)
        if blur_err:
            cache_json_event(f"fiass:face:search:req:{req_id}", {"status": False, "reason": "blur", "response": blur_err}, 3600)
            return blur_err

        encoding = get_face_encoding(img_np, face_location)
        if encoding is None:
            resp = {
                "status": False,
                "message": "Could not extract face features"
            }
            cache_json_event(f"fiass:face:search:req:{req_id}", {"status": False, "reason": "encoding_failed", "response": resp}, 3600)
            return resp

        # Same policy as get-face-embedding: high-confidence + margin-safe.
        results, stats = search_face_with_margin(
            encoding, threshold_override=SEARCH_HIGH_CONFIDENCE
        )

        if results and stats.get("margin") is not None and stats.get("top2_similarity") is not None:
            if stats["margin"] < FACE_MATCH_MIN_MARGIN:
                logger.warning(
                    f"[search][margin] too close: top1={stats.get('top1_user')}@{stats.get('top1_similarity')} "
                    f"top2={stats.get('top2_user')}@{stats.get('top2_similarity')} margin={stats['margin']:.4f}"
                )
                results = []

        if not results:
            resp = {
                "status": False,
                "message": "No matching face found or confidence below minimum",
                "threshold": SEARCH_HIGH_CONFIDENCE,
                "margin_required": FACE_MATCH_MIN_MARGIN,
                "top1_similarity": stats.get("top1_similarity"),
                "top2_similarity": stats.get("top2_similarity"),
                "margin_observed": stats.get("margin"),
                "appearance_variation": False
            }
            cache_json_event(f"fiass:face:search:req:{req_id}", {"status": False, "reason": "no_match", "response": resp}, 3600)
            return resp

        match = results[0]
        resp = {
            "status": True,
            "match": {
                "user_id": match["user_id"],
                "similarity": round(match["similarity"], 4),
                "margin": round(stats["margin"], 4) if stats.get("margin") is not None else None,
                "top2_similarity": round(stats["top2_similarity"], 4) if stats.get("top2_similarity") is not None else None,
            },
            "message": f"Face matched with {match['similarity']*100:.1f}% confidence",
            "appearance_variation": False,
            "threshold": SEARCH_HIGH_CONFIDENCE,
            "margin_required": FACE_MATCH_MIN_MARGIN,
        }
        cache_payload = {
            "endpoint": "search",
            "request_id": req_id,
            "status": True,
            "user_id": match["user_id"],
            "similarity": round(match["similarity"], 4),
            "margin": stats.get("margin"),
            "top2_similarity": stats.get("top2_similarity"),
            "appearance_variation": False,
            "threshold": SEARCH_HIGH_CONFIDENCE,
        }
        cache_json_event(f"fiass:face:search:req:{req_id}", cache_payload, 3600)
        cache_json_event("fiass:face:search:last", cache_payload, 86400)
        cache_json_event(f"fiass:face:user:{match['user_id']}:last_match", cache_payload, 86400)
        return resp

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        cache_json_event(f"fiass:face:search:req:{req_id}", {"status": False, "reason": "exception", "message": str(e)}, 3600)
        raise HTTPException(500, detail=str(e))

@app.post("/face/search_descriptor")
async def search_descriptor_endpoint(data: FaceData):
    """Search by raw 128-dim descriptor array (from ERP/client-side face-api.js)."""
    req_id = str(uuid.uuid4())
    try:
        arr = np.array(data.faceData, dtype=np.float32)
        if arr.shape[0] != DIM:
            raise HTTPException(400, f"Invalid vector length. Expected {DIM}, got {arr.shape[0]}.")

        normalized = arr / np.linalg.norm(arr)
        results = search_face(normalized, k=1)

        if not results:
            resp = {
                "status": False,
                "message": "No matching face found",
                "threshold": SEARCH_FACE_MATCH_THRESHOLD
            }
            cache_json_event(f"fiass:face:search-descriptor:req:{req_id}", {"status": False, "reason": "no_match", "response": resp}, 3600)
            return resp

        match = results[0]
        resp = {
            "status": True,
            "match": {
                "user_id": match["user_id"],
                "similarity": round(match["similarity"], 4)
            },
            "message": f"Face matched with {match['similarity']*100:.1f}% confidence"
        }
        cache_payload = {
            "endpoint": "search_descriptor",
            "request_id": req_id,
            "status": True,
            "user_id": match["user_id"],
            "similarity": round(match["similarity"], 4),
        }
        cache_json_event(f"fiass:face:search-descriptor:req:{req_id}", cache_payload, 3600)
        cache_json_event("fiass:face:search-descriptor:last", cache_payload, 86400)
        cache_json_event(f"fiass:face:user:{match['user_id']}:last_match", cache_payload, 86400)
        return resp

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Search descriptor error: {str(e)}", exc_info=True)
        cache_json_event(f"fiass:face:search-descriptor:req:{req_id}", {"status": False, "reason": "exception", "message": str(e)}, 3600)
        raise HTTPException(500, detail=str(e))

@app.post("/face/diagnose")
async def diagnose_endpoint(image: UploadFile = File(...)):
    """Admin/diagnostic endpoint: return the top-K nearest face neighbours for
    an uploaded image **without applying any similarity threshold**. Useful to
    find which user_id(s) a face is colliding with when /face/add_face rejects
    with `similar_face_exists`.
    Response shape:
      { status, top_k: [{rank, user_id, similarity, distance}], quality, pose }
    """
    req_id = str(uuid.uuid4())
    try:
        img_np, face_location = await process_uploaded_image(image)
        if img_np is None or face_location is None:
            return {"status": False, "message": "No face detected"}

        encoding = get_face_encoding(img_np, face_location)
        if encoding is None:
            return {"status": False, "message": "Could not extract face features"}

        with faiss_lock:
            if index.ntotal == 0:
                return {"status": False, "message": "Empty index", "top_k": []}
            distances, ids = index.search(encoding[None, :], 10)

        top_k = []
        for rank, (idx, dist) in enumerate(zip(ids[0], distances[0])):
            if idx == -1:
                continue
            uid = id_to_user_id.get(int(idx))
            if not uid:
                continue
            top_k.append({
                "rank": int(rank),
                "user_id": uid,
                "similarity": round(float(1.0 - dist), 4),
                "distance": round(float(dist), 6),
            })

        # Quick pose + quality snapshot (best-effort, doesn't reject).
        h, w = img_np.shape[:2]
        landmarks_list = face_recognition.face_landmarks(
            img_np, face_locations=[face_location]
        )
        lm = landmarks_list[0] if landmarks_list else None
        pose = estimate_face_pose(img_np, face_location, lm) if lm else {"yaw": None, "pitch": None, "roll": None}
        blur_var = face_crop_laplacian_variance(img_np, face_location)

        payload = {
            "status": True,
            "request_id": req_id,
            "top_k": top_k,
            "pose": pose,
            "quality": {
                "image_width": int(w),
                "image_height": int(h),
                "blur_variance": round(float(blur_var), 2),
            },
            "thresholds": {
                "search_face_match": SEARCH_FACE_MATCH_THRESHOLD,
                "search_high_confidence": SEARCH_HIGH_CONFIDENCE,
                "add_face_reject": ADD_FACE_REJECT_THRESHOLD,
                "face_match_min_margin": FACE_MATCH_MIN_MARGIN,
            },
        }
        cache_json_event(f"fiass:face:diagnose:req:{req_id}", payload, 3600)
        return payload
    except Exception as e:
        logger.error(f"Diagnose error: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.delete("/face/force_delete/{user_id}")
def force_delete_user(user_id: str, admin_key: str = ""):
    """Admin: remove all faiss+PG rows for a user_id. Protected via ADMIN_API_KEY.
    Use this to clean up orphan rows surfaced by /face/diagnose (e.g. stale
    user_ids left after a Mongo account was deleted)."""
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected or admin_key != expected:
        raise HTTPException(403, detail="admin_key invalid")
    ok, message = delete_face_descriptor(user_id)
    return {"status": ok, "user_id": user_id, "message": message}


@app.get("/face/check")
def check():
    return {"status": {"ok": 1, "server": "working"}}

# ------------------------------------------------------------------------------
# Startup/Shutdown
# ------------------------------------------------------------------------------

async def cleanup_old_uploads():
    """Delete uploaded images older than 24 hours from static/uploads/."""
    import asyncio
    import glob
    while True:
        try:
            cutoff = time.time() - 86400  # 24 hours
            for filepath in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
                if os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old upload: {filepath}")
        except Exception as e:
            logger.error(f"Upload cleanup error: {e}")
        await asyncio.sleep(3600)  # Run every hour

@app.on_event("startup")
def on_startup():
    """Initialize services on startup."""
    import asyncio
    asyncio.get_event_loop().create_task(cleanup_old_uploads())

    max_attempts = 10
    delay = 2  # seconds

    for attempt in range(max_attempts):
        conn = None
        try:
            conn = get_db_conn()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            put_db_conn(conn)
            logger.info("Connected to PostgreSQL.")
            initialize_faiss_index()
            break
        except Exception as e:
            if conn is not None:
                put_db_conn(conn)
            logger.warning(f"PostgreSQL not ready (attempt {attempt + 1}/{max_attempts}): {e}")
            time.sleep(delay)
    else:
        logger.error("Failed to connect to PostgreSQL.")
        raise RuntimeError("PostgreSQL connection failed.")

    # Wait for Redis (skipped when using fakeredis for local dev)
    if _redis_is_fake:
        logger.info("Using in-memory Redis, no TCP connection.")
    else:
        for attempt in range(max_attempts):
            try:
                r.ping()
                if REDIS_URL:
                    logger.info("Connected to Redis via REDIS_URL/KV_URL.")
                else:
                    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}.")
                break
            except Exception as e:
                logger.warning(f"Redis not ready (attempt {attempt + 1}/{max_attempts}): {e}")
                time.sleep(delay)
        else:
            logger.warning("Failed to connect to Redis. Falling back to in-memory fakeredis.")
            import fakeredis
            globals()["r"] = fakeredis.FakeRedis()

@app.on_event("shutdown")
def on_shutdown():
    """Cleanup resources on shutdown."""
    pool.closeall()
    logger.info("Database connection pool closed.")