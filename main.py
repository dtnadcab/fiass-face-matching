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

def should_adjust_threshold(face_location, img_np: np.ndarray) -> bool:
    """Detect appearance variations (glasses, beard, cap, mask) that need threshold adjustment."""
    try:
        landmarks_list = face_recognition.face_landmarks(img_np, [face_location])
        if not landmarks_list:
            return True  # Can't get landmarks → face partially occluded → lower threshold
        landmarks = landmarks_list[0]

        # ── Glasses detection ──
        left_eye = landmarks.get('left_eye', [])
        right_eye = landmarks.get('right_eye', [])
        left_eyebrow = landmarks.get('left_eyebrow', [])
        right_eyebrow = landmarks.get('right_eyebrow', [])

        if left_eye and right_eye:
            all_eye_pts = left_eye + right_eye
            y_min = max(0, min(p[1] for p in all_eye_pts) - 8)
            y_max = min(img_np.shape[0], max(p[1] for p in all_eye_pts) + 8)
            x_min = max(0, min(p[0] for p in all_eye_pts) - 5)
            x_max = min(img_np.shape[1], max(p[0] for p in all_eye_pts) + 5)

            if y_max > y_min and x_max > x_min:
                eye_roi = img_np[y_min:y_max, x_min:x_max]
                gray_roi = np.mean(eye_roi, axis=2) if eye_roi.ndim == 3 else eye_roi

                # Dark frames / sunglasses: very low mean brightness in eye region
                if np.mean(gray_roi) < 65:
                    return True

                # Spectacle frames: high edge contrast around eyes
                edges = np.abs(np.diff(gray_roi, axis=1))
                if np.mean(edges) > 25:
                    return True

                # Bridge check: dark strip between the two eyes
                if left_eye and right_eye:
                    bridge_x1 = max(p[0] for p in left_eye)
                    bridge_x2 = min(p[0] for p in right_eye)
                    if bridge_x2 > bridge_x1:
                        bridge_y = int(np.mean([p[1] for p in left_eye + right_eye]))
                        bridge_strip = img_np[max(0, bridge_y-3):bridge_y+3, bridge_x1:bridge_x2]
                        if bridge_strip.size > 0 and np.mean(bridge_strip) < 80:
                            return True

        # ── Eyebrow-to-forehead distance (cap / hat detection) ──
        top, right, bottom, left = face_location
        if left_eyebrow and right_eyebrow:
            brow_y = min(min(p[1] for p in left_eyebrow), min(p[1] for p in right_eyebrow))
            forehead_ratio = (brow_y - top) / max(bottom - top, 1)
            if forehead_ratio < 0.08:
                return True

        # ── Beard / facial hair detection ──
        chin = landmarks.get('chin', [])
        nose_tip = landmarks.get('nose_tip', [])
        if chin and nose_tip and len(chin) >= 9:
            chin_bottom = chin[8][1]
            nose_bottom = nose_tip[0][1]
            lower_face = img_np[nose_bottom:chin_bottom, left:right]
            if lower_face.size > 0:
                gray_lower = np.mean(lower_face, axis=2) if lower_face.ndim == 3 else lower_face
                # Beard tends to make the lower face darker and more textured
                if np.mean(gray_lower) < 90 or np.std(gray_lower) > 45:
                    return True

        return False
    except Exception as e:
        logger.debug(f"Threshold adjustment check failed: {e}")
        return False

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

# Constants at the top of your file
ADD_FACE_REJECT_THRESHOLD = float(os.getenv("ADD_FACE_REJECT_THRESHOLD", "0.895"))
SEARCH_FACE_MATCH_THRESHOLD = float(os.getenv("SEARCH_FACE_MATCH_THRESHOLD", "0.84"))
SEARCH_HIGH_CONFIDENCE = float(os.getenv("SEARCH_HIGH_CONFIDENCE", "0.92"))
# Only used when should_adjust_threshold() is true (second pass); stricter default than 0.68
SEARCH_APPEARANCE_VARIATION = float(os.getenv("SEARCH_APPEARANCE_VARIATION", "0.72"))

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
    """Search for faces with adaptive threshold for appearance variations"""
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
        
        # Strict search first; loosen only when appearance-variation heuristics fire (never always-loosen).
        results = search_face(arr, threshold_override=SEARCH_FACE_MATCH_THRESHOLD)
        has_variation = False
        
        if not results:
            has_variation = should_adjust_threshold(best_face, det_img)
            if has_variation:
                results = search_face(arr, threshold_override=SEARCH_APPEARANCE_VARIATION)
        
        elapsed_ms = int((time.time() - t0) * 1000)
        
        if not results:
            resp = {
                "status": False,
                "matches": [],
                "message": "No matching face found",
                "threshold": SEARCH_APPEARANCE_VARIATION if has_variation else SEARCH_FACE_MATCH_THRESHOLD,
                "appearance_variation": has_variation,
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
                "confidence": best_match.get("confidence", "medium")
            },
            "message": f"Face matched with {best_match['similarity']*100:.1f}% confidence",
            "appearance_variation": has_variation,
            "elapsed_ms": elapsed_ms,
            "rotated_image_url": rotated_image_url
        }
        cache_payload = {
            "endpoint": "get-face-embedding",
            "status": True,
            "request_id": req_id,
            "user_id": best_match["user_id"],
            "similarity": round(best_match["similarity"], 4),
            "appearance_variation": has_variation,
            "elapsed_ms": elapsed_ms,
        }
        cache_json_event(f"fiass:face:get-face-embedding:req:{req_id}", cache_payload, ttl_sec=3600)
        cache_json_event("fiass:face:get-face-embedding:last", cache_payload, ttl_sec=86400)
        cache_json_event(f"fiass:face:user:{best_match['user_id']}:last_match", cache_payload, ttl_sec=86400)
        cache_face_details(
            f"fiass:face:user:{best_match['user_id']}:details:last",
            descriptor=arr,
            image_bytes=contents,
            ttl_sec=86400,
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
    try:
        # Keep original image bytes for Redis debugging snapshot.
        await image.seek(0)
        raw_image_bytes = await image.read()
        await image.seek(0)

        # Process image and get face location
        img_np, face_location = await process_uploaded_image(image)
        if img_np is None or face_location is None:
            raise HTTPException(400, "No face detected or face too small")

        blur_err = blur_rejection_payload(img_np, face_location, FACE_MIN_BLUR_VARIANCE)
        if blur_err:
            raise HTTPException(400, blur_err.get("message", "Image too blurry"))

        # Get face encoding
        encoding = get_face_encoding(img_np, face_location)
        if encoding is None:
            raise HTTPException(400, "Could not extract face features")

        # Check for existing similar faces (89.5% threshold)
        exists, existing_user, similarity = face_exists(encoding)
        if exists:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "similar_face_exists",
                    "message": "Face already exists in system",
                    "existing_user": existing_user,
                    "similarity": round(similarity, 3),
                    "threshold": ADD_FACE_REJECT_THRESHOLD
                }
            )

        # Add to database
        face_id, message = add_face(user_id, encoding)
        cache_face_details(
            f"fiass:face:user:{user_id}:details:registered",
            descriptor=np.array(encoding, dtype=np.float32),
            image_bytes=raw_image_bytes,
            ttl_sec=7 * 86400,
            meta={"endpoint": "add_face", "user_id": user_id, "face_id": face_id},
        )
        return {
            "status": True,
            "id": face_id,
            "message": message
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

        exists, existing_user, similarity = face_exists(normalized)
        if exists:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "similar_face_exists",
                    "message": "Face already exists in system",
                    "existing_user": existing_user,
                    "similarity": round(similarity, 3),
                    "threshold": ADD_FACE_REJECT_THRESHOLD
                }
            )

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

        has_variation = should_adjust_threshold(face_location, img_np)
        effective_threshold = SEARCH_APPEARANCE_VARIATION if has_variation else SEARCH_FACE_MATCH_THRESHOLD
        results = search_face(encoding, k=1, threshold_override=effective_threshold)
        
        if not results:
            resp = {
                "status": False,
                "message": "No matching face found",
                "threshold": effective_threshold,
                "appearance_variation": has_variation
            }
            cache_json_event(f"fiass:face:search:req:{req_id}", {"status": False, "reason": "no_match", "response": resp}, 3600)
            return resp

        match = results[0]
        resp = {
            "status": True,
            "match": {
                "user_id": match["user_id"],
                "similarity": round(match["similarity"], 4)
            },
            "message": f"Face matched with {match['similarity']*100:.1f}% confidence",
            "appearance_variation": has_variation
        }
        cache_payload = {
            "endpoint": "search",
            "request_id": req_id,
            "status": True,
            "user_id": match["user_id"],
            "similarity": round(match["similarity"], 4),
            "appearance_variation": has_variation,
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