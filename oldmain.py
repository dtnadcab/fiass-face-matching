from fastapi import FastAPI, HTTPException, Request, File, UploadFile
import faiss
import numpy as np
import psycopg2
from pydantic import BaseModel
import logging
from typing import List, Dict, Optional, Tuple
import redis
import pickle
import hashlib
import os
import time
import face_recognition
from PIL import Image, ExifTags, ImageOps
import io
from fastapi.staticfiles import StaticFiles
import uuid
import math

# ------------------------------------------------------------------------------
# Env Variables and Database/Redis Connection
# ------------------------------------------------------------------------------

POSTGRES_HOST = os.getenv("DB_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("DB_PORT", 5432))
POSTGRES_DB = os.getenv("DB_NAME", "postgres")
POSTGRES_USER = os.getenv("DB_USER", "postgres")
POSTGRES_PASS = os.getenv("DB_PASS", "postgres")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

conn = psycopg2.connect(
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASS,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
)

conn.autocommit = True
cur = conn.cursor()

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# ------------------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=" %(asctime)s - %(levelname)s - %(message)s ",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()


def face_hash(vector: np.ndarray) -> str:
    """Create a unique hash for the face vector to use as a cache key."""
    return hashlib.md5(vector.tobytes()).hexdigest()


# ------------------------------------------------------------------------------
# FAISS Setup
# ------------------------------------------------------------------------------

dim = 128
index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
id_to_user_id: Dict[int, str] = {}

def initialize_faiss_index():
    """Load face descriptors from Postgres into FAISS index on startup."""
    index.reset()
    id_to_user_id.clear()

    # Ensure the table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            face_descriptor FLOAT4[]
        );
    """)
    conn.commit()

    # Load the data
    cur.execute("SELECT id, user_id, face_descriptor FROM faces;")
    rows = cur.fetchall()

    if rows:
        vectors = np.array([row[2] for row in rows], dtype=np.float32)
        ids = np.array([row[0] for row in rows], dtype='int64')
        index.add_with_ids(vectors, ids)
        id_to_user_id.update({row[0]: row[1] for row in rows})
        logger.info(f"✅ Loaded {len(rows)} face descriptors into FAISS.")
    else:
        logger.info("⚠️ No face descriptors in database.")

def search_face(vector: np.ndarray, k: int = 5, threshold: float = 15.0) -> List[Dict[str, float]]:
    """Search for nearest faces in FAISS, returning only matches with distance <= 0.05."""
    assert vector.shape == (dim,), f"Must be {dim}-dim vector."
    
    vector_hash = face_hash(vector)
    cache_key = f"face:{vector_hash}:{k}"
    
    logger.info(f"[SEARCH] Vector hash: {vector_hash}, Shape: {vector.shape}")
    logger.info(f"[SEARCH] FAISS index size: {index.ntotal}")

    cached = r.get(cache_key)
    if cached:
        logger.info(f"[CACHE] Hit for key: {cache_key}")
        return pickle.loads(cached)

    logger.info(f"[CACHE] Miss for key: {cache_key}")
    
    if index.ntotal == 0:
        logger.warning("[FAISS] No vectors in index. Returning empty result.")
        return []

    distances, ids = index.search(vector[None, :], k * 2)
    
    logger.info(f"[FAISS] Raw distances: {distances[0]}")
    logger.info(f"[FAISS] Raw IDs: {ids[0]}")

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        if idx == -1:
            logger.debug(f"[SKIP] idx=-1 (invalid)")
            continue
        if dist > 0.1478:
            logger.debug(f"[SKIP] Distance {dist:.6f} > 0.05 for index {idx}")
            continue
        if idx not in id_to_user_id:
            logger.debug(f"[SKIP] idx={idx} not found in id_to_user_id")
            continue
        
        user_id = id_to_user_id[idx]
        logger.info(f"[MATCH] user_id={user_id}, idx={idx}, distance={dist:.6f}")
        results.append({
            "user_id": user_id,
            "distance": float(dist)
        })

    logger.info(f"[RESULT] Matched {len(results)} faces within threshold.")
    
    # Cache filtered results
    r.setex(cache_key, 60 * 5, pickle.dumps(results))
    logger.info(f"[CACHE] Set result for key: {cache_key} (expires in 5 min)")

    return results

def face_exists(vector: np.ndarray, threshold: float = 15.0) -> Tuple[bool, Optional[str]]:
    """Check if a face already exists in the index."""
    results = search_face(vector, k=1, threshold=threshold)
    if not results:
        return False, None, None
    nearest = results[0]
    if nearest["distance"] <= threshold:
        return True, nearest["user_id"], nearest["distance"]
    return False, None, None

def add_face(user_id: str, vector: np.ndarray, threshold: float = 15.0) -> Tuple[Optional[int], str]:
    """Add face to Postgres & FAISS only if not too similar and user_id is new."""
    assert vector.shape == (dim,), f"Must be {dim}-dim vector."

    # Clear Redis cache for this face
    cache_key = f"face:{face_hash(vector)}:*"
    for key in r.keys(cache_key):
        r.delete(key)

    # Check if user_id already exists
    cur.execute("SELECT id FROM faces WHERE user_id = %s", (user_id,))
    existing = cur.fetchone()
    if existing:
        return None, f"❌ Face already exists for user_id {user_id}"

    # Check for similar face
    exists, existing_user, distance = face_exists(vector, threshold)
    if exists and distance <= 0.05:
        return None, f"❌ Similar face already exists for user {existing_user} (distance={distance:.4f})"

    try:
        # Insert new face
        cur.execute(
            "INSERT INTO faces (user_id, face_descriptor) VALUES (%s, %s) RETURNING id",
            (user_id, vector.tolist())
        )
        face_id = cur.fetchone()[0]
        index.add_with_ids(vector[None, :], np.array([face_id]))
        id_to_user_id[face_id] = user_id

        logger.info(f"✅ Added new face for {user_id}")
        return face_id, f"✅ Face added for {user_id}"

    except psycopg2.Error as e:
        logger.error(f"❌ DB error while adding face for {user_id}: {str(e)}")
        return None, f"❌ Database error: {str(e)}"

def delete_face_descriptor(user_id: str) -> Tuple[bool, str]:
    # Step 1: Get FAISS ID from user_id
    cur.execute("SELECT id FROM faces WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    if not row:
        return False, f"❌ No face found for user_id {user_id}"

    face_id = row[0]

    # Step 2: Remove from FAISS index
    try:
        index.remove_ids(np.array([face_id], dtype=np.int64))
        id_to_user_id.pop(face_id, None)  # Remove from in-memory mapping
    except Exception as e:
        return False, f"❌ Error removing from FAISS index: {str(e)}"

    # Step 3: Remove from Redis cache
    for key in r.scan_iter(f"face:*:*"):
        value = r.get(key)
        if value and user_id.encode() in value:
            r.delete(key)

    # Step 4: Delete from DB
    try:
        cur.execute("DELETE FROM faces WHERE id = %s", (face_id,))
        return True, f"✅ Deleted face for user_id {user_id}"
    except Exception as e:
        return False, f"❌ DB error: {str(e)}"

def correct_exif_orientation(img: Image.Image) -> Image.Image:
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
    img_np = np.array(img)
    face_landmarks_list = face_recognition.face_landmarks(img_np)

    if not face_landmarks_list:
        print("No face detected.")
        return img

    landmarks = face_landmarks_list[0]
    if "left_eye" not in landmarks or "right_eye" not in landmarks:
        return img

    left_eye = np.mean(landmarks["left_eye"], axis=0)
    right_eye = np.mean(landmarks["right_eye"], axis=0)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    print(f"Detected face tilt angle: {angle:.2f}°")

    # Rotate only if face is sideways (more than ±15°)
    if abs(angle) > 15:
        img = img.rotate(-angle, expand=True)

    return img

async def get_face_array(request: Request, image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img = rotate_image_if_face_sideways(img)

        # Save rotated image
        file_ext = image.filename.split('.')[-1].lower()
        file_id = f"{uuid.uuid4()}.{file_ext}"
    
        saved_path = os.path.join(UPLOAD_FOLDER, file_id)
        img.save(saved_path)
        # Build absolute URL
        rotated_image_url = f"{request.base_url}static/uploads/{file_id}"
        # logger.info(f"img url::{rotated_image_url}")
        # Extract face descriptor
        img_np = np.array(img)
        face_locations = face_recognition.face_locations(img_np)
        if not face_locations:
            # raise HTTPException(status_code=404, detail="No face found in image.")
            return {
                "status": False,
                "message": "No face detected in file",
                "rotated_image_url": rotated_image_url
            }
        face_encodings = face_recognition.face_encodings(img_np, face_locations)
        descriptor = face_encodings[0].tolist()
        
        return descriptor

        # return {
        #     "status": True,
        #     "descriptor": descriptor,
        #     "message": f"Face descriptor extracted. {len(descriptor)} values.",
        #     "rotated_image_url": rotated_image_url
        # }

    except HTTPException as he:
        raise he
    except Exception as e:
        return {
            "status": False,
            "message": str(e)
        }

# ------------------------------------------------------------------------------
# Pydantic Model
# ------------------------------------------------------------------------------

class FaceData(BaseModel):
    faceData: List[float]

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    try:
        # contents = await image.read()
        # img = Image.open(io.BytesIO(contents)).convert("RGB")
        # img = ImageOps.exif_transpose(img)
        # img = rotate_image_if_face_sideways(img)

        # # Save rotated image
        # file_ext = image.filename.split('.')[-1].lower()
        # file_id = f"{uuid.uuid4()}.{file_ext}"
    
        # saved_path = os.path.join(UPLOAD_FOLDER, file_id)
        # img.save(saved_path)
        # # Build absolute URL
        # rotated_image_url = f"{request.base_url}static/uploads/{file_id}"
        # # logger.info(f"img url::{rotated_image_url}")
        # # Extract face descriptor
        # img_np = np.array(img)
        # face_locations = face_recognition.face_locations(img_np)
        # if not face_locations:
        #     # raise HTTPException(status_code=404, detail="No face found in image.")
        #     return {
        #         "status": False,
        #         "message": "No face detected in file",
        #         "rotated_image_url": rotated_image_url
        #     }
        # face_encodings = face_recognition.face_encodings(img_np, face_locations)
        # descriptor = face_encodings[0].tolist()        
        # logger.info(f"des::{descriptor}")
        descriptor = await get_face_array(request, image)
        arr = np.array(descriptor, dtype=np.float32)
        if arr.shape[0] != dim:
          raise HTTPException(400, f"Invalid face vector length. Expected {dim}, got {arr.shape[0]}.")
        results = search_face(arr, 5, 15)
        if not results:
          return {"matches": [], "msg": "No matching face found."}
        return {"matches": results}

        # return {
        #     "status": True,
        #     "descriptor": descriptor,
        #     "message": f"Face descriptor extracted. {len(descriptor)} values.",
        #     "rotated_image_url": rotated_image_url
        # }

    except HTTPException as he:
        raise he
    except Exception as e:
        return {
            "status": False,
            "message": str(e)
        }

@app.post("/face/add_face/{user_id}")
async def add_face_endpoint(request:Request, user_id: str, image: UploadFile = File(...)):
    descriptor = await get_face_array(request, image)
    logger.info(f"des: {descriptor}")
    arr = np.array(descriptor, dtype=np.float32)
    if arr.shape[0] != dim:
        raise HTTPException(400, f"Invalid face vector length. Expected {dim}, got {arr.shape[0]}.")
    face_id, message = add_face(user_id, arr)
    if face_id is None:
        raise HTTPException(409, message)
    return {"msg": message, "id": face_id}

@app.post("/face/search")
async def search_endpoint(face: FaceData, k: int = 5, threshold: float = 15.0):
    arr = np.array(face.faceData, dtype=np.float32)
    if arr.shape[0] != dim:
        raise HTTPException(400, f"Invalid face vector length. Expected {dim}, got {arr.shape[0]}.")
    results = search_face(arr, k, threshold)
    if not results:
        return {"matches": [], "msg": "No matching face found."}
    return {"matches": results}

@app.get("/face/check")
def check():
    return {"status": {"ok": 1, "server": "working"}}

def log_db_diagnostics():
    # Log current database
    # cur.execute("DROP TABLE faces;")
    cur.execute("SELECT current_database();")
    logger.info(f"✅ Connected DB:{cur.fetchone()}")

    # Log current schema
    cur.execute("SHOW search_path;")
    logger.info(f"✅ Search Path:{cur.fetchone()}")

    # Log all tables in all schemas


@app.on_event("startup")
def on_startup():
    """Wait for DB/Redis, then initialize FAISS index."""
    max_attempts = 10
    delay = 2  # seconds
    logger.info("Connecting to PostgreSQL....")
    # Wait for PostgreSQL
    for attempt in range(max_attempts):
        try:
            cur.execute("SELECT 1")
            log_db_diagnostics()
            logger.info("Connected to PostgreSQL.")
            initialize_faiss_index()
            break
        except Exception as e:
            logger.warning(f"PostgreSQL not ready (attempt {attempt + 1}/{max_attempts}): {e}")
            time.sleep(delay)
    else:
        logger.error("Failed to connect to PostgreSQL after multiple attempts.")
        raise RuntimeError("PostgreSQL connection failed.")

    # Wait for Redis
    for attempt in range(max_attempts):
        try:
            r.ping()
            logger.info("Connected to Redis.")
            break
        except Exception as e:
            logger.warning(f"Redis not ready (attempt {attempt + 1}/{max_attempts}): {e}")
            time.sleep(delay)
    else:
        logger.error("Failed to connect to Redis after multiple attempts.")
        raise RuntimeError("Redis connection failed.")

    # Now initialize the FAISS index


@app.on_event("shutdown")
def on_shutdown():
    """Clean-up resources on shutdown."""
    cur.close()
    conn.close()
    logger.info("Database connection closed.")
