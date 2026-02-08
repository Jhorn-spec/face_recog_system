# facecore.py (optimized)
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --------------------
# Config
# --------------------
BASE_DIR = Path(__file__).resolve().parent
IMG_DB = BASE_DIR / "img_db"
EMB_DB = BASE_DIR / "embeddings.npz"
EMB_META = BASE_DIR / "embeddings.json"

# You can switch between "ArcFace" and "SFace"
MODEL_NAME = "SFace"          # faster than ArcFace
DETECTOR_BACKEND = "opencv"   # much faster than retinaface
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.45           # recognition threshold (used at runtime)
DUPLICATE_THRESHOLD = 0.9  # stricter threshold to block duplicate registrations

IMG_DB.mkdir(parents=True, exist_ok=True)

# --------------------
# In-memory cache (avoid disk I/O each call)
# --------------------
_EMBS_CACHE: Optional[np.ndarray] = None
_META_CACHE: Optional[Dict[str, Any]] = None
_NN_CACHE: Optional[NearestNeighbors] = None


def _invalidate_cache():
    global _NN_CACHE
    _NN_CACHE = None   # NN depends on embeddings; rebuild when needed


# --------------------
# Image utilities
# --------------------
def enhance_image(frame_bgr: np.ndarray) -> np.ndarray:
    """Lighten / enhance contrast in low-light images."""
    try:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        ycrcb = cv2.merge((y, cr, cb))
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return enhanced
    except Exception:
        return frame_bgr


def _resize_for_detection(frame_bgr: np.ndarray, max_width: int = 640) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def _crop_face_with_deepface(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Extract first detected face using DeepFace.extract_faces.
    Returns a BGR cropped face or original frame if extraction fails.
    """
    try:
        frame_bgr = _resize_for_detection(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        faces = DeepFace.extract_faces(
            img_path=frame_rgb,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )

        if not faces:
            return frame_bgr

        face_rgb = np.asarray(faces[0]["face"])
        if face_rgb.max() <= 1.0:
            face_rgb = (face_rgb * 255).astype(np.uint8)
        else:
            face_rgb = face_rgb.astype(np.uint8)

        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        return face_bgr
    except Exception as e:
        print("[_crop_face_with_deepface] Fallback:", e)
        return frame_bgr

# --------------------
# Embedding I/O
# --------------------
def _save_embeddings(embs: np.ndarray, meta: Dict[str, Any]) -> None:
    global _EMBS_CACHE, _META_CACHE
    np.savez_compressed(EMB_DB, embs=embs.astype(np.float32))
    with open(EMB_META, "w") as f:
        json.dump(meta, f, indent=2)

    _EMBS_CACHE = embs.astype(np.float32)
    _META_CACHE = meta
    _invalidate_cache()


def _load_embeddings() -> Tuple[np.ndarray, Dict[str, Any]]:
    global _EMBS_CACHE, _META_CACHE

    if _EMBS_CACHE is not None and _META_CACHE is not None:
        return _EMBS_CACHE, _META_CACHE

    if EMB_DB.exists() and EMB_META.exists():
        data = np.load(EMB_DB, allow_pickle=True)
        embs = data["embs"].astype(np.float32)
        with open(EMB_META, "r") as f:
            meta = json.load(f)
        _EMBS_CACHE = embs
        _META_CACHE = meta
        return embs, meta

    _EMBS_CACHE = np.empty((0, 0), dtype=np.float32)
    _META_CACHE = {}
    return _EMBS_CACHE, _META_CACHE


def _append_embedding(new_emb: np.ndarray, id_str: str, image_path: str) -> None:
    embs, meta = _load_embeddings()

    new_emb = np.asarray(new_emb, dtype=np.float32).reshape(1, -1)

    if embs.size == 0:
        embs = new_emb
    else:
        if embs.shape[1] != new_emb.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: existing={embs.shape[1]}, new={new_emb.shape[1]}"
            )
        embs = np.vstack([embs, new_emb])

    meta[id_str] = {
        "index": int(embs.shape[0] - 1),
        "image_path": str(image_path),
        "ts": time.time(),
    }
    _save_embeddings(embs, meta)


# --------------------
# Embedding extraction
# --------------------
def get_embedding_from_image_bgr(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        frame_bgr = enhance_image(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        rep = DeepFace.represent(
            img_path=frame_rgb,
            model_name=MODEL_NAME,
            detector_backend="skip",  # cropped already
            enforce_detection=False,
        )

        if isinstance(rep, list) and len(rep) > 0 and "embedding" in rep[0]:
            return np.array(rep[0]["embedding"], dtype=np.float32)

        if isinstance(rep, dict) and "embedding" in rep:
            return np.array(rep["embedding"], dtype=np.float32)

    except Exception as e:
        print("[get_embedding_from_image_bgr] Error:", e)

    return None

def _search_embedding(
    emb: np.ndarray, top_k: int, threshold: float
) -> Dict[str, Any]:
    """
    Internal helper: given an embedding, search DB and return the usual result dict.
    Does NOT do any detection or embedding extraction.
    """
    result = {"id": "", "success": False, "message": "", "matches": []}

    nn, embs, meta = build_nn_index()
    if embs.size == 0:
        result["message"] = "no registered users"
        return result

    if nn is not None:
        from sklearn.exceptions import NotFittedError
        try:
            dists, idxs = nn.kneighbors(
                emb.reshape(1, -1), n_neighbors=min(top_k, embs.shape[0])
            )
        except NotFittedError:
            result["message"] = "index not fitted"
            return result

        dists = dists.flatten()
        idxs = idxs.flatten()

        for dist, idx in zip(dists, idxs):
            sim = 1.0 - float(dist)
            found_id = None
            for k, v in meta.items():
                if int(v["index"]) == int(idx):
                    found_id = k
                    break
            if found_id is None:
                continue
            if sim >= threshold:
                result["matches"].append(
                    {
                        "id": found_id,
                        "sim": sim,
                        "image_path": meta[found_id]["image_path"],
                    }
                )
    else:
        # brute-force fallback
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(emb)
        sims = np.dot(embs, emb) / np.where(norms == 0, 1e-6, norms)
        idxs = np.argsort(-sims)[:top_k]

        for idx in idxs:
            sim = float(sims[idx])
            found_id = None
            for k, v in meta.items():
                if int(v["index"]) == int(idx):
                    found_id = k
                    break
            if found_id and sim >= threshold:
                result["matches"].append(
                    {
                        "id": found_id,
                        "sim": sim,
                        "image_path": meta[found_id]["image_path"],
                    }
                )

    if result["matches"]:
        result["success"] = True
        result["id"] = result["matches"][0]["id"]
        result["message"] = "match found"
    else:
        result["message"] = "no match above threshold"

    return result


# --------------------
# Public API – Registration
# --------------------
def register_from_image(id_str: str, upload_image_path: str) -> Dict[str, Any]:
    result = {"success": False, "message": "", "id": id_str, "image_path": ""}

    try:
        dest_path = IMG_DB / f"{id_str}.jpg"
        if dest_path.exists():
            result["message"] = "id already exists"
            return result

        frame = cv2.imread(str(upload_image_path))
        if frame is None:
            result["message"] = "invalid image path"
            return result

        frame = enhance_image(frame)
        cropped_bgr = _crop_face_with_deepface(frame)

        emb = get_embedding_from_image_bgr(cropped_bgr)
        if emb is None:
            result["message"] = "no face embedding could be extracted"
            return result

        # --- duplicate check ---
        dup_result = _search_embedding(
            emb,
            top_k=1,
            threshold=DUPLICATE_THRESHOLD,
        )

        if dup_result.get("success") and dup_result.get("matches"):
            existing = dup_result["matches"][0]
            result["success"] = False
            result["message"] = (
                f"face already registered as '{existing['id']}' "
                f"(similarity {existing['sim']:.3f} ≥ duplicate threshold)"
            )
            result["image_path"] = existing["image_path"]
            return result

        # --- no duplicate: proceed with registration ---
        cv2.imwrite(str(dest_path), cropped_bgr)
        _append_embedding(emb, id_str, str(dest_path))

        result["success"] = True
        result["message"] = f"id {id_str} registered successfully"
        result["image_path"] = str(dest_path)
        return result

    except Exception as e:
        result["message"] = f"error: {e}"
        return result



def register_from_camera(
    id_str: str, cam_index: int = 0, max_attempts: int = 300
) -> Dict[str, Any]:
    result = {"success": False, "message": "", "id": id_str, "image_path": ""}

    dest_path = IMG_DB / f"{id_str}.jpg"
    if dest_path.exists():
        result["message"] = "id already exists"
        return result

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        result["message"] = "cannot open camera"
        return result

    attempts = 0
    try:
        while attempts < max_attempts:
            attempts += 1
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = enhance_image(frame)
            cropped_bgr = _crop_face_with_deepface(frame)
            emb = get_embedding_from_image_bgr(cropped_bgr)

            if emb is not None:
                # --- duplicate check ---
                dup_result = _search_embedding(
                    emb,
                    top_k=1,
                    threshold=DUPLICATE_THRESHOLD,
                )

                if dup_result.get("success") and dup_result.get("matches"):
                    existing = dup_result["matches"][0]
                    result["success"] = False
                    result["message"] = (
                        f"face already registered as '{existing['id']}' "
                        f"(similarity {existing['sim']:.3f} ≥ duplicate threshold)"
                    )
                    result["image_path"] = existing["image_path"]
                    break

                # --- not a duplicate: register ---
                cv2.imwrite(str(dest_path), cropped_bgr)
                _append_embedding(emb, id_str, str(dest_path))
                result["success"] = True
                result["message"] = f"id {id_str} registered"
                result["image_path"] = str(dest_path)
                break

            time.sleep(0.02)

        # Only override message if nothing ever succeeded and no duplicate message was set
        if not result["success"] and not result["message"]:
            result["message"] = "no face detected in camera during attempt window"

    finally:
        cap.release()

    return result


# --------------------
# Public API – Search
# --------------------
def build_nn_index():
    global _NN_CACHE
    embs, meta = _load_embeddings()
    if embs.size == 0 or len(meta) == 0:
        return None, embs, meta

    if _NN_CACHE is not None:
        return _NN_CACHE, embs, meta

    if SKLEARN_AVAILABLE:
        nn = NearestNeighbors(
            n_neighbors=min(5, embs.shape[0]),
            metric=DISTANCE_METRIC,
        )
        nn.fit(embs)
        _NN_CACHE = nn
        return nn, embs, meta

    return None, embs, meta


def find_match_from_image(
    image_bgr: np.ndarray,
    top_k: int = 3,
    threshold: float = THRESHOLD,
) -> Dict[str, Any]:
    """
    Given a full image (BGR), do detection + embedding + search.
    """
    image_bgr = enhance_image(image_bgr)
    cropped_bgr = _crop_face_with_deepface(image_bgr)
    emb = get_embedding_from_image_bgr(cropped_bgr)

    if emb is None:
        return {"id": "", "success": False, "message": "no embedding from input image", "matches": []}

    return _search_embedding(emb, top_k=top_k, threshold=threshold)


def find_match_from_face_crop(
    crop_bgr: np.ndarray,
    top_k: int = 3,
    threshold: float = THRESHOLD,
) -> Dict[str, Any]:
    """
    Given a *cropped* face image (BGR), return best match.
    Assumes the crop already contains the face; no detection is done here.
    """
    emb = get_embedding_from_image_bgr(crop_bgr)
    if emb is None:
        return {"id": "", "success": False, "message": "no embedding from crop", "matches": []}
    return _search_embedding(emb, top_k=top_k, threshold=threshold)


# --------------------
# Admin
# --------------------
def list_registered_users() -> List[str]:
    _, meta = _load_embeddings()
    return list(meta.keys())


def delete_user(id_str: str) -> bool:
    embs, meta = _load_embeddings()
    if id_str not in meta:
        return False

    try:
        img_path = Path(meta[id_str]["image_path"])
        if img_path.exists():
            img_path.unlink()
    except Exception:
        pass

    idx_to_remove = meta[id_str]["index"]
    if embs.size == 0:
        return True

    new_embs = np.delete(embs, idx_to_remove, axis=0)

    new_meta = {}
    i = 0
    for k, v in meta.items():
        if k == id_str:
            continue
        new_meta[k] = {"index": i, "image_path": v["image_path"], "ts": v["ts"]}
        i += 1

    _save_embeddings(new_embs, new_meta)
    return True
