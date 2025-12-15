import os
import json
import threading
import hashlib
from datetime import datetime
import numpy as np

# FAISS imports are required; this module expects FAISS with CUDA support
import faiss

_LOCK = threading.Lock()

DATA_DIR = os.path.join(os.path.dirname(__file__), 'vector_store_data')
os.makedirs(DATA_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def _text_to_vector(text: str, dim: int = 128):
    if text is None:
        text = ''
    h = hashlib.sha512(text.encode('utf-8')).digest()
    out = np.empty(dim, dtype='float32')
    ln = len(h)
    for i in range(dim):
        b = h[i % ln]
        out[i] = (b / 255.0) * 2.0 - 1.0
    # Normalize for cosine (use inner-product on normalized vectors)
    norm = np.linalg.norm(out)
    if norm > 0:
        out /= norm
    return out


class FAISSVectorStore:
    """FAISS-backed vector store that stores metadata as JSON and vectors in a FAISS index.

    This implementation rebuilds the FAISS index on upsert/delete operations which
    is acceptable for small-medium collections (tasks/subtasks). It requires a
    FAISS build with CUDA support and will move the index to GPU.
    """

    def __init__(self, dim: int = 128, use_gpu: bool = True):
        self.dim = dim
        self.use_gpu = use_gpu
        self._meta_path = os.path.join(DATA_DIR, 'meta.json')
        self._index_path = os.path.join(DATA_DIR, 'index.faiss')
        self._meta = self._load_meta()
        self._index = None
        self._gpu_res = None
        self._ensure_index()

    def _load_meta(self):
        if not os.path.exists(self._meta_path):
            return {}
        try:
            with open(self._meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_meta(self):
        tmp = self._meta_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._meta_path)

    def _ensure_index(self):
        # Build a CPU index from metadata vectors and optionally move to GPU
        vectors = []
        ids = []
        for i, (k, v) in enumerate(self._meta.items()):
            vec = v.get('vector')
            if vec is not None:
                vectors.append(np.array(vec, dtype='float32'))
                ids.append(k)
        if vectors:
            xb = np.vstack(vectors)
            # use inner product on normalized vectors for cosine similarity
            cpu_index = faiss.IndexFlatIP(self.dim)
            cpu_index.add(xb)
            # keep mapping externally; we'll use metadata ordering for results
            self._cpu_index = cpu_index
            if self.use_gpu:
                # create GPU resources and move index
                try:
                    self._gpu_res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
                except Exception as e:
                    raise RuntimeError('Failed to move FAISS index to GPU; ensure faiss-gpu is installed and CUDA available') from e
            else:
                self._index = cpu_index
        else:
            # empty index
            cpu_index = faiss.IndexFlatIP(self.dim)
            self._cpu_index = cpu_index
            if self.use_gpu:
                self._gpu_res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
            else:
                self._index = cpu_index

    def _rebuild_index(self):
        # Rebuild CPU index and move to GPU
        vectors = []
        for k, v in self._meta.items():
            vec = v.get('vector')
            if vec is not None:
                vectors.append(np.array(vec, dtype='float32'))
        if vectors:
            xb = np.vstack(vectors)
        else:
            xb = np.empty((0, self.dim), dtype='float32')
        cpu_index = faiss.IndexFlatIP(self.dim)
        if xb.shape[0] > 0:
            cpu_index.add(xb)
        self._cpu_index = cpu_index
        if self.use_gpu:
            if self._gpu_res is None:
                self._gpu_res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(self._gpu_res, 0, cpu_index)
        else:
            self._index = cpu_index
        # persist CPU index to disk (faiss.write_index writes CPU index)
        faiss.write_index(self._cpu_index, self._index_path)

    def upsert(self, collection: str, id: str, text: str = None, metadata: dict = None, vector: list = None):
        with _LOCK:
            item = self._meta.get(id, {})
            item['id'] = id
            item['text'] = text
            item['metadata'] = metadata or item.get('metadata', {})
            if 'created_at' not in item['metadata']:
                item['metadata']['created_at'] = _now_iso()
            if vector is None and text is not None:
                vec = _text_to_vector(text, dim=self.dim)
            elif vector is not None:
                vec = np.array(vector, dtype='float32')
            else:
                vec = item.get('vector')
            if vec is not None:
                # ensure list for JSON storage
                item['vector'] = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            self._meta[id] = item
            self._save_meta()
            self._rebuild_index()
            return item

    def get(self, collection: str, id: str):
        return self._meta.get(id)

    def delete(self, collection: str, id: str):
        with _LOCK:
            if id in self._meta:
                del self._meta[id]
                self._save_meta()
                self._rebuild_index()
                return True
            return False

    def list(self, collection: str):
        return list(self._meta.values())

    def search(self, collection: str, query_text: str = None, query_vector: list = None, top_k: int = 10):
        # build a query vector if needed
        if query_vector is None and query_text is not None:
            qv = _text_to_vector(query_text, dim=self.dim)
        elif query_vector is not None:
            qv = np.array(query_vector, dtype='float32')
        else:
            return []
        if hasattr(self, '_index') and self._index is not None and self._index.ntotal > 0:
            q = np.expand_dims(qv.astype('float32'), axis=0)
            D, I = self._index.search(q, top_k)
            results = []
            # I contains positions (0..n-1) in the meta ordering; map by order
            keys = list(self._meta.keys())
            for idx in I[0]:
                if idx < 0 or idx >= len(keys):
                    continue
                k = keys[idx]
                results.append(self._meta.get(k))
            return results
        # fallback: substring match
        q = (query_text or '').lower()
        scored = []
        for it in self._meta.values():
            text = (it.get('text') or '')
            score = 1 if q and q in text.lower() else 0
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for s, it in scored[:top_k]]


# singleton
STORE = FAISSVectorStore()


__all__ = ['STORE']
