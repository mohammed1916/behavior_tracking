import os
import json
import threading
import hashlib
from datetime import datetime
from math import sqrt

_LOCK = threading.Lock()

DATA_DIR = os.path.join(os.path.dirname(__file__), 'vector_store_data')
os.makedirs(DATA_DIR, exist_ok=True)


def _collection_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.json")


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def _text_to_vector(text: str, dim: int = 128):
    # deterministic pseudo-embedding from text (no external deps)
    if text is None:
        text = ''
    h = hashlib.sha512(text.encode('utf-8')).digest()
    out = []
    ln = len(h)
    for i in range(dim):
        b = h[i % ln]
        out.append((b / 255.0) * 2.0 - 1.0)
    return out


def _cosine(a, b):
    if a is None or b is None:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


class VectorStore:
    """Simple file-backed vector store.

    Each collection is stored as JSON mapping id -> item where item contains
    'id', 'text', 'metadata' and optional 'vector' (list of floats).
    """

    def __init__(self):
        self._cache = {}

    def _load_collection(self, name: str):
        path = _collection_path(name)
        if name in self._cache:
            return self._cache[name]
        if not os.path.exists(path):
            data = {}
        else:
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
        self._cache[name] = data
        return data

    def _save_collection(self, name: str, data: dict):
        path = _collection_path(name)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        self._cache[name] = data

    def upsert(self, collection: str, id: str, text: str = None, metadata: dict = None, vector: list = None):
        with _LOCK:
            data = self._load_collection(collection)
            item = data.get(id, {})
            item['id'] = id
            item['text'] = text
            item['metadata'] = metadata or {}
            if 'created_at' not in item['metadata']:
                item['metadata']['created_at'] = _now_iso()
            if vector is None and text is not None:
                vector = _text_to_vector(text)
            item['vector'] = vector
            data[id] = item
            self._save_collection(collection, data)
            return item

    def get(self, collection: str, id: str):
        data = self._load_collection(collection)
        return data.get(id)

    def delete(self, collection: str, id: str):
        with _LOCK:
            data = self._load_collection(collection)
            if id in data:
                del data[id]
                self._save_collection(collection, data)
                return True
            return False

    def list(self, collection: str):
        data = self._load_collection(collection)
        return list(data.values())

    def search(self, collection: str, query_text: str = None, query_vector: list = None, top_k: int = 10):
        data = self._load_collection(collection)
        items = list(data.values())
        if query_vector is None and query_text is not None:
            query_vector = _text_to_vector(query_text)
        if query_vector is not None:
            scored = []
            for it in items:
                score = _cosine(query_vector, it.get('vector'))
                scored.append((score, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for s, it in scored[:top_k]]
        # fallback: substring scoring by presence
        if query_text:
            q = query_text.lower()
            scored = []
            for it in items:
                text = (it.get('text') or '')
                score = 0
                if q in text.lower():
                    score += 1
                scored.append((score, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for s, it in scored[:top_k]]
        return items[:top_k]


# singleton
STORE = VectorStore()


__all__ = ['STORE']
