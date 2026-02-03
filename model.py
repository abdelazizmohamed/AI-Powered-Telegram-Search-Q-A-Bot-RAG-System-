
# -*- coding: utf-8 -*-
import asyncio
import logging
from typing import Optional, Tuple, List, Sequence, Dict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Config
from .cache import LRUCache

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    مسؤول عن:
      - تحميل SentenceTransformer (معFallback لو اختلف بعد التضمين عن بُعد فهرس FAISS).
      - تنسيق الاستعلام لنماذج E5 تلقائيًا (prefix: 'query: ').
      - كاش LRU للتضمينات الفردية.
      - encode_many بدُفعات batch لتجنب استهلاك الذاكرة.
    """

    def __init__(self, cfg: Config, executor: ThreadPoolExecutor):
        self.cfg = cfg
        self.executor = executor
        self.model: Optional[SentenceTransformer] = None
        self.loaded_model_name: Optional[str] = None
        self.backend: str = "sentence_transformers"  # or "openai"
        self._openai_client = None
        self.cache = LRUCache(cfg.embed_cache_size)

    # -------------------- OpenAI embeddings helpers --------------------

    def _get_openai_client(self):
        """Lazy import + client init so the project doesn't crash if openai isn't installed."""
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI embeddings requested but python package 'openai' is not installed. "
                "Install it with: pip install openai"
            ) from e
        if not self.cfg.openai_api_key:
            raise RuntimeError("OpenAI embeddings requested but OPENAI_API_KEY is missing")
        self._openai_client = OpenAI(api_key=self.cfg.openai_api_key)
        return self._openai_client

    def _encode_openai_sync(self, inputs: List[str]) -> np.ndarray:
        client = self._get_openai_client()
        resp = client.embeddings.create(model=self.cfg.openai_emb_model, input=inputs)
        vecs = [d.embedding for d in resp.data]
        arr = np.array(vecs, dtype="float32")
        # normalize (FAISS cosine-style dot with normalized vectors)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    # -------------------- Loading --------------------

    def _load_model_sync(self, name: str, device: Optional[str] = None) -> Tuple[SentenceTransformer, int]:
        """
        تحميل الموديل بشكل متزامن (يُستدعى داخل run_in_executor).
        يمكن تمرير device="cpu"/"cuda" إن رغبت. نتركها None افتراضيًا ليقرر ST الأفضل.
        """
        m = SentenceTransformer(name, device=device)  # device=None ⇒ يختار تلقائيًا
        dim = m.get_sentence_embedding_dimension()
        return m, dim

    async def load(self, expected_dim: Optional[int] = None):
        """
        يحاول تحميل model_name الرئيسي؛ لو البُعد مختلف عن expected_dim، يجرب alt_models بنفس الترتيب.
        """
        loop = asyncio.get_running_loop()

        # إذا كان EMB_MODEL يشير صراحةً إلى موديل OpenAI embeddings
        # (مثل text-embedding-3-small/large) فسنستخدم OpenAI مباشرة.
        mn = (self.cfg.model_name or "").strip().lower()
        if mn.startswith("text-embedding-") or mn.startswith("openai:"):
            self.backend = "openai"
            self.model = None
            self.loaded_model_name = self.cfg.openai_emb_model
            # تحقق مبكر من وجود المفتاح والمكتبة حتى نفشل برسالة واضحة
            _ = self._get_openai_client()
            logger.info("Embedding backend=OpenAI (model=%s)", self.cfg.openai_emb_model)
            return

        # حاول الأساسي
        try:
            logger.info("Trying to load model %s", self.cfg.model_name)
            m, dim = await loop.run_in_executor(self.executor, lambda: self._load_model_sync(self.cfg.model_name))
        except Exception as e:
            logger.warning("Primary model load failed (%s): %s", self.cfg.model_name, e)
            m, dim = None, None  # نسمح للفولباك يكمل

        # تحقّق التوافق مع expected_dim أو جرّب الفولباك
        if expected_dim is not None:
            if m is None or dim != expected_dim:
                if m is not None:
                    logger.warning("Requested model dim %s != index dim %s", dim, expected_dim)
                loaded = False
                for name in self.cfg.alt_models:
                    try:
                        m2, d2 = await loop.run_in_executor(
                            self.executor, lambda: self._load_model_sync(name)
                        )
                        if d2 == expected_dim:
                            self.model = m2
                            self.loaded_model_name = name
                            loaded = True
                            logger.info("Using fallback model %s (dim=%s)", name, d2)
                            break
                    except Exception as e:
                        logger.warning("Failed to load fallback %s: %s", name, e)
                if not loaded:
                    # لو الفهرس dim=1536/3072 غالبًا متوافق مع OpenAI embeddings
                    if expected_dim in (1536, 3072) and self.cfg.openai_api_key:
                        self.backend = "openai"
                        self.model = None
                        self.loaded_model_name = self.cfg.openai_emb_model
                        _ = self._get_openai_client()
                        logger.info(
                            "No SentenceTransformer with dim=%s; falling back to OpenAI embeddings (model=%s)",
                            expected_dim,
                            self.cfg.openai_emb_model,
                        )
                        return
                    raise RuntimeError(f"No model with dim {expected_dim} found from candidates")
            else:
                # البُعد متوافق
                self.model = m
                self.loaded_model_name = self.cfg.model_name
                self.backend = "sentence_transformers"
        else:
            # مفيش expected_dim — استخدم اللي اتحمّل (أو حاول أوّل بديل لو الأساسي فشل)
            if m is not None:
                self.model = m
                self.loaded_model_name = self.cfg.model_name
                self.backend = "sentence_transformers"
            else:
                last_err = None
                for name in self.cfg.alt_models:
                    try:
                        m2, d2 = await loop.run_in_executor(self.executor, lambda: self._load_model_sync(name))
                        self.model = m2
                        self.loaded_model_name = name
                        break
                    except Exception as e:
                        last_err = e
                        logger.warning("Failed to load fallback %s: %s", name, e)
                if self.model is None:
                    raise RuntimeError(f"Failed to load any embedding model. Last error: {last_err}")

                self.backend = "sentence_transformers"

        logger.info(
            "Model loaded OK: %s (index_dim=%s)",
            self.loaded_model_name or self.cfg.model_name,
            expected_dim if expected_dim is not None else "unknown",
        )

    # -------------------- Encoding helpers --------------------

    def format_query(self, text: str) -> str:
        """
        E5-family models تحتاج prefix: 'query: ' للكواري (وبتستخدم 'passage: ' للدك).
        هنا إحنا بنكود فقط كواري، فبنضيف 'query: ' لو الموديل E5.
        """
        name = (self.loaded_model_name or self.cfg.model_name or "").lower()
        if "e5" in name:
            return f"query: {text}"
        return text

    def _ensure_loaded(self):
        if self.backend == "openai":
            # في وضع OpenAI، model=None طبيعي
            _ = self._get_openai_client()
            return
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded. Call load() first.")

    # -------------------- Public API --------------------

    async def encode(self, text: str) -> np.ndarray:
        """
        ترجع مصفوفة شكلها (1, dim) نوع float32 ومُطبّعـة (normalize_embeddings=True).
        فيها كاش LRU على النص بعد تنسيقه بـ format_query.
        """
        self._ensure_loaded()
        q = self.format_query(text)
        cached = self.cache.get(q)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        if self.backend == "openai":
            vec1 = await loop.run_in_executor(self.executor, lambda: self._encode_openai_sync([q]))
            vec = vec1.astype("float32")[0:1]
        else:
            vec = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32"),  # type: ignore[union-attr]
            )
        # vec شكلها (1, dim)
        self.cache.set(q, vec)
        return vec

    async def encode_many(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        ترجع مصفوفة شكلها (N, dim) نوع float32. تقسم الداتا لدُفعات لتقليل استهلاك الذاكرة.
        - بتحترم الكاش لكل نص (لو use_cache=True).
        - بتضيف 'query: ' لو الموديل E5 لكل عنصر.
        """
        self._ensure_loaded()
        if not texts:
            if self.backend == "openai":
                # dim معروف من إعدادات الفهرس عادةً، لكن هنا نرجّع مصفوفة فارغة آمنة
                return np.zeros((0, 0), dtype="float32")
            return np.zeros((0, getattr(self.model, "get_sentence_embedding_dimension", lambda: 0)()), dtype="float32")  # type: ignore

        # جهّز القائمة مع مراعاة الكاش
        formatted: List[str] = []
        out_blocks: List[np.ndarray] = []
        to_encode_idx: List[int] = []
        order_map: Dict[int, int] = {}  # idx_in_to_encode -> original_idx

        # 1) حاول تدي من الكاش
        cached_rows: Dict[int, np.ndarray] = {}
        for i, t in enumerate(texts):
            q = self.format_query(t)
            if use_cache:
                c = self.cache.get(q)
            else:
                c = None
            if c is not None:
                # (1, dim) → خزنه لنركّبه في الآخر
                cached_rows[i] = c
            else:
                order_map[len(formatted)] = i
                formatted.append(q)
                to_encode_idx.append(i)

        # 2) إنكود اللي مش في الكاش على دفعات
        if formatted:
            loop = asyncio.get_running_loop()

            def _encode_batch(batch: List[str]) -> np.ndarray:
                if self.backend == "openai":
                    return self._encode_openai_sync(batch)
                return self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True).astype("float32")  # type: ignore[union-attr]

            start = 0
            while start < len(formatted):
                end = min(start + batch_size, len(formatted))
                batch = formatted[start:end]
                arr = await loop.run_in_executor(self.executor, lambda b=batch: _encode_batch(b))
                out_blocks.append(arr)

                # خزّن في الكاش
                if use_cache:
                    for j, q in enumerate(batch):
                        # arr[j:j+1] => (1, dim)
                        self.cache.set(q, arr[j:j+1])

                start = end

        # 3) ركّب النتائج بنفس ترتيب الإدخال
        if out_blocks:
            encoded_concat = np.vstack(out_blocks)  # (M, dim)
        else:
            encoded_concat = np.zeros((0, cached_rows[next(iter(cached_rows))].shape[1] if cached_rows else 0), dtype="float32")

        # جهّز مصفوفة الناتج
        if encoded_concat.size:
            dim = encoded_concat.shape[1]
        elif cached_rows:
            dim = next(iter(cached_rows.values())).shape[1]
        else:
            dim = 0 if self.backend == "openai" else getattr(self.model, "get_sentence_embedding_dimension", lambda: 0)()  # type: ignore
        out = np.zeros((len(texts), dim), dtype="float32")

        # حط الكاش
        for i, v in cached_rows.items():
            out[i, :] = v[0]

        # وحط المُشَفَّر حديثًا في مواضعه
        if formatted:
            k = 0
            for idx_in_formatted, orig_i in order_map.items():
                out[orig_i, :] = encoded_concat[k, :]
                k += 1

        return out