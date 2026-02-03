# -*- coding: utf-8 -*-
import os
import math, re, asyncio, logging
from typing import Any, Dict, List, Optional, Tuple, Deque
from collections import defaultdict, deque
from datetime import date, datetime
import numpy as np

from .config import Config
from .repository import DataRepository
from .index import FaissIndex
from .model import EmbeddingModel
from .utils import PHONE_RE, tokenize_ar

logger = logging.getLogger(__name__)

# ---------------- BM25 بسيط وسريع (يدعم IDF جاهز) ----------------
class BM25Engine:
    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75,
                 idf: Optional[Dict[str, float]] = None):
        self.k1 = k1; self.b = b
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0

        if idf is not None:
            # استخدم IDF الجاهز (أسرع بكثير في الإقلاع)
            self.idf: Dict[str, float] = idf
        else:
            # fallback: احسب DF/IDF من الصفر (أبطأ)
            df: Dict[str, int] = defaultdict(int)
            for d in tokenized_docs:
                for w in set(d): df[w] += 1
            self.idf = {w: math.log(1 + (self.N - dfw + 0.5) / (dfw + 0.5)) for w, dfw in df.items()}

    def score_one(self, query_toks: List[str], idx: int) -> float:
        if idx < 0 or idx >= self.N: return 0.0
        doc = self.docs[idx]
        if not doc: return 0.0
        dl = self.doc_len[idx]
        tf: Dict[str, int] = defaultdict(int)
        for w in doc: tf[w] += 1
        score = 0.0
        denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
        for q in query_toks:
            f = tf.get(q, 0)
            if f == 0: continue
            idf = self.idf.get(q, 0.0)
            score += idf * ((f * (self.k1 + 1)) / (f + denom_norm))
        return score

    def topn(self, query_toks: List[str], n: int) -> List[Tuple[int, float]]:
        if not self.N or not query_toks: return []
        scores: List[Tuple[int, float]] = []
        for i in range(self.N):
            s = self.score_one(query_toks, i)
            if s > 0: scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ---------------- Reranker (محسّن ومتوافق مع BGE) ----------------
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

class Reranker:
    def __init__(self, cfg: Config, executor):
        self.cfg = cfg
        self.executor = executor
        self.model: Optional[CrossEncoder] = None
        self._batch_size = getattr(cfg, "reranker_batch_size", 16)
        self._max_length = getattr(cfg, "reranker_max_length", 512)
        self._device = getattr(cfg, "reranker_device", "auto")  # "auto"|"cuda"|"cpu"

    async def load(self):
        if not getattr(self.cfg, "use_reranker", False) or CrossEncoder is None:
            return
        loop = asyncio.get_running_loop()

        def _load():
            try:
                return CrossEncoder(
                    self.cfg.reranker_model,
                    device=None if self._device == "auto" else self._device,
                    max_length=self._max_length,
                )
            except Exception as e:
                raise RuntimeError(f"CrossEncoder load failed: {e}")

        try:
            self.model = await loop.run_in_executor(self.executor, _load)
            logger.info("Reranker loaded: %s", self.cfg.reranker_model)
        except Exception as e:
            logger.warning("Failed to load reranker %s: %s", self.cfg.reranker_model, e)
            self.model = None

    def _prep_text(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        return s[:1200]

    async def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model or not items:
            return items

        top_n = min(max(getattr(self.cfg, "rerank_top_n", 40), 1), len(items))
        subset = items[:top_n]

        pairs = []
        for it in subset:
            seed = it.get("seed", {})
            msg = self._prep_text(seed.get("message", "") or "")
            title = self._prep_text(seed.get("title", "") or "")
            text = (title + " — " + msg) if title else msg
            pairs.append((query, text))

        loop = asyncio.get_running_loop()

        def _predict():
            try:
                return self.model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
            except Exception as e:
                raise RuntimeError(f"CrossEncoder predict failed: {e}")

        try:
            scores = await loop.run_in_executor(self.executor, _predict)
            scored = list(zip(scores, subset))
            scored.sort(key=lambda x: float(x[0]), reverse=True)
            reranked = [it for _, it in scored] + items[top_n:]
            return reranked
        except Exception as e:
            logger.warning("Rerank failed: %s", e)
            return items

# ---------------- فلاتر ----------------
from dataclasses import dataclass
@dataclass
class SearchFilters:
    only_with_replies: bool = False
    date_filter: Optional[Tuple[int, int, int]] = None
    date_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    keyword: Optional[str] = None
    only_with_contact: bool = False


# ---------------- محرك البحث الهجين ----------------
class SearchEngine:
    def __init__(self, cfg: Config, repo: DataRepository, index: FaissIndex, model: EmbeddingModel, reranker: Optional[Reranker] = None):
        self.cfg = cfg
        self.repo = repo
        self.index = index
        self.model = model
        self.reranker = reranker
        self._bm25: Optional[BM25Engine] = None

        # أوزان الدمج من ENV (لو متاحة) وإلا من Config كـ fallback
        self._emb_weight = self._get_weight_from_env("EMB_WEIGHT", getattr(cfg, "emb_weight", 0.6))
        self._bm25_weight = self._get_weight_from_env("BM25_WEIGHT", getattr(cfg, "bm25_weight", 0.4))

        # تفعيل/تعطيل BM25 عبر ENV أيضًا (ENABLE_BM25=0/1)، مع fallback لـ cfg.enable_bm25
        self._enable_bm25 = self._get_bool_from_env("ENABLE_BM25", getattr(cfg, "enable_bm25", True))

    # --- Helpers لقراءة ENV ---
    @staticmethod
    def _get_weight_from_env(name: str, default_val: float) -> float:
        try:
            v = os.getenv(name)
            return float(v) if v is not None else default_val
        except Exception:
            return default_val

    @staticmethod
    def _get_bool_from_env(name: str, default_val: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return default_val
        return v.strip() not in ("0", "false", "False", "FALSE", "")

    # --- مساعدات للتعامل مع اختلاف شكل IDs ---
    @staticmethod
    def _variants(any_id: Any) -> List[str]:
        v: List[str] = []
        s = str(any_id)
        v.append(s)
        if s.startswith("message"):
            raw = s.replace("message", "", 1)
            if raw.isdigit(): v.append(raw)
        else:
            if s.isdigit(): v.append("message" + s)
        return list(dict.fromkeys(v))  # unique order

    def _resolve_idx(self, any_id: Any) -> Optional[int]:
        for k in self._variants(any_id):
            idx = self.repo.id_to_idx.get(k)  # type: ignore
            if idx is not None:
                return idx
        return None

    def _children_for(self, any_id: Any) -> List[int]:
        seen: set[int] = set()
        out: List[int] = []
        for k in self._variants(any_id):
            for c in self.repo.children.get(k, []):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    def _ensure_bm25(self):
        """
        BM25 كسول: يُبنى فقط عند الحاجة.
        يستخدم توكنز وIDF من DataRepository إن وُجدا لتسريع الإقلاع.
        """
        if not self._enable_bm25:
            self._bm25 = None
            return
        if self._bm25 is None:
            # جرّب الحصول على الكاش الجاهز
            tokens = getattr(self.repo, "get_tokens", lambda: getattr(self.repo, "_tokens_cache", []))()
            idf = getattr(self.repo, "get_idf", lambda: None)()
            self._bm25 = BM25Engine(tokens, idf=idf)

    @staticmethod
    def _minmax(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores: return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi - lo <= 1e-12:
            return {k: 0.0 for k in scores}
        inv = 1.0 / (hi - lo)
        return {k: (v - lo) * inv for k, v in scores.items()}


    def _seed_datetime_utc(self, seed: Dict[str, Any]) -> Optional[datetime]:
        """Build a datetime from seed fields (year/month/day/hour/minute/second). Returns naive UTC datetime."""
        try:
            y = int(seed.get("year") or 0)
            mo = int(seed.get("month") or 0)
            d = int(seed.get("day") or 0)
            if y and mo and d:
                hh = int(seed.get("hour") or 0)
                mm = int(seed.get("minute") or 0)
                ss = int(seed.get("second") or 0)
                return datetime(y, mo, d, hh, mm, ss)
        except Exception:
            return None
        # fallback: try parsing date_str
        ds = (seed.get("date_str") or seed.get("date") or "").strip()
        if not ds:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
            try:
                return datetime.strptime(ds, fmt)
            except Exception:
                continue
        return None
    
    def _recency_score(self, dt: Optional[datetime], half_life_days: int) -> float:
        """0..1, where 1 is newest. Uses exponential decay with half-life."""
        if not dt:
            return 0.0
        try:
            now = datetime.utcnow()
            days = max((now - dt).total_seconds() / 86400.0, 0.0)
            return float(math.exp(-math.log(2) * (days / float(max(1, half_life_days)))))
        except Exception:
            return 0.0
    
    async def _smart_replies(self, q_vec, seed_id: Any, max_depth: int, max_replies: int, keyword: Optional[str]):
        idxs: List[Tuple[int, int]] = []
        queue: Deque[Tuple[int, int]] = deque((c, 1) for c in self._children_for(seed_id))
        while queue and len(idxs) < max_replies * 4:
            r_idx, depth = queue.popleft()
            if depth > max_depth: continue
            idxs.append((r_idx, depth))
            r_mid = self.repo.metas[r_idx].get("id", "")
            for cc in self._children_for(r_mid):
                queue.append((cc, depth + 1))
        if not idxs: return []
        texts = [(self.repo.metas[r_idx].get("message", "") or "").strip() for r_idx, _ in idxs]
        try:
            vecs = await self.model.encode_many(texts)
        except Exception:
            vecs = None
        out = []
        kw = (keyword or "").lower().strip()
        for i, (r_idx, depth) in enumerate(idxs):
            msg = (self.repo.metas[r_idx].get("message", "") or "").strip()
            base = 0.0
            if vecs is not None:
                try:
                    base = float(np.dot(q_vec[0], vecs[i]))
                except Exception:
                    base = 0.0
            bonus = 0.0
            if kw and kw in msg.lower(): bonus += 0.05
            if PHONE_RE.search(msg) or ("@" in msg): bonus += 0.05
            out.append((base + bonus, r_idx, depth))
        out.sort(key=lambda x: x[0], reverse=True)
        top = out[:max_replies]
        return [(depth, self.repo.metas[r_idx]) for _, r_idx, depth in top]

    async def search(self, query: str, top_k: int, flt: SearchFilters) -> List[Dict[str, Any]]:
        if not query or not query.strip(): return []

        # ----- Encode query -----
        q_vec = await self.model.encode(query)
        try:
            if isinstance(q_vec, np.ndarray):
                if q_vec.ndim == 1:
                    q_vec = q_vec.reshape(1, -1)
            else:
                q_vec = np.asarray(q_vec, dtype=np.float32)
                if q_vec.ndim == 1:
                    q_vec = q_vec.reshape(1, -1)
        except Exception:
            pass

        loop = asyncio.get_running_loop()

        # ----- مرشحي FAISS -----
        emb_scores: Dict[int, float] = {}
        metas_len = len(self.repo.metas)
        D = I = None

        def _faiss_sync():
            if not getattr(self.index, "index", None):
                return None, None
            return self.index.index.search(q_vec, faiss_k)

        faiss_k = max(getattr(self.cfg, "faiss_candidates", 60), 3 * top_k)
        try:
            D, I = await loop.run_in_executor(self.model.executor, _faiss_sync)
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)
            D, I = None, None

        if D is not None and I is not None:
            skipped = 0
            for s, idx in zip(D[0], I[0]):
                if idx < 0: continue
                if idx >= metas_len:
                    skipped += 1
                    continue
                if s < self.cfg.min_similarity: continue
                emb_scores[int(idx)] = float(s)
            if skipped:
                logger.warning("FAISS returned %d indices >= metas_len=%d — skipping", skipped, metas_len)

        # ----- مرشحي BM25 (اختياري/كسول) -----
        bm_scores: Dict[int, float] = {}
        if self._enable_bm25:
            self._ensure_bm25()
            q_toks = tokenize_ar(query)
            bm_k = max(getattr(self.cfg, "bm25_candidates", 60), 3 * top_k)
            hits = self._bm25.topn(q_toks, bm_k) if self._bm25 else []
            bm_scores = {i: s for i, s in hits}

        # ----- دمج -----
        candidates: Dict[int, Dict[str, float]] = {}
        for i, s in emb_scores.items(): candidates.setdefault(i, {})['emb'] = s
        for i, s in bm_scores.items():  candidates.setdefault(i, {})['bm25'] = s
        if not candidates: return []

        emb_n = self._minmax({i: v.get('emb', 0.0) for i, v in candidates.items()})
        bm_n  = self._minmax({i: v.get('bm25', 0.0) for i, v in candidates.items()})

        weight_emb = self._emb_weight
        weight_bm  = self._bm25_weight

        fused_pairs: List[Tuple[float, int]] = []
        for i in candidates.keys():
            s = weight_emb * emb_n.get(i, 0.0) + weight_bm * bm_n.get(i, 0.0)
            fused_pairs.append((s, i))
        fused_pairs.sort(key=lambda x: x[0], reverse=True)

        # ----- فلاتر + ترقية الرد إلى الأب + تجميع درجات الأب -----
        pos_terms: List[str] = []; neg_terms: List[str] = []
        if flt.keyword:
            raw = [k.strip() for k in re.split(r"[|]+", flt.keyword) if k.strip()]
            for term in raw:
                if term.startswith("-") and len(term) > 1: neg_terms.append(term[1:].lower())
                else: pos_terms.append(term.lower())

        def _match_keywords(text: str) -> bool:
            t = (text or "").lower()
            if pos_terms and not any(term in t for term in pos_terms): return False
            if neg_terms and any(term in t for term in neg_terms): return False
            return True

        def _in_range(seed, dr):
            (y1, m1, d1), (y2, m2, d2) = dr
            try:
                ds = date(seed.get("year", 0), seed.get("month", 0), seed.get("day", 0))
                return date(y1, m1, d1) <= ds <= date(y2, m2, d2)
            except Exception:
                return False

        cap = max(top_k * 3, getattr(self.cfg, "rerank_top_n", 50))

        # مجمّع: أقوى نتيجة للأب + أفضل رد + عدد الردود ضمن المرشحين
        agg_by_seed: Dict[str, Dict[str, Any]] = {}

        # باراميترات الترقية (قابلة للضبط عبر Config لو أردت)
        reply_boost = float(getattr(self.cfg, "reply_boost", 0.15))
        reply_count_boost = float(getattr(self.cfg, "reply_count_boost", 0.02))
        reply_as_parent_weight = float(getattr(self.cfg, "reply_as_parent_weight", 0.9))

        for fused_score, idx in fused_pairs:
            if idx >= metas_len:
                continue

            cand = self.repo.metas[idx]
            cand_id = cand.get("id", "")
            parent_id = cand.get("reply_to", "") or cand_id

            # لو رد → رقّيه للأب لو قدرنا
            role = "parent"
            orig_reply = None
            seed = cand
            seed_id = str(cand_id)
            if str(parent_id) != str(cand_id):
                pidx = self._resolve_idx(parent_id)
                if pidx is not None:
                    role = "reply"
                    orig_reply = cand
                    seed = self.repo.metas[pidx]
                    seed_id = str(seed.get("id", ""))

            # فلاتر تُطبّق على الأب
            if flt.only_with_replies and not self._children_for(seed_id):
                continue
            if flt.date_filter:
                y, mo, d = flt.date_filter
                if not (seed.get("year", 0) == y and seed.get("month", 0) == mo and seed.get("day", 0) == d):
                    continue
            if flt.date_range and not _in_range(seed, flt.date_range):
                continue
            msg_text = (seed.get("message", "") or "").strip()
            if len(msg_text) < self.cfg.min_text_len:
                continue
            if (pos_terms or neg_terms) and not _match_keywords(msg_text):
                continue
            if flt.only_with_contact:
                has_phone = bool(PHONE_RE.search(msg_text))
                has_username = "@" in msg_text
                if not (has_phone or has_username):
                    continue

            # حدّث مجمّع الأب
            agg = agg_by_seed.get(seed_id)
            if not agg:
                agg = {
                    "seed": seed,
                    "parent_fused": 0.0,
                    "parent_emb": 0.0,
                    "parent_bm": 0.0,
                    "reply_best_fused": 0.0,
                    "reply_best_meta": None,
                    "reply_hits": 0,
                }
                agg_by_seed[seed_id] = agg

            fused = float(fused_score)
            if role == "parent":
                if fused > agg["parent_fused"]:
                    agg["parent_fused"] = fused
                    agg["parent_emb"] = float(emb_scores.get(idx, 0.0))
                    agg["parent_bm"] = float(bm_scores.get(idx, 0.0))
            else:
                agg["reply_hits"] += 1
                if fused > agg["reply_best_fused"]:
                    agg["reply_best_fused"] = fused
                    agg["reply_best_meta"] = orig_reply

            if len(agg_by_seed) >= cap:
                # نسمح بتحديث نفس الأب لو ظهر لاحقًا بنتيجة أعلى، لكن لا نضيف آباء جدد
                pass

        if not agg_by_seed:
            return []

        # حوّل المُجمّع لعناصر + احسب درجة نهائية للأب
        items: List[Dict[str, Any]] = []
        for seed_id, agg in agg_by_seed.items():
            parent_s = float(agg.get("parent_fused", 0.0))
            reply_best_s = float(agg.get("reply_best_fused", 0.0))
            reply_hits = int(agg.get("reply_hits", 0))

            base = parent_s
            if base <= 0.0 and reply_best_s > 0.0:
                base = reply_best_s * reply_as_parent_weight
            fused_final = base + reply_best_s * reply_boost + min(reply_hits, 5) * reply_count_boost

            # ✅ Recency bias (default: newest always). Controlled via ENV:
            # RECENCY_WEIGHT (float) and RECENCY_HALF_LIFE_DAYS (int)
            try:
                rec_w = float(os.getenv("RECENCY_WEIGHT", getattr(self.cfg, "recency_weight", 0.55)))
            except Exception:
                rec_w = float(getattr(self.cfg, "recency_weight", 0.55))
            try:
                half_life = int(os.getenv("RECENCY_HALF_LIFE_DAYS", getattr(self.cfg, "recency_half_life_days", 60)))
            except Exception:
                half_life = int(getattr(self.cfg, "recency_half_life_days", 60))

            dt = self._seed_datetime_utc(agg["seed"])
            rec = self._recency_score(dt, half_life)
            fused_final = fused_final + (rec_w * rec)

            item = {
                "seed": agg["seed"],
                "replies": [],
                "emb_score": float(agg.get("parent_emb", 0.0)),
                "bm25_score": float(agg.get("parent_bm", 0.0)),
                "fused_score": fused_final,
                "score": fused_final,
            }
            if agg.get("reply_best_meta") is not None:
                item["best_reply"] = (1, agg["reply_best_meta"])
            items.append(item)

        # رتّب حسب الدرجة النهائية
        items.sort(key=lambda it: it.get("fused_score", 0.0), reverse=True)

        # ----- Rerank (اختياري) -----
        if self.reranker and getattr(self.cfg, "use_reranker", False) and items:
            items = await self.reranker.rerank(query, items)

        # ----- top_k + حساب الردود الذكية -----
        items = items[:top_k]
        out: List[Dict[str, Any]] = []
        for it in items:
            seed = it["seed"]
            sid = seed.get("id", "")
            replies = await self._smart_replies(q_vec, sid, self.cfg.max_depth, self.cfg.max_replies, flt.keyword)
            it["replies"] = replies
            if replies and "best_reply" not in it:
                it["best_reply"] = replies[0]
            out.append(it)
        return out
