# -*- coding: utf-8 -*-
import os, re, glob, gzip, json, asyncio, logging, pickle, hashlib
from typing import Any, Dict, List, DefaultDict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import log

from .config import Config
from .utils import build_date_str, tokenize_ar

logger = logging.getLogger(__name__)


def _norm_msg_id(x) -> str:
    """
    وحّد أية قيمة ID على شكل message{ID}
    - لو x رقم: message123
    - لو x سترنج بدون prefix: message{stripped}
    - لو أصلاً بيبدأ بـ message: سيبه كما هو
    - لو فاضي: ""
    """
    s = str(x or "").strip()
    if not s:
        return ""
    return s if s.startswith("message") else f"message{s}"


def _norm_reply_to(rt) -> str:
    """وحّد reply_to برضه (رقم/سترنج) لنفس الفورمات."""
    if rt is None:
        return ""
    if isinstance(rt, int):
        return f"message{rt}"
    s = str(rt).strip()
    if not s:
        return ""
    return s if s.startswith("message") else f"message{s}"


def _paths_signature(paths: List[str]) -> str:
    """
    توقيع بسيط مبني على (المسار + mtime + الحجم) لكل ملف.
    لو اتغير أي جزء من الملفات، يتغير التوقيع ⇒ نعيد بناء الكاش.
    """
    h = hashlib.md5()
    for p in paths:
        try:
            st = os.stat(p)
            h.update(p.encode("utf-8", "ignore"))
            h.update(str(int(st.st_mtime)).encode())
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            # لو الملف اختفى فجأة؛ لسه هنكمّل — التوقيع هيختلف على أي حال
            h.update(p.encode("utf-8", "ignore"))
            h.update(b"0")
            h.update(b"0")
    return h.hexdigest()


class DataRepository:
    def __init__(self, cfg: Config, executor: ThreadPoolExecutor):
        self.cfg = cfg
        self.executor = executor

        # بيانات الرسائل بعد التحميل
        self.metas: List[Dict[str, Any]] = []

        # خرائط علاقات
        self.children: DefaultDict[str, List[int]] = defaultdict(list)
        self.id_to_idx: Dict[str, int] = {}

        # كاشات للبحث
        self._tokens_cache: List[List[str]] = []   # قوائم التوكنز لكل رسالة (لـ BM25)
        self._idf: Dict[str, float] = {}           # IDF لكل تيرم

        # مسار كاش على الديسك (قابل للتغيير بمتغير بيئة)
        # افتراضيًا نستخدم مجلد داخل base_dir
        self.cache_dir: str = os.environ.get(
            "CACHE_DIR",
            os.path.join(self.cfg.base_dir, "data", ".cache")
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    # -------------------- parts discovery --------------------
    def _list_parts(self, dirpath: str) -> List[str]:
        """
        ابحث عن أجزاء البيانات داخل مجلد:
          - ملفات مباشرة: rows_part*.jsonl(.gz)
          - أو مجلدات باسم rows_partN.jsonl/ وبداخلها *.jsonl | *.jsonl.gz
        نُعيد "قائمة ملفات فقط" مرتبة حسب رقم الجزء ثم بالاسم.
        """
        # مطابقات أولية (قد تكون ملفات أو مجلدات)
        raw_matches = glob.glob(os.path.join(dirpath, "rows_part*.jsonl.gz")) + \
                      glob.glob(os.path.join(dirpath, "rows_part*.jsonl"))

        files: List[str] = []

        for p in raw_matches:
            if os.path.isdir(p):
                # توسعة المجلد: خذ كل *.jsonl / *.jsonl.gz داخله
                inner = glob.glob(os.path.join(p, "*.jsonl")) + glob.glob(os.path.join(p, "*.jsonl.gz"))
                # رتب داخل الجزء
                inner.sort()
                for f in inner:
                    if os.path.isfile(f):
                        files.append(f)
            elif os.path.isfile(p):
                files.append(p)
            else:
                # تجاهل أي شيء آخر (symlink مكسور/لا شيء)
                continue

        def _part_num(path: str) -> int:
            """
            استخرج رقم الجزء من اسم مجلد parent لو كان اسمه rows_partN.jsonl،
            وإلا من basename للملف نفسه لو كان بصيغة rows_partN.jsonl(.gz).
            """
            base = os.path.basename(path)
            parent = os.path.basename(os.path.dirname(path))

            m_parent = re.match(r"rows_part(\d+)\.jsonl$", parent)
            if m_parent:
                return int(m_parent.group(1))

            m_file = re.match(r"rows_part(\d+)\.jsonl(\.gz)?$", base)
            if m_file:
                return int(m_file.group(1))

            return 1_000_000  # لو ما عرف يطلع رقم

        # رتّب الملفات حسب رقم الجزء، ثم بالاسم لضمان ترتيب ثابت
        files.sort(key=lambda x: (_part_num(x), x))
        return files

    # -------------------- resolve sources --------------------
    def resolve_rows_sources(self, rows_path: str) -> List[str]:
        # دعم pattern بـ wildcard زي rows_part*.jsonl(.gz)
        if rows_path and ("*" in rows_path or "?" in rows_path or "[" in rows_path):
            files = glob.glob(rows_path)
            # فلترة ملفات فقط
            files = [fp for fp in files if os.path.isfile(fp)]
            files.sort()
            if files:
                logger.info("Using wildcard pattern: matched %d file(s)", len(files))
                return files
            logger.warning("Wildcard pattern matched nothing: %s", rows_path)

        if rows_path and os.path.isdir(rows_path):
            parts = self._list_parts(rows_path)
            if parts:
                logger.info("Found %d part files under %s (numeric order)", len(parts), rows_path)
                return parts
            cand1 = os.path.join(rows_path, "rows.jsonl.gz")
            cand2 = os.path.join(rows_path, "rows.jsonl")
            if os.path.isfile(cand1):
                return [cand1]
            if os.path.isfile(cand2):
                return [cand2]
            raise FileNotFoundError(f"No rows files found in directory: {rows_path}")

        elif rows_path and os.path.isfile(rows_path):
            return [rows_path]

        logger.warning(
            "ROWS_PATH not found or empty: %s — trying defaults under %s",
            rows_path, self.cfg.default_rows_dir
        )
        if os.path.isdir(self.cfg.default_rows_dir):
            parts = self._list_parts(self.cfg.default_rows_dir)
            if parts:
                logger.info("Using parts under default dir: %s", self.cfg.default_rows_dir)
                return parts
            if os.path.isfile(self.cfg.default_rows_file1):
                logger.info("Using default file: %s", self.cfg.default_rows_file1)
                return [self.cfg.default_rows_file1]
            if os.path.isfile(self.cfg.default_rows_file2):
                logger.info("Using default file: %s", self.cfg.default_rows_file2)
                return [self.cfg.default_rows_file2]

        raise FileNotFoundError(
            f"Could not resolve rows files. Tried: {rows_path} and {self.cfg.default_rows_dir}"
        )

    # -------------------- sync IO: load all rows --------------------
    def _load_rows_all_sync(self) -> List[Dict[str, Any]]:
        files = self.resolve_rows_sources(self.cfg.rows_path)
        # تحوّط إضافي: تأكد أنها ملفات فقط (لا تمرر مجلدات لـ open)
        files = [fp for fp in files if os.path.isfile(fp)]

        out: List[Dict[str, Any]] = []
        for fp in files:
            if fp.endswith(".gz"):
                with gzip.open(fp, "rt", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            m = json.loads(line)
                        except Exception as e:
                            logger.warning("Skip bad JSON (gz:%s line:%d): %s", os.path.basename(fp), i, e)
                            continue
                        m.setdefault("date_str", build_date_str(m))
                        out.append(m)
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            m = json.loads(line)
                        except Exception as e:
                            logger.warning("Skip bad JSON (file:%s line:%d): %s", os.path.basename(fp), i, e)
                            continue
                        m.setdefault("date_str", build_date_str(m))
                        out.append(m)
        return out

    # -------------------- maps (IDs/children) --------------------
    def _build_maps_only(self):
        """
        - توحيد id / reply_to → message{ID}
        - بناء id_to_idx
        - بناء children (بالمعرّفات الموحّدة)
        (بدون BM25 tokens)
        """
        self.children.clear()
        self.id_to_idx.clear()

        # 1) وحّد المعرفات داخل الصفوف
        for m in self.metas:
            m["id"] = _norm_msg_id(m.get("id"))
            m["reply_to"] = _norm_reply_to(m.get("reply_to"))

        # 2) خريطة id → index
        for i, m in enumerate(self.metas):
            mid = m.get("id", "")
            if mid:
                self.id_to_idx[mid] = i

        # 3) children
        for i, m in enumerate(self.metas):
            parent = m.get("reply_to", "")
            if parent:
                self.children[parent].append(i)

        logger.info(
            "Maps ready: id_to_idx=%d, parents_with_children=%d",
            len(self.id_to_idx), len(self.children)
        )

    # -------------------- tokens & IDF (build / cache) --------------------
    def _build_tokens_sync(self) -> List[List[str]]:
        """
        يبني التوكنز من نص الرسالة (message) لكل meta.
        """
        toks: List[List[str]] = []
        for m in self.metas:
            txt = (m.get("message", "") or "")
            toks.append(tokenize_ar(txt))
        return toks

    def _build_idf_from_tokens(self, tokens: List[List[str]]) -> Dict[str, float]:
        """
        حساب IDF بشكل قياسي:
        idf(term) = log( (N - df + 0.5) / (df + 0.5) + 1 )
        """
        N = max(1, len(tokens))
        df: Dict[str, int] = {}
        for toks in tokens:
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        idf: Dict[str, float] = {}
        for t, d in df.items():
            # صيغة BM25 الشائعة
            idf[t] = log((N - d + 0.5) / (d + 0.5) + 1.0)
        return idf

    def _load_or_build_tokens_idf_sync(self, files_for_sig: List[str]) -> Tuple[List[List[str]], Dict[str, float]]:
        """
        يحاول تحميل الكاش (tokens + idf) من الديسك.
        لو غير موجود/قديم أو لو REBUILD_CACHE=1 ⇒ يبني من جديد ويحفظ.
        """
        sig = _paths_signature(files_for_sig)
        toks_pkl = os.path.join(self.cache_dir, f"tokens_{sig}.pkl")
        idf_pkl  = os.path.join(self.cache_dir, f"idf_{sig}.pkl")

        force_rebuild = os.getenv("REBUILD_CACHE", "0") == "1"

        if (not force_rebuild) and os.path.exists(toks_pkl) and os.path.exists(idf_pkl):
            try:
                with open(toks_pkl, "rb") as f:
                    tokens = pickle.load(f)
                with open(idf_pkl, "rb") as f:
                    idf = pickle.load(f)
                logger.info("Loaded tokens/IDF cache from disk (sig=%s)", sig)
                return tokens, idf
            except Exception as e:
                logger.warning("Failed to load cache (sig=%s), rebuilding. Error: %s", sig, e)

        # بناء جديد
        tokens = self._build_tokens_sync()
        idf = self._build_idf_from_tokens(tokens)

        # حفظ
        try:
            with open(toks_pkl, "wb") as f:
                pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(idf_pkl, "wb") as f:
                pickle.dump(idf, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Rebuilt and saved tokens/IDF cache (sig=%s)", sig)
        except Exception as e:
            logger.warning("Failed to save cache files (sig=%s): %s", sig, e)

        return tokens, idf

    # -------------------- async load --------------------
    async def load(self):
        """
        تحميل الصفوف وبناء الخرائط بسرعة،
        ثم تحميل/بناء كاش التوكنز وIDF (معتمد على توقيع الملفات).
        """
        loop = asyncio.get_running_loop()

        # 1) حمل كل الرسائل
        metas = await loop.run_in_executor(self.executor, self._load_rows_all_sync)
        self.metas = metas
        logger.info("Loaded %d messages from rows (source: %s)", len(self.metas), self.cfg.rows_path)

        # 2) الخرائط (IDs/children) — خفيفة وسريعة
        await loop.run_in_executor(self.executor, self._build_maps_only)

        # 3) كاش التوكنز/IDF — ثقيل؛ نبنيه/نحمله في executor
        #    استخدم نفس قائمة الملفات التي اعتمدناها في _load_rows_all_sync لتوليد التوقيع
        files = self.resolve_rows_sources(self.cfg.rows_path)
        files = [fp for fp in files if os.path.isfile(fp)]
        tokens, idf = await loop.run_in_executor(self.executor, self._load_or_build_tokens_idf_sync, files)

        self._tokens_cache = tokens
        self._idf = idf

        logger.info("Tokens/IDF ready: tokens=%d docs, vocab=%d terms",
                    len(self._tokens_cache), len(self._idf))

    # -------------------- accessors --------------------
    def get_tokens(self) -> List[List[str]]:
        """قوائم التوكنز لكل رسالة (لاستخدام BM25)."""
        return self._tokens_cache

    def get_idf(self) -> Dict[str, float]:
        """قاموس IDF لكل تيرم."""
        return self._idf
