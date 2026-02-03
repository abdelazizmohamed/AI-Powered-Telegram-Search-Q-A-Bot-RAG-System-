# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    # ========== أساسي ==========
    token: str = field(default_factory=lambda: os.environ.get("QIMAH_BOT_TOKEN", ""))

    # مسارات الفهرس والصفوف — يمكن تمريرها من ENV
    index_path: str = field(default_factory=lambda: os.environ.get("INDEX_PATH", ""))
    rows_path:  str = field(default_factory=lambda: os.environ.get("ROWS_PATH", ""))

    # جذور الافتراضيات (داخل الباكدج)
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    default_index_path: str = field(init=False)
    default_rows_dir:   str = field(init=False)
    default_rows_file1: str = field(init=False)  # rows.jsonl.gz
    default_rows_file2: str = field(init=False)  # rows.jsonl

    # ========== نماذج التضمين ==========
    model_name: str = field(default_factory=lambda: os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small"))
    alt_models: Tuple[str, ...] = (
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "BAAI/bge-multilingual-base",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # ========== OpenAI Embeddings (اختياري) ==========
    # لو فهرس FAISS مبنيّ بتضمينات OpenAI (dim=1536 غالبًا):
    # - استخدم EMB_MODEL=text-embedding-3-small (أو text-embedding-3-large)
    # - واضبط OPENAI_API_KEY
    # ملاحظة: هذا مستقل عن OpenAI Answer (الدردشة). ممكن تستخدم واحد بدون الآخر.
    openai_emb_model: str = field(default_factory=lambda: os.environ.get("OPENAI_EMB_MODEL", "text-embedding-3-small"))
    openai_emb_batch_size: int = int(os.environ.get("OPENAI_EMB_BATCH_SIZE", "64"))

    # ========== Reranker (اختياري) ==========
    use_reranker: bool = field(default_factory=lambda: os.environ.get("USE_RERANKER", "true").lower() == "true")
    reranker_model: str = field(default_factory=lambda: os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    rerank_top_n: int = int(os.environ.get("RERANK_TOP_N", "40"))
    reranker_batch_size: int = int(os.environ.get("RERANKER_BATCH_SIZE", "16"))
    reranker_max_length: int = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))
    reranker_device: str = os.environ.get("RERANKER_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

    # ========== Hybrid Search (FAISS + BM25) ==========
    emb_weight: float = float(os.environ.get("EMB_WEIGHT", "0.7"))
    bm25_weight: float = float(os.environ.get("BM25_WEIGHT", "0.3"))
    faiss_candidates: int = int(os.environ.get("FAISS_CANDIDATES", "40"))
    bm25_candidates: int  = int(os.environ.get("BM25_CANDIDATES", "40"))
    enable_bm25: bool = field(default_factory=lambda: os.environ.get("ENABLE_BM25", "true").lower() == "true")

    # ========== تجميع الردود ==========
    reply_boost: float = float(os.environ.get("REPLY_BOOST", "0.15"))
    reply_count_boost: float = float(os.environ.get("REPLY_COUNT_BOOST", "0.02"))
    reply_as_parent_weight: float = float(os.environ.get("REPLY_AS_PARENT_WEIGHT", "0.9"))

    # ========== UX/الأداء ==========
    min_similarity: float = float(os.environ.get("MIN_SIMILARITY", "0.35"))
    top_k_default: int = int(os.environ.get("TOP_K_DEFAULT", "7"))
    max_replies: int = int(os.environ.get("MAX_REPLIES", "3"))
    max_depth: int = int(os.environ.get("MAX_DEPTH", "1"))
    page_size_default: int = int(os.environ.get("PAGE_SIZE_DEFAULT", "5"))
    page_size_min: int = int(os.environ.get("PAGE_SIZE_MIN", "3"))
    page_size_max: int = int(os.environ.get("PAGE_SIZE_MAX", "20"))

    # ✅ IDs الأدمن من ENV: ADMIN_IDS="123,456"
    admin_ids: Tuple[int, ...] = field(default_factory=lambda: tuple(
        int(x) for x in (os.environ.get("ADMIN_IDS", "").replace(" ", "").split(",")
                         if os.environ.get("ADMIN_IDS") else [])
        if x.isdigit()
    ))

    # nprobe
    nprobe: int = int(os.environ.get("NPROBE", "16"))
    thread_workers: int = int(os.environ.get("THREAD_WORKERS", "6"))
    embed_cache_size: int = int(os.environ.get("EMBED_CACHE_SIZE", "1024"))
    min_text_len: int = int(os.environ.get("MIN_TEXT_LEN", "12"))

    # ========== Inline ==========
    inline_top_k: int = int(os.environ.get("INLINE_TOP_K", "5"))
    inline_use_reranker: bool = field(default_factory=lambda: os.environ.get("INLINE_USE_RERANKER", "false").lower() == "true")

    # ========== FAISS OMP threads ==========
    faiss_omp_threads: int = max(1, int(os.environ.get("FAISS_OMP_THREADS", "2")))

    # ========== براندنج/تعريب ديناميكي ==========
    bot_name: str = field(default_factory=lambda: os.environ.get("BOT_NAME", "Search Bot"))
    university_name: str = field(default_factory=lambda: os.environ.get("UNIVERSITY_NAME", "جامعة الحدود الشمالية"))
    audience_label: str = field(default_factory=lambda: os.environ.get("AUDIENCE_LABEL", "طلاب"))
    home_intro_override: str = field(default_factory=lambda: os.environ.get("HOME_INTRO", ""))  # لو حاب نص /start مخصص

    # ========== تحليلات/إحصائيات ==========
    enable_analytics: bool = field(default_factory=lambda: os.environ.get("ENABLE_ANALYTICS", "true").lower() == "true")
    multibot_id: str = field(default_factory=lambda: os.environ.get("MULTIBOT_ID", ""))  # مُعرّف منطقي لكل جامعة/بوت

    # ========== OpenAI Answer (اختياري) ==========
    # تفعيل: USE_OPENAI_ANSWER=true
    # المفتاح: OPENAI_API_KEY=...
    # الموديل: OPENAI_CHAT_MODEL=gpt-4o-mini  (موديل mini اقتصادي مناسب)
    use_openai_answer: bool = field(default_factory=lambda: os.environ.get("USE_OPENAI_ANSWER", "false").lower() == "true")
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_chat_model: str = field(default_factory=lambda: os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

    # تحكم في تكلفة/دقة الإجابة
    openai_temperature: float = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
    openai_max_tokens: int = int(os.environ.get("OPENAI_MAX_TOKENS", "450"))

    # بناء السياق (RAG) من نتائج البحث
    openai_max_refs: int = int(os.environ.get("OPENAI_MAX_REFS", "8"))                 # عدد النتائج اللي تُرسل للموديل
    openai_ref_max_chars: int = int(os.environ.get("OPENAI_REF_MAX_CHARS", "800"))     # قص كل نتيجة قبل الإرسال
    openai_replies_per_ref: int = int(os.environ.get("OPENAI_REPLIES_PER_REF", "2"))   # كم رد لكل نتيجة (لو متاح)
    openai_top_sources: int = int(os.environ.get("OPENAI_TOP_SOURCES", "3"))           # عدد المصادر المعروضة (Top 3)
    openai_context_char_limit: int = int(os.environ.get("OPENAI_CONTEXT_CHAR_LIMIT", "9000"))  # حد أقصى للسياق

    # حد توازي نداءات OpenAI داخل البوت
    openai_max_concurrency: int = int(os.environ.get("OPENAI_MAX_CONCURRENCY", "5"))

    # ========== إعدادات الحماية الجديدة ==========
    max_user_text_len: int = int(os.environ.get("MAX_USER_TEXT_LEN", "2000"))     # أقصى طول لنص المستخدم
    max_concurrency: int = int(os.environ.get("MAX_CONCURRENCY", "20"))           # أقصى عدد عمليات متزامنة (بحث)
    enable_whitelist: bool = field(default_factory=lambda: os.environ.get("ENABLE_WHITELIST", "false").lower() == "true")
    # ALLOWED_CHAT_IDS="111,222,333"
    allowed_chat_ids: Tuple[int, ...] = field(default_factory=lambda: tuple(
        int(x) for x in (os.environ.get("ALLOWED_CHAT_IDS", "").replace(" ", "").split(",")
                         if os.environ.get("ALLOWED_CHAT_IDS") else [])
        if x.isdigit()
    ))

    def __post_init__(self):
        # مسارات افتراضية تحت search_bot/data/
        self.default_index_path = os.path.join(self.base_dir, "data", "index.faiss")
        self.default_rows_dir   = os.path.join(self.base_dir, "data")
        self.default_rows_file1 = os.path.join(self.base_dir, "data", "rows.jsonl.gz")
        self.default_rows_file2 = os.path.join(self.base_dir, "data", "rows.jsonl")

        # لو مفيش INDEX_PATH في ENV، استخدم index.faiss تحت data/
        if not self.index_path:
            self.index_path = self.default_index_path

        # تحديد rows_path تلقائيًا إن لم تُمرَّر
        if not self.rows_path:
            if os.path.isfile(self.default_rows_file1):
                self.rows_path = self.default_rows_file1
            elif os.path.isfile(self.default_rows_file2):
                self.rows_path = self.default_rows_file2
            else:
                self.rows_path = self.default_rows_dir

        # ===== قيود وحدود ذكية =====
        # page size
        if self.page_size_min < 1:
            self.page_size_min = 1
        if self.page_size_max < self.page_size_min:
            self.page_size_max = max(self.page_size_min, 5)
        self.page_size_default = min(max(self.page_size_default, self.page_size_min), self.page_size_max)

        # top_k
        if self.top_k_default < 1:
            self.top_k_default = 5

        # nprobe منطقي (الـ runtime cap الحقيقي بيكون nlist لو IVF)
        if self.nprobe < 1:
            self.nprobe = 1
        if self.nprobe > 256:
            self.nprobe = 256

        # أوزان الدمج: لو الاتنين صفر → رجّعها لقيم عملية
        if self.emb_weight == 0.0 and self.bm25_weight == 0.0:
            self.emb_weight, self.bm25_weight = 0.7, 0.3

        # تشابه أدنى ضمن [0, 1]
        self.min_similarity = min(max(self.min_similarity, 0.0), 1.0)

        # rerank_top_n
        if self.rerank_top_n < 1:
            self.rerank_top_n = 20
        self.rerank_top_n = min(self.rerank_top_n, max(self.faiss_candidates, self.rerank_top_n))

        # دفعات/طول reranker
        if self.reranker_batch_size < 1:
            self.reranker_batch_size = 8
        if self.reranker_max_length < 64:
            self.reranker_max_length = 256

        # Inline limits
        if self.inline_top_k < 1:
            self.inline_top_k = 5

        # ===== تطبيع باراميترات تجميع الردود =====
        if self.reply_boost < 0.0:
            self.reply_boost = 0.0
        if self.reply_count_boost < 0.0:
            self.reply_count_boost = 0.0
        if self.reply_as_parent_weight < 0.0:
            self.reply_as_parent_weight = 0.0
        if self.reply_as_parent_weight > 1.0:
            self.reply_as_parent_weight = 1.0

        # ===== تطبيع إعدادات الحماية =====
        if self.max_user_text_len < 256:   # حد منطقي أدنى
            self.max_user_text_len = 256
        if self.max_concurrency < 1:
            self.max_concurrency = 1

        # تأكد أن allowed_chat_ids هي tuple نظيفة من أرقام
        self.allowed_chat_ids = tuple(int(x) for x in self.allowed_chat_ids if isinstance(x, int) or str(x).isdigit())

        # ===== تطبيع إعدادات OpenAI =====
        # refs
        if self.openai_max_refs < 1:
            self.openai_max_refs = 5
        if self.openai_max_refs > 15:
            self.openai_max_refs = 15

        # chars per ref
        if self.openai_ref_max_chars < 200:
            self.openai_ref_max_chars = 200
        if self.openai_ref_max_chars > 2000:
            self.openai_ref_max_chars = 2000

        # replies per ref
        if self.openai_replies_per_ref < 0:
            self.openai_replies_per_ref = 0
        if self.openai_replies_per_ref > 10:
            self.openai_replies_per_ref = 10

        # sources
        if self.openai_top_sources < 0:
            self.openai_top_sources = 0
        if self.openai_top_sources > 5:
            self.openai_top_sources = 5

        # temperature
        if self.openai_temperature < 0.0:
            self.openai_temperature = 0.0
        if self.openai_temperature > 1.0:
            self.openai_temperature = 1.0

        # max tokens
        if self.openai_max_tokens < 64:
            self.openai_max_tokens = 64
        if self.openai_max_tokens > 2000:
            self.openai_max_tokens = 2000

        # context char limit
        if self.openai_context_char_limit < 1500:
            self.openai_context_char_limit = 1500
        if self.openai_context_char_limit > 30000:
            self.openai_context_char_limit = 30000

        # OpenAI concurrency
        if self.openai_max_concurrency < 1:
            self.openai_max_concurrency = 1
        if self.openai_max_concurrency > 50:
            self.openai_max_concurrency = 50
