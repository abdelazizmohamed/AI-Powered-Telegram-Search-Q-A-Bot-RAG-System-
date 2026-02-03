# -*- coding: utf-8 -*-
import time
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Deque, DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict, deque
from .config import Config


@dataclass
class ChatState:
    """
    حالة المحادثة لكل شات:
    - فلاتر البحث (ردود/تواصل/تاريخ/نطاق/كلمة مهمة)
    - أعلام الإدخال التفاعلي (وش قاعدين ننتظر من المستخدم؟)
    - وضع البحث السريع (quick search)
    - إعدادات العرض والبحث (top_k / page_size / max_depth)
    - حالة النتائج الحالية (الصفحة/المجموع/الرسائل المرسلة ... إلخ)
    - أعلام إدارة الرد الذكي للأدمن
    """
    # داخل dataclass ChatState:
    expecting_nprobe: bool = False

    # لمنع تمرير نفس الرسالة كاستعلام مباشرة بعد تغيير رقم/إعداد
    suppress_next_text: bool = False
    # لمنع تكرار /start مرتين بسرعة
    last_start_ts: float = 0.0

    # ===== فلاتر =====
    only_replies: bool = False
    only_with_contact: bool = False  # رسائل بها رقم/يوزر فقط
    date: Optional[Tuple[int, int, int]] = None
    date_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    keyword: Optional[str] = None

    # نقاط مساعدة للتقويم (يوم/نطاق)
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None

    # تثبيت فلاتر كافتراضية (pin/apply)
    pinned: Optional[Dict[str, Any]] = None

    # ===== أسئلة الإدخال التفاعلية =====
    expecting_date: bool = False
    expecting_date_range: bool = False
    expecting_topk: bool = False
    expecting_pagesize: bool = False
    expecting_query: bool = False
    expecting_keyword: bool = False
    expecting_nprobe: bool = False  

    # ===== وضع البحث السريع =====
    expecting_quick_query: bool = False
    quick_query: Optional[str] = None
    last_quick_results: List[Dict[str, Any]] = field(default_factory=list)

    # ===== إعدادات البحث/الواجهة =====
    top_k: int = 10
    page_size: int = 5
    max_depth: int = 5

    # ===== الحالة/النتائج (المتقدم) =====
    query: Optional[str] = None
    saved_query: Optional[Dict[str, Any]] = None
    last_results: List[Dict[str, Any]] = field(default_factory=list)
    result_message_ids: List[int] = field(default_factory=list)
    reply_page_message_ids: List[int] = field(default_factory=list)
    current_page: int = 0
    total_pages: int = 0
    last_page_before_replies: int = 0

    # ===== إدارة “الرد الذكي” للأدمن =====
    expecting_admin_smart_reply: bool = False
    pending_smart_req_id: Optional[str] = None

    # أدوات مساعدة اختيارية (ممكن تستخدمها من الهاندلرز لو حبيت)
    def clear_transient_flags(self) -> None:
        """يمسح أعلام الانتظار المؤقتة + مؤشرات النطاق."""
        self.expecting_date = False
        self.expecting_date_range = False
        self.expecting_topk = False
        self.expecting_pagesize = False
        self.expecting_query = False
        self.expecting_keyword = False
        self.expecting_quick_query = False
        self.date_range_start = None
        self.date_range_end = None


@dataclass
class AdminStats:
    unique_users: set[int] = field(default_factory=set)
    user_profile: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # {uid: {"name":..., "username":..., "first_seen": ts, "last_seen": ts}}
    user_searches: DefaultDict[int, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))  # {uid: [{"t": ts, "q": "...", "mode": "quick/adv"}]}


class StateManager:
    """
    مدير حالة الشات:
    - ينشئ حالة افتراضية مضبوطة حسب Config
    - reset يحافظ على pinned و saved_query (منطقي أكثر لزر "مسح الفلاتر")
    - check_rate_limit: حد بسيط للطلبات لكل شات
    - تجميع إحصاءات للأدمن
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.filters_state: DefaultDict[int, ChatState] = defaultdict(self._default_state)
        self.rate_store: DefaultDict[int, Deque[float]] = defaultdict(lambda: deque())
        # 🛠️ إحصاءات الأدمن
        self.admin_stats = AdminStats()

    def _default_state(self) -> ChatState:
        s = ChatState()
        # اربط القيم الافتراضية بـ Config مع القيود المناسبة
        # page_size ضمن [min, max]
        ps_min = max(1, getattr(self.cfg, "page_size_min", 3))
        ps_max = max(ps_min, getattr(self.cfg, "page_size_max", 20))
        ps_def = getattr(self.cfg, "page_size_default", 5)
        s.page_size = min(max(ps_def, ps_min), ps_max)

        # top_k لا يقل عن 1
        tk_def = getattr(self.cfg, "top_k_default", 7)
        s.top_k = max(1, tk_def)

        # عمق الردود
        s.max_depth = getattr(self.cfg, "max_depth", 1)
        return s

    # ========= تتبُّع وإحصاءات الأدمن =========
    def track_user_seen(self, uid: int, name: str = "", username: str = "") -> None:
        """نادِ هذه الدالة عند /start أو أي نشاط يُظهر وجود المستخدم."""
        now = time.time()
        self.admin_stats.unique_users.add(uid)
        p = self.admin_stats.user_profile.get(uid) or {}
        p.setdefault("first_seen", now)
        p["last_seen"] = now
        if name:
            p["name"] = name
        if username:
            p["username"] = username
        self.admin_stats.user_profile[uid] = p

    def track_search(self, uid: int, query: str, mode: str = "adv") -> None:
        """سجِّل عملية بحث للمستخدم (mode: "quick" | "adv")."""
        self.admin_stats.user_searches[uid].append({
            "t": time.time(),
            "q": (query or "").strip(),
            "mode": mode,
        })

    def get_admin_snapshot(self, limit: int = 10) -> Dict[str, Any]:
        """
        يُرجع ملخصًا جاهزًا للعرض في لوحة الأدمن:
        - users_count: عدد المستخدمين الفريدين
        - searches_count: مجموع عمليات البحث
        - recent_users: آخر مستخدمين شوهدوا
        - recent_queries: آخر الاستعلامات (من كافة المستخدمين)
        """
        profiles = self.admin_stats.user_profile
        searches = self.admin_stats.user_searches

        # آخر مستخدمين شوهدوا (حسب last_seen)
        recent_users = sorted(
            (
                {
                    "id": uid,
                    "name": profiles.get(uid, {}).get("name", ""),
                    "username": profiles.get(uid, {}).get("username", ""),
                    "last_seen": profiles.get(uid, {}).get("last_seen", 0.0),
                }
                for uid in profiles.keys()
            ),
            key=lambda x: x["last_seen"],
            reverse=True
        )[:limit]

        # صيغة زمن قابلة للعرض
        for ru in recent_users:
            ts = ru.get("last_seen", 0.0) or 0.0
            ru["ts_h"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

        # آخر الاستعلامات من كل المستخدمين
        all_q: List[Dict[str, Any]] = []
        for uid, lst in searches.items():
            for rec in lst:
                all_q.append({
                    "uid": uid,
                    "t": rec.get("t", 0.0),
                    "q": rec.get("q", ""),
                    "mode": rec.get("mode", "adv"),
                })
        recent_queries = sorted(all_q, key=lambda x: x["t"], reverse=True)[:limit]
        for r in recent_queries:
            r["ts_h"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("t", 0.0)))

        return {
            "users_count": len(self.admin_stats.unique_users),
            "searches_count": sum(len(v) for v in searches.values()),
            "recent_users": recent_users,
            "recent_queries": recent_queries,
        }

    # ========= إدارة الحالات =========
    def get(self, chat_id: int) -> ChatState:
        return self.filters_state[chat_id]

    def reset(self, chat_id: int) -> ChatState:
        """
        إعادة ضبط الفلاتر والإعدادات المؤقتة، مع الإبقاء على:
        - pinned: الافتراضيات المثبّتة
        - saved_query: البحث المحفوظ
        """
        old = self.filters_state.get(chat_id)
        new_state = self._default_state()
        if old:
            new_state.pinned = old.pinned
            new_state.saved_query = old.saved_query
        self.filters_state[chat_id] = new_state
        return self.filters_state[chat_id]

    def check_rate_limit(self, chat_id: int, max_per_min: int = 6) -> bool:
        """بسيط: يسمح بـ max_per_min طلب/دقيقة لكل شات."""
        now = time.time()
        dq = self.rate_store[chat_id]
        while dq and dq[0] < now - 60:
            dq.popleft()
        if len(dq) >= max_per_min:
            return False
        dq.append(now)
        return True
