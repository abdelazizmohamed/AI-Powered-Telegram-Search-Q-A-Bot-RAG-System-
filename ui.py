# -*- coding: utf-8 -*-
from __future__ import annotations

import calendar
import html
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# لو عندك utils الخاصة، استخدمها؛ غير كذا نوفر بدائل بسيطة
try:
    from .utils import mask_sensitive, highlight_html, build_date_str
except Exception:
    import re

    def mask_sensitive(t: str) -> str:
        t = t or ""
        t = re.sub(r"(\d{3})\d{3,}(\d{2})", r"\1***\2", t)
        t = re.sub(
            r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            r"\1***\2",
            t,
        )
        return t

    def highlight_html(text: str, query: str) -> str:
        if not text:
            return ""
        safe = html.escape(text, quote=False)
        q = (query or "").strip()
        if not q:
            return safe
        terms = [w for w in re.split(r"\s+", q) if w and not w.startswith("-")]
        for w in sorted(set(terms), key=len, reverse=True):
            try:
                safe = re.sub(rf"(?i)({re.escape(w)})", r"<b>\1</b>", safe)
            except Exception:
                pass
        return safe

    def build_date_str(m: dict) -> str:
        y = m.get("year")
        mo = m.get("month")
        d = m.get("day")
        hh = m.get("hour")
        mm = m.get("minute")
        ss = m.get("second")
        try:
            if all(x is not None for x in (y, mo, d, hh, mm, ss)):
                from datetime import datetime

                dt = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            elif all(x is not None for x in (y, mo, d)):
                from datetime import datetime

                dt = datetime(int(y), int(mo), int(d))
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
        return m.get("date_str") or m.get("date") or ""


# ملاحظة: UIBuilder يشتغل مع Config الحقيقي تبع المشروع.
# الداتاكلاس هنا فقط للـ type hints الافتراضية لو اشتغل الملف لوحده.
@dataclass
class Config:
    page_size_default: int = 10
    page_size_min: int = 3
    page_size_max: int = 20

    # إضافات مفيدة للعرض
    top_k_default: int = 100
    nprobe: int = 10

    # تخصيصات اختيارية للواجهة
    university_name: str = "جامعتك"
    intro_text: Optional[str] = None


class UIBuilder:
    # ثوابت حالات الـ Conversation
    PAGE_HOME = "STATE_PAGE_HOME"
    PAGE_SEARCH = "STATE_PAGE_SEARCH"
    PAGE_REPLIES = "STATE_PAGE_REPLIES"

    # حدود القصّ للحفاظ على طول الرسائل تحت حد تيليجرام
    PREVIEW_LIMIT = 220        # أقصى طول لمقتطف نص الرسالة
    BEST_REPLY_LIMIT = 180     # أقصى طول لمقتطف أفضل رد
    REPLY_PREVIEW_LIMIT = 240  # أقصى طول لمقتطف الرد في صفحة الردود
    QUICK_SOFT_LIMIT = 3200    # حد طري لنتيجة البحث السريع (الهاندلر لا يقسم هنا)

    # فواصل الكروت
    CARD_DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ===== Helpers =====
    @staticmethod
    def _esc(s: Optional[str]) -> str:
        return html.escape(s or "", quote=False)

    @staticmethod
    def _link_html(url: Optional[str], text: str) -> str:
        if not url:
            return html.escape(text, quote=False)
        return f'<a href="{html.escape(url, quote=True)}">{html.escape(text, quote=False)}</a>'

    def _display_name_from_seed(self, seed: Dict[str, Any]) -> str:
        name = seed.get("user") or seed.get("username") or seed.get("sender") or ""
        if not name and seed.get("chat_title"):
            name = seed["chat_title"]
        return str(name or "مستخدم")

    def _tg_link(self, seed: Dict[str, Any]) -> Optional[str]:
        """يبني رابط تيليجرام للرسالة لو توافر username/message_id أو link مباشر."""
        if seed.get("link"):
            return seed["link"]
        username = seed.get("username") or seed.get("channel")
        mid = seed.get("message_id") or seed.get("id")
        if username and mid:
            try:
                return f"https://t.me/{username}/{int(mid)}"
            except Exception:
                return None
        return None

    def _short_datetime(self, meta: Dict[str, Any]) -> str:
        """يوحّد شكل الوقت: YYYY-MM-DD HH:MM (بدون ثواني)."""
        raw = (meta.get("date_str") or "").strip()
        if not raw:
            raw = (build_date_str(meta) or "").strip()
        if not raw:
            return ""

        # أشكال شائعة: 2024-12-09T19:48:09 أو 2024-12-09 19:48:09
        raw = raw.replace("T", " ").replace("Z", "").strip()

        # لو النص طويل، قص لحد الدقائق مباشرة إن كان مطابق
        # 2024-12-09 19:48:09 -> 2024-12-09 19:48
        if len(raw) >= 16 and raw[4] == "-" and raw[7] == "-":
            # يوجد وقت؟
            if len(raw) >= 16 and raw[10] == " ":
                return raw[:16]
            # تاريخ فقط
            if len(raw) >= 10:
                return raw[:10]

        # محاولة parsing احتياطية
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(raw)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return raw

    def _card_header(self, idx1: int, seed: Dict[str, Any]) -> str:
        author = self._esc(self._display_name_from_seed(seed))
        date_s = self._esc(self._short_datetime(seed))
        # كارت: رقم كبير + سطر ميتاداتا خفيف
        if date_s and author:
            meta = f"📅 {date_s} | 👤 {author}"
        else:
            meta = " ".join([p for p in [f"📅 {date_s}" if date_s else "", f"👤 {author}" if author else ""] if p]).strip()
        return f"<b>#{idx1}</b>\n<i>{meta}</i>".strip()

    def _soft_limit_join(self, lines: List[str], limit: int) -> str:
        """يجمع الأسطر بحد أقصى (soft) ويضيف ملحوظة لو تم التقصير."""
        out: List[str] = []
        total = 0
        for ln in lines:
            add = len(ln) + 1  # +\n
            if total + add > limit:
                out.append("…")
                out.append("<i>تم تقصير العرض لتجنب تجاوز حد الرسائل. جرّب البحث المتقدم لعرض موسّع.</i>")
                break
            out.append(ln)
            total += add
        return "\n".join(out)

    def _clip_and_highlight(self, text: str, query: str, limit: int) -> str:
        raw = mask_sensitive((text or "").strip())
        if not raw:
            return "—"
        if len(raw) > limit:
            raw = raw[:limit] + "…"
        return highlight_html(raw, query or "")

    # ======= Home (الشاشة الرئيسية) =======
    def render_home_text(self) -> str:
        intro = getattr(self.cfg, "intro_text", None)
        if intro:
            return intro

        uni = getattr(self.cfg, "university_name", "جامعتك")
        return (
            f"حياك 👋\n"
            f"هذا البوت يساعد طلاب <b>{html.escape(uni)}</b> يبحثون داخل محادثات الجروب السابقة ويجيب لهم الرسائل والردود المرتبطة.\n\n"
            "اختر وحدة من الخيارات ذي وخلّنا نبدأ:\n\n"
            "• 🔎 <b>بحث سريع</b>: تكتب عبارة، ونجيب لك أفضل 10 نتائج مباشرة.\n"
            "• 🧠 <b>بحث متقدم</b>: فلاتر التاريخ/الكلمات + تنقّل + تصدير (للمشرفين فقط).\n"
            "• ℹ️ <b>معلومات عن البوت</b>: شرح مختصر وكيف تبدأ.\n\n"
            "<i>تقدر ترجع هنا بكتابة /start بأي وقت.</i>"
        )

    def build_home_menu(self) -> InlineKeyboardMarkup:
        rows = [
            [InlineKeyboardButton("🔎 بحث سريع", callback_data="home:quick")],
            [InlineKeyboardButton("🧠 بحث متقدم", callback_data="home:advanced")],
            [InlineKeyboardButton("ℹ️ معلومات عن البوت", callback_data="home:about")],
        ]
        return InlineKeyboardMarkup(rows)

    def build_quick_prompt_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("↩️ القائمة الرئيسية", callback_data="back_to_home"),
                    InlineKeyboardButton("🧠 فتح البحث المتقدم", callback_data="quick:to_advanced"),
                ]
            ]
        )

    # (سريع) بنفس ستايل النتائج المتقدمة + حد طري للطول
    def render_quick_results_text(self, query: str, items: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        q = (query or "").strip()

        lines.append("🔎 <b>نتائج البحث السريع</b>")
        if q:
            lines.append(f"🧠 الاستعلام: <code>{html.escape(q)}</code>")
        lines.append("")

        for i, it in enumerate(items[:10], 1):
            seed = it.get("seed", {}) or {}
            link = self._tg_link(seed)

            lines.append(self.CARD_DIVIDER)
            lines.append(self._card_header(i, seed))

            # متن الرسالة
            msg_html = self._clip_and_highlight(seed.get("message") or "", q, self.PREVIEW_LIMIT)
            lines.append("📝 <b>السؤال/الرسالة</b>")
            lines.append("<blockquote>")
            lines.append(msg_html)
            lines.append("</blockquote>")

            # أفضل رد (اختياري)
            if it.get("best_reply"):
                try:
                    _depth, br = it["best_reply"]
                    br_html = self._clip_and_highlight(br.get("message") or "", q, self.BEST_REPLY_LIMIT)
                    lines.append("⭐ <b>أفضل رد</b>")
                    lines.append("<blockquote>")
                    lines.append(br_html)
                    lines.append("</blockquote>")
                except Exception:
                    pass

            # ذيل: الردود + رابط
            replies = it.get("replies") or []
            nrep = len(replies)
            tail = [f"💬 الردود ({nrep})"]
            if link:
                tail.append(self._link_html(link, "🔗 فتح الرسالة"))
            lines.append(" — ".join(tail))

        lines.append("")
        lines.append("اختر إجراء من تحت:")

        return self._soft_limit_join(lines, self.QUICK_SOFT_LIMIT)

    def build_quick_results_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("📝 بحث سريع جديد", callback_data="quick:new")],
                [InlineKeyboardButton("🧠 فتح البحث المتقدم", callback_data="quick:to_advanced")],
                [InlineKeyboardButton("↩️ القائمة الرئيسية", callback_data="back_to_home")],
            ]
        )

    def build_about_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("↩️ رجوع للرئيسية", callback_data="back_to_home"),
                    InlineKeyboardButton("🧠 فتح البحث المتقدم", callback_data="home:advanced"),
                ]
            ]
        )

    def build_help_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("📘 الدليل الكامل", callback_data="home:about"),
                    InlineKeyboardButton("🏠 الرئيسية", callback_data="back_to_home"),
                ],
                [
                    InlineKeyboardButton("🔎 بحث سريع", callback_data="home:quick"),
                    InlineKeyboardButton("🧠 بحث متقدم", callback_data="home:advanced"),
                ],
            ]
        )

    def render_tldr_text(self) -> str:
        return (
            "TL;DR 👇\n\n"
            "• من /start بتشوف 3 أزرار: <b>🔎 بحث سريع</b>، <b>🧠 بحث متقدم</b>، <b>ℹ️ معلومات</b>.\n"
            "• <b>بحث سريع</b>: اكتب عبارة قصيرة وتطلع أفضل 10 نتائج.\n"
            "• <b>بحث متقدم</b>: فلاتر التاريخ/الكلمات + تنقّل + تصدير + ردود (للمشرفين)."
        )

    def render_about_text(self) -> str:
        return (
            "<b>نظرة سريعة (TL;DR)</b>\n"
            "<blockquote>\n"
            "• من <code>/start</code> عندك: 🔎 بحث سريع — 🧠 بحث متقدم — ℹ️ معلومات.\n"
            "• البحث السريع: اكتب عبارة قصيرة ويطلع لك أفضل 10 نتائج.\n"
            "• المتقدم: فلاتر بالتاريخ/الكلمات + تنقّل + ردود + تصدير.\n"
            "</blockquote>\n\n"
            "<b>عن وش يخدمك البوت؟</b>\n"
            "<blockquote>\n"
            "يبحث داخل محادثات جروب الجامعة القديمة ويجيب لك الرسائل والردود المرتبطة بموضوعك.\n"
            "</blockquote>\n\n"
            "<b>أول تشغيل</b>\n"
            "<blockquote>\n"
            "1) أرسل <code>/start</code>\n"
            "2) اختر: <b>بحث سريع</b> أو <b>بحث متقدم</b> أو <b>معلومات</b>\n"
            "</blockquote>\n"
        )

    # ======= Paging =======
    @staticmethod
    def compute_total_pages(total: int, page_size: int) -> int:
        if total <= 0 or page_size <= 0:
            return 0
        return (total + page_size - 1) // page_size

    @staticmethod
    def slice_page(items: List[Any], page_size: int, page_idx: int) -> Tuple[List[Any], int]:
        if page_size <= 0:
            return (items, 0)
        start = max(page_idx, 0) * page_size
        end = start + page_size
        return (items[start:end], start)

    # ======= Keyboards (Advanced) =======
    def _build_page_numbers_row(self, state: "ChatState") -> List[List[InlineKeyboardButton]]:
        total = state.total_pages or self.compute_total_pages(len(state.last_results or []), state.page_size)
        if not total or total <= 1:
            return []
        cur = max(state.current_page, 0)
        max_buttons = 8
        start = max(0, cur - max_buttons // 2)
        end = min(total, start + max_buttons)
        if end - start < max_buttons:
            start = max(0, end - max_buttons)

        row: List[InlineKeyboardButton] = []
        for p in range(start, end):
            if p == cur:
                row.append(InlineKeyboardButton(f"〔{p+1}〕", callback_data="noop"))
            else:
                row.append(InlineKeyboardButton(f"{p+1}", callback_data=f"page:{p}"))
        return [row] if row else []

    def build_main_menu(self, state: "ChatState") -> InlineKeyboardMarkup:
        rows: List[List[InlineKeyboardButton]] = []

        rows.append(
            [
                InlineKeyboardButton("🔎 ابدأ البحث", callback_data="start_search"),
                InlineKeyboardButton("💾 حفظ الاستعلام", callback_data="save_query"),
                InlineKeyboardButton("↻ تشغيل المحفوظ", callback_data="rerun_saved"),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton(
                    ("✅ فقط الردود" if state.only_replies else "❌ فقط الردود"),
                    callback_data="toggle_only_replies",
                ),
                InlineKeyboardButton(
                    ("✅ بها تواصل" if getattr(state, "only_with_contact", False) else "❌ بها تواصل"),
                    callback_data="toggle_only_contact",
                ),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton("🔑 كلمة مهمة", callback_data="ask_keyword"),
                InlineKeyboardButton("📄 حجم الصفحة", callback_data="ask_pagesize"),
                InlineKeyboardButton("🔢 TopK", callback_data="ask_topk"),
            ]
        )

        rows.append([InlineKeyboardButton("⚙️ NPROBE", callback_data="ask_nprobe")])

        rows.append(
            [
                InlineKeyboardButton("📅 اختيار يوم", callback_data="ask_date"),
                InlineKeyboardButton("🗓️ اختيار نطاق", callback_data="ask_date_range"),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton("🚫 تعطيل اليوم", callback_data="disable_date"),
                InlineKeyboardButton("🚫 تعطيل النطاق", callback_data="disable_daterange"),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton("آخر 7 أيام", callback_data="qf:last7"),
                InlineKeyboardButton("آخر 30 يوم", callback_data="qf:last30"),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton("آخر سنة", callback_data="qf:last365"),
                InlineKeyboardButton("الكل (افتراضي)", callback_data="qf:all"),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton("📌 تثبيت كافتراضي", callback_data="pin_filters"),
                InlineKeyboardButton("📥 تطبيق الافتراضيات", callback_data="apply_pinned"),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton("🔁 تحديث الصفحة", callback_data="refresh_page"),
                InlineKeyboardButton("♻️ مسح الفلاتر", callback_data="reset_filters"),
                InlineKeyboardButton("📝 بحث جديد", callback_data="new_search"),
            ]
        )

        rows.append([InlineKeyboardButton("🏠 الرئيسية", callback_data="back_to_home")])

        return InlineKeyboardMarkup(rows)

    def build_search_page_keyboard(self, state: "ChatState") -> InlineKeyboardMarkup:
        rows: List[List[InlineKeyboardButton]] = []

        rows.append(
            [
                InlineKeyboardButton("⬅️ السابق", callback_data="nav:prev"),
                InlineKeyboardButton("➡️ التالي", callback_data="nav:next"),
            ]
        )

        rows.extend(self._build_page_numbers_row(state))

        rows.append(
            [
                InlineKeyboardButton("📤 JSON", callback_data="export_json"),
                InlineKeyboardButton("📤 CSV", callback_data="export_csv"),
                InlineKeyboardButton("📤 HTML", callback_data="export_html"),
            ]
        )

        rows.append(
            [
                InlineKeyboardButton("🏠 القائمة (فلاتر)", callback_data="back_to_menu"),
                InlineKeyboardButton("🏠 الرئيسية", callback_data="back_to_home"),
            ]
        )

        # أزرار فتح الردود لعناصر الصفحة الحالية (💬1, 💬2, …)
        page_items, base_idx = self.slice_page(state.last_results or [], state.page_size, state.current_page)
        if page_items:
            row: List[InlineKeyboardButton] = []
            for i, _ in enumerate(page_items):
                idx = base_idx + i
                row.append(InlineKeyboardButton(f"💬{i+1}", callback_data=f"show:{idx}"))
                if len(row) == 8:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)

        return InlineKeyboardMarkup(rows)

    def build_calendar(self, year: int, month: int) -> InlineKeyboardMarkup:
        cal = calendar.Calendar(firstweekday=6)
        days = list(cal.itermonthdates(year, month))

        prev_y, prev_m = (year - 1, 12) if month == 1 else (year, month - 1)
        next_y, next_m = (year + 1, 1) if month == 12 else (year, month + 1)

        rows: List[List[InlineKeyboardButton]] = []
        title = f"{year}-{month:02d}"
        rows.append(
            [
                InlineKeyboardButton("«", callback_data=f"prevmonth:{prev_y}-{prev_m:02d}"),
                InlineKeyboardButton(title, callback_data="noop"),
                InlineKeyboardButton("»", callback_data=f"nextmonth:{next_y}-{next_m:02d}"),
            ]
        )

        rows.append([InlineKeyboardButton(w, callback_data="noop") for w in ["س", "ح", "ن", "ث", "ر", "خ", "ج"]])

        week: List[InlineKeyboardButton] = []
        this_month = month
        for d in days:
            if d.month != this_month:
                week.append(InlineKeyboardButton("·", callback_data="noop"))
            else:
                week.append(InlineKeyboardButton(f"{d.day:02d}", callback_data=f"setdate:{d.strftime('%Y-%m-%d')}"))
            if len(week) == 7:
                rows.append(week)
                week = []
        if week:
            rows.append(week)

        rows.append(
            [
                InlineKeyboardButton("↩️ رجوع", callback_data="calendar_back"),
                InlineKeyboardButton("🏠 القائمة", callback_data="back_to_menu"),
            ]
        )

        return InlineKeyboardMarkup(rows)

    # ======= Rendering (Advanced) =======
    def render_search_page_text(self, state: "ChatState") -> str:
        results = state.last_results or []
        if not results:
            return "❌ لا توجد نتائج مطابقة.\nجرّب تعديل الاستعلام أو الفلاتر."

        total_pages = self.compute_total_pages(len(results), state.page_size)
        state.total_pages = total_pages
        page_items, base_idx = self.slice_page(results, state.page_size, state.current_page)

        q = (state.query or "").strip()

        lines: List[str] = []

        # رأس الصفحة
        lines.append(f"🔎 <b>نتائج البحث</b> (صفحة {state.current_page+1}/{state.total_pages or 1})")
        total = len(results)
        shown_from = base_idx + 1
        shown_to = base_idx + len(page_items)
        lines.append(f"📊 الإجمالي: <b>{total}</b> — المعروض: <b>{shown_from}-{shown_to}</b>")
        if q:
            lines.append(f"🧠 الاستعلام: <code>{html.escape(q)}</code>")

        # فلاتر مفعّلة (مختصرة ومنسقة)
        filters_info: List[str] = []
        if state.only_replies:
            filters_info.append("ردود فقط")
        if getattr(state, "only_with_contact", False):
            filters_info.append("بها تواصل")
        if state.date:
            y, m, d = state.date
            filters_info.append(f"تاريخ: {y:04d}-{m:02d}-{d:02d}")
        if state.date_range:
            (y1, m1, d1), (y2, m2, d2) = state.date_range
            filters_info.append(f"نطاق: {y1:04d}-{m1:02d}-{d1:02d} → {y2:04d}-{m2:02d}-{d2:02d}")
        if state.keyword:
            filters_info.append(f"كلمة: {html.escape(state.keyword)}")
        filters_info.append(f"TopK={state.top_k}")
        if hasattr(self.cfg, "nprobe") and getattr(self.cfg, "nprobe", None) is not None:
            filters_info.append(f"nprobe={getattr(self.cfg, 'nprobe')}")

        if filters_info:
            lines.append("• " + " — ".join(filters_info))

        # تلميح بسيط لاستخدام أزرار الردود
        lines.append("")
        lines.append("<i>لفتح كل الردود لأي نتيجة: اضغط زر 💬 المقابل لها أسفل الكيبورد.</i>")
        lines.append("")

        # كروت النتائج
        for i, item in enumerate(page_items):
            idx1 = i + 1
            global_idx1 = base_idx + idx1

            seed = item.get("seed", {}) or {}
            link = self._tg_link(seed)

            lines.append(self.CARD_DIVIDER)
            lines.append(self._card_header(global_idx1, seed))

            # متن الرسالة
            msg_html = self._clip_and_highlight(seed.get("message") or "", q, self.PREVIEW_LIMIT)
            lines.append("📝 <b>السؤال/الرسالة</b>")
            lines.append("<blockquote>")
            lines.append(msg_html)
            lines.append("</blockquote>")

            # أفضل رد (مخفيّة باقي الردود افتراضيًا)
            best = item.get("best_reply")
            if (not best) and item.get("replies"):
                # لو الردود موجودة ولم يتم حساب best_reply: خذ أول رد
                try:
                    first = (item.get("replies") or [])[0]
                    if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
                        best = (first[0], first[1])
                except Exception:
                    best = None

            if best:
                try:
                    _depth, br = best
                    br_html = self._clip_and_highlight(br.get("message") or "", q, self.BEST_REPLY_LIMIT)
                    lines.append("⭐ <b>أفضل رد</b>")
                    lines.append("<blockquote>")
                    lines.append(br_html)
                    lines.append("</blockquote>")
                except Exception:
                    pass

            # ذيل: أفضل رد فقط + زر/إرشاد لعرض كل الردود + رابط
            # ملاحظة UX: زر عرض الردود موجود في الكيبورد أسفل النتائج بعنوان 💬1, 💬2...
            replies = item.get("replies") or []
            nrep = len(replies)

            if nrep > 0:
                # رقم الزر يطابق ترتيب النتيجة داخل الصفحة الحالية
                tail_main = f"💬 عرض الردود ({nrep}) — اضغط زر <b>💬{i+1}</b>"
            else:
                tail_main = "💬 لا يوجد ردود"

            tail: List[str] = [tail_main]
            if link:
                tail.append(self._link_html(link, "🔗 فتح الرسالة"))
            lines.append(" — ".join(tail))

        return "\n".join(lines).strip()

    def render_replies_page_text(self, item: Dict[str, Any], query: Optional[str]) -> str:
        q = (query or "").strip()

        seed = item.get("seed", {}) or {}
        link = self._tg_link(seed)

        # كارت رأس للرسالة الأصلية
        lines: List[str] = []
        lines.append(self.CARD_DIVIDER)
        lines.append(self._card_header(0, seed).replace("#0", "🧾 <b>الرسالة الأصلية</b>"))

        msg_html = self._clip_and_highlight(seed.get("message") or "", q, 1200)
        lines.append("<blockquote>")
        lines.append(msg_html)
        lines.append("</blockquote>")

        if link:
            lines.append(self._link_html(link, "🔗 فتح الرسالة"))

        lines.append("")

        replies = item.get("replies", []) or []
        if not replies:
            lines.append("💬 لا يوجد ردود مطابقة للفلترة الحالية.")
            return "\n".join(lines).strip()

        lines.append("💬 <b>الردود</b>")
        lines.append("")

        for (depth, r) in replies:
            r = r or {}

            rauthor = self._esc(r.get("user") or r.get("username") or "مستخدم")
            rdate = self._esc(self._short_datetime(r))
            rlink = self._tg_link(r)

            # كارت رد
            lines.append("────────────")
            meta = []
            if rdate:
                meta.append(f"📅 {rdate}")
            meta.append(f"👤 {rauthor}")
            meta.append(f"عمق={depth}")
            lines.append("<i>" + " | ".join(meta) + "</i>")

            r_html = self._clip_and_highlight(r.get("message") or "", q, self.REPLY_PREVIEW_LIMIT)
            lines.append("<blockquote>")
            lines.append(r_html)
            lines.append("</blockquote>")

            if rlink:
                lines.append(self._link_html(rlink, "↳ 🔗 فتح الرد"))

        return "\n".join(lines).strip()
