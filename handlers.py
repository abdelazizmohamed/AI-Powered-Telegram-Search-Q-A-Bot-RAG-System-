#-*- coding: utf-8 -*-
import os
import io, csv, json, html, asyncio, logging, re
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update,
    InlineQueryResultArticle, InputTextMessageContent,
)
from telegram import constants as C
from telegram.ext import ContextTypes, ConversationHandler
from telegram.error import BadRequest

from .config import Config
from .state import ChatState, StateManager
from .ui import UIBuilder
from .search import SearchEngine, SearchFilters
from .openai_answerer import OpenAIAnswerer

logger = logging.getLogger(__name__)


class Handlers:
    # حالات اختيار التاريخ (تقويم)
    DATE_PICKER = "STATE_DATE_PICKER"
    DATE_PICKER_RANGE_START = "STATE_DATE_RANGE_START"
    DATE_PICKER_RANGE_END = "STATE_DATE_RANGE_END"

    def __init__(
        self,
        cfg: Config,
        ui: UIBuilder,
        state_mgr: StateManager,
        search: SearchEngine,
        answerer: Optional[OpenAIAnswerer] = None
    ):
        self.cfg = cfg
        self.ui = ui
        self.state_mgr = state_mgr
        self.search_engine = search
        self.answerer = answerer

        # قائمة طلبات "الرد الذكي" المعلّقة
        self.smart_requests: Dict[str, Dict[str, Any]] = {}

        # 🔒 حد التوازي لمنع الضغط/DoS
        self._sem = asyncio.Semaphore(getattr(self.cfg, "max_concurrency", 20))

    # ---------- helpers ----------
    def _reset_expect_flags(self, state: ChatState):
        state.expecting_nprobe = False
        state.expecting_date = False
        state.expecting_date_range = False
        state.expecting_topk = False
        state.expecting_pagesize = False
        state.expecting_query = False
        state.expecting_keyword = False
        state.expecting_quick_query = False
        # لإدارة رد الأدمن على طلب "الرد الذكي"
        state.expecting_admin_smart_reply = False
        state.pending_smart_req_id = None

        state.date_range_start = None
        state.date_range_end = None

    @staticmethod
    def _strip_text(s: Optional[str]) -> str:
        return (s or "").strip()

    @staticmethod
    def _rewrite_query_for_search(q: str) -> str:
        """تبسيط سريع لأسئلة "القوائم" والأسئلة الطويلة لتحسين الاسترجاع."""
        q = (q or "").strip()
        if not q:
            return q
        low = q
        # أسئلة من نوع "هات قائمة بكل ..." غالبًا تحتاج كلمات مفتاحية فقط
        if any(k in low for k in ("قائمة", "كل", "جميع", "هاتلي", "هات", "اذكر")):
            # heuristic: ابتعاث + جامعة
            if ("ابتعاث" in low) and ("جامعة" in low or "الجامعات" in low):
                return "جامعة ابتعاث"
            # خُد أهم كلمتين/ثلاثة
            toks = [t for t in re.split(r"\s+", low) if t and len(t) > 2]
            toks = [t for t in toks if t not in {"هاتلي", "هات", "اذكر", "قائمة", "كل", "جميع", "الي", "اللي", "في", "من", "عن", "مع"}]
            return " ".join(toks[:4]) if toks else q
        return q

    @staticmethod
    def _now_str() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _delete_messages_safely(bot, chat_id: int, msg_ids: List[int]):
        async def _runner(coro):
            try:
                await coro
            except Exception as e:
                logger.debug("delete_message error: %s", e)

        for mid in list(msg_ids or []):
            try:
                asyncio.create_task(_runner(bot.delete_message(chat_id=chat_id, message_id=mid)))
            except Exception:
                pass

    def _faiss_probe_limits(self) -> Tuple[Optional[int], Optional[int]]:
        """
        يرجّع (nprobe الحالي، الحد الأقصى nlist) لو الفهرس IVF.
        غير ذلك يرجّع (cfg.nprobe أو None، None).
        """
        try:
            import faiss  # type: ignore
            idx = getattr(self.search_engine.index, "index", None)
            if isinstance(idx, faiss.IndexIVF):
                cur = int(getattr(idx, "nprobe", 0))
                mx = int(idx.nlist)
                return cur if cur > 0 else None, mx
        except Exception:
            pass
        # فهارس غير IVF
        return (getattr(self.cfg, "nprobe", None), None)

    def _is_admin(self, user_id: Optional[int]) -> bool:
        try:
            return bool(self.cfg.admin_ids and user_id in self.cfg.admin_ids)
        except Exception:
            return False

    def _guard_chat_whitelist(self, chat_id: int) -> bool:
        """يرجع True إذا الشات مسموح له؛ وإلا False عند تفعيل whitelist."""
        try:
            if getattr(self.cfg, "enable_whitelist", False):
                allowed = set(getattr(self.cfg, "allowed_chat_ids", []) or [])
                return chat_id in allowed
            return True
        except Exception:
            # في حال أي خطأ، لا نمنع بشكل صلب
            return True

    def _build_home_menu(self, is_admin: bool):
        rows: List[List[InlineKeyboardButton]] = []
        rows.append([InlineKeyboardButton("ابحث في القروب الان  🔎", callback_data="home:quick")])
        if is_admin:
            rows.append([InlineKeyboardButton("بحث متقدم", callback_data="home:advanced")])
            rows.append([InlineKeyboardButton("🛠️ لوحة الأدمن", callback_data="admin:dashboard")])
        rows.append([InlineKeyboardButton("عن البوت", callback_data="home:about")])
        return InlineKeyboardMarkup(rows)

    def _quick_prompt_kb(self, is_admin: bool) -> InlineKeyboardMarkup:
        # للمستخدم العادي: كيبورد بسيط بدون أي تحويل للبحث المتقدم
        if not is_admin:
            return InlineKeyboardMarkup([[InlineKeyboardButton("🏠 رجوع", callback_data="back_to_home")]])
        # للأدمن: استخدم كيبورد الواجهة المعتادة (قد تحتوي زر انتقال للمتقدم)
        return self.ui.build_quick_prompt_keyboard()

    def _with_smart_button(self, kb: Optional[InlineKeyboardMarkup]) -> InlineKeyboardMarkup:
        rows = list(getattr(kb, "inline_keyboard", [])) if kb else []
        rows = [list(r) for r in rows]
        rows.append([InlineKeyboardButton("🧠 توليد رد ذكي", callback_data="gen_ai_reply")])
        return InlineKeyboardMarkup(rows)

    def _build_sources_section(self, results: List[Dict[str, Any]], max_refs: int = 8) -> str:
        """يبني قسم مصادر بشكل "كروت" واضح + أفضل رد فقط (HTML آمن لتيليجرام)."""
        from .utils import make_tg_link, safe_truncate, build_date_str

        def fmt_dt(meta: Dict[str, Any]) -> str:
            """Normalize date/time to: YYYY-MM-DD HH:MM (no seconds)."""
            # Prefer numeric fields if present
            try:
                y = meta.get("year")
                mo = meta.get("month")
                d = meta.get("day")
                hh = meta.get("hour")
                mm = meta.get("minute")
                if all(v is not None for v in (y, mo, d, hh, mm)):
                    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d} {int(hh):02d}:{int(mm):02d}"
            except Exception:
                pass

            s = (meta.get("date_str") or build_date_str(meta) or "").strip()
            if not s:
                return ""
            s = s.replace("T", " ").replace("Z", "").strip()

            # YYYY-MM-DD HH:MM(:SS)
            m = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})", s)
            if m:
                return f"{m.group(1)} {m.group(2)}"

            # Weird legacy: DDTHH:MM:SS-MM-YYYY (or -MM-YYYY)
            m = re.search(r"(\d{2})\s*T\s*(\d{2}:\d{2})(?::\d{2})?\s*-\s*(\d{2})\s*-\s*(\d{4})", s)
            if m:
                dd, tm, mo2, yy = m.group(1), m.group(2), m.group(3), m.group(4)
                return f"{yy}-{int(mo2):02d}-{int(dd):02d} {tm}"

            # Fallback: show first 16 chars if it looks like a datetime
            if len(s) >= 16 and re.search(r"\d{2}:\d{2}", s):
                return s[:16]
            return s

        def compact_text(t: str, limit: int) -> str:
            t = (t or "").strip().replace("\n", " ")
            return safe_truncate(t, limit)

        cards: List[str] = []
        for i, it in enumerate((results or [])[:max_refs], start=1):
            seed = (it.get("seed") or {})
            msg = (seed.get("message") or "")
            user = (seed.get("user") or seed.get("username") or seed.get("sender") or "").strip()
            dt = fmt_dt(seed)

            link = (seed.get("link") or "").strip() or make_tg_link(seed.get("chat_id"), seed.get("message_id"))

            # message body (2-4 lines 느낌) — keep compact
            body = compact_text(msg, 220)
            body_esc = html.escape(body)
            user_esc = html.escape(user or "مستخدم")
            dt_esc = html.escape(dt)

            lines: List[str] = []
            lines.append("────────────────────")
            lines.append(f"<b>#{i}</b>")
            if dt_esc or user_esc:
                lines.append(f"<i>📅 {dt_esc} | 👤 {user_esc}</i>")
            lines.append(f"📝 {body_esc}")

            # replies: show only best reply (if any) + count
            reps = it.get("replies") or []
            reps_count = len(reps)
            br_meta: Optional[Dict[str, Any]] = None
            br_tuple = it.get("best_reply")
            if isinstance(br_tuple, (list, tuple)) and len(br_tuple) == 2 and isinstance(br_tuple[1], dict):
                br_meta = br_tuple[1]
            elif reps and isinstance(reps[0], (list, tuple)) and len(reps[0]) >= 2 and isinstance(reps[0][1], dict):
                br_meta = reps[0][1]

            if reps_count:
                lines.append("")
                lines.append(f"💬 <b>الردود ({reps_count})</b>")
                if br_meta:
                    r_user = (br_meta.get("user") or br_meta.get("username") or "").strip()
                    r_dt = fmt_dt(br_meta)
                    r_msg = compact_text(br_meta.get("message") or "", 160)
                    lines.append(f"↳ <i>📅 {html.escape(r_dt)} | 👤 {html.escape(r_user or 'مستخدم')}</i>")
                    lines.append(f"   {html.escape(r_msg)}")

            if link:
                lines.append(f"🔗 <a href=\"{html.escape(link)}\">فتح الرسالة</a>")

            cards.append("\n".join(lines).strip())

        if not cards:
            return ""

        return "\n".join(cards).strip()

    async def _encode_query_safely(self, text: str):
        try:
            enc = self.search_engine.model.encode
        except Exception as e:
            logger.warning("model.encode not available: %s", e)
            return None

        try:
            res = enc(text)
            if asyncio.iscoroutine(res):
                return await res
            return res
        except Exception as e:
            logger.warning("encode failed: %s", e)
            return None

    async def _send_long_text(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None):
        """يقسم النص الطويل (حد ~4096) ويرسله على دفعات، ويرجع قائمة IDs للرسائل المرسلة."""
        CHUNK = 3500
        parts: List[str] = []
        t = (text or "").strip()
        while t:
            if len(t) <= CHUNK:
                parts.append(t)
                break
            cut = t.rfind("\n\n", 0, CHUNK)
            if cut == -1:
                cut = t.rfind("\n", 0, CHUNK)
            if cut == -1:
                cut = CHUNK
            parts.append(t[:cut])
            t = t[cut:].lstrip()

        sent_ids: List[int] = []
        for i, p in enumerate(parts):
            rm = reply_markup if (i == len(parts) - 1) else None
            msg = await context.bot.send_message(
                chat_id=chat_id,
                text=p,
                reply_markup=rm,
                parse_mode=C.ParseMode.HTML,
                disable_web_page_preview=True,
            )
            sent_ids.append(msg.message_id)
        return sent_ids

    async def _maybe_openai_answer(
        self,
        question: str,
        results: List[Dict[str, Any]],
        chat_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        status_message_id: Optional[int] = None,
        back_markup: Optional[InlineKeyboardMarkup] = None,
    ) -> bool:
        """
        لو OpenAI مفعّل: يولّد إجابة من نتائج البحث ويرسلها/يعدل رسالة الحالة.
        يرجّع True لو اتنفّذ الرد عبر OpenAI، وإلا False.
        """
        if not getattr(self.cfg, "use_openai_answer", False):
            return False
        if not self.answerer:
            return False
        if not (getattr(self.cfg, "openai_api_key", "") or os.environ.get("OPENAI_API_KEY")):
            return False
        if not results:
            return False

        max_refs = int(getattr(self.cfg, "openai_max_refs", 8))
        max_chars = int(getattr(self.cfg, "openai_ref_max_chars", 800))

        # --- OpenAI streaming (فقرة بفقرة) ---
        loop = asyncio.get_running_loop()

        # لو ما عندنا رسالة حالة، أنشئ واحدة
        if not status_message_id:
            try:
                msg = await context.bot.send_message(
                    chat_id=chat_id,
                    text="✍️ جاري توليد الإجابة…",
                    reply_markup=back_markup,
                    parse_mode=C.ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                status_message_id = msg.message_id
            except Exception:
                status_message_id = None

        # ⚠️ مهم: asyncio.Queue ليست thread-safe — لازم نستخدم call_soon_threadsafe من الثريد
        q: asyncio.Queue = asyncio.Queue()

        def _threadsafe_put(item: Any):
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except Exception:
                # لو اللوب اتقفل/حصل خطأ، نتجاهل
                pass

        def _producer():
            try:
                for delta in self.answerer.stream_answer(
                    question, results, max_items=max_refs, max_chars=max_chars
                ):
                    _threadsafe_put(delta)
            except Exception as e:
                _threadsafe_put(f"⚠️ حصل خطأ أثناء توليد الإجابة: {e}")
            finally:
                _threadsafe_put(None)

        # شغّل المنتج في ثريد لتفادي حجز الـ event loop
        prod_task = asyncio.create_task(asyncio.to_thread(_producer))

        raw_accum = ""
        buf = ""
        last_edit = loop.time()
        min_edit_interval = float(getattr(self.cfg, "stream_edit_interval", 0.9))  # seconds
        max_live_chars = int(getattr(self.cfg, "stream_live_max_chars", 3500))

        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                buf += str(item)

                # فقرة بفقرة: نفلش عند أول \n\n
                flushed = False
                while "\n\n" in buf:
                    part, buf = buf.split("\n\n", 1)
                    raw_accum += part + "\n\n"
                    flushed = True

                now = loop.time()
                if flushed and status_message_id and (now - last_edit) >= min_edit_interval:
                    # عرض جزء حي (بدون المصادر حالياً)
                    preview = raw_accum
                    if len(preview) > max_live_chars:
                        preview = preview[-max_live_chars:]
                        preview = "…\n" + preview

                    live = html.escape(preview) + "\n\n<i>…</i>"
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_message_id,
                            text=live,
                            reply_markup=back_markup,
                            parse_mode=C.ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                        last_edit = now
                    except Exception:
                        pass

            # انتظر المنتج لو لسه شغال
            try:
                await prod_task
            except Exception:
                pass

        except Exception as e:
            logger.exception("OpenAI stream loop failed: %s", e)
            try:
                if status_message_id:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message_id,
                        text="⚠️ حصل خطأ أثناء توليد الرد الذكي. هنعرض النتائج العادية بدل ذلك.",
                        reply_markup=back_markup,
                        parse_mode=C.ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
            except Exception:
                pass
            return False

        # النص النهائي (مع أي بواقي)
        raw_final = (raw_accum + buf).strip()
        if not raw_final:
            return False

        # نفس تنظيف answerer
        raw_final = raw_final.replace("✅ إجابة مختصرة", "✅ الإجابة")
        raw_final = raw_final.replace("🧾 تفاصيل", "🧠 الشرح")
        raw_final = re.sub(r"🔗\s*مصادر.*$", "", raw_final, flags=re.DOTALL).strip()
        # Format & escape for Telegram HTML (prefer OpenAIAnswerer formatter)
        final_answer = None
        if self.answerer and hasattr(self.answerer, '_format_final_output'):
            try:
                final_answer = self.answerer._format_final_output(raw_final)
            except Exception:
                final_answer = None
        if final_answer is None:
            final_answer = html.escape(raw_final)

        # ✅ أضف المصادر (روابط + اقتباسات) من النتائج نفسها
        sources = self._build_sources_section(results, max_refs=max_refs)
        if sources:
            final_answer = f"{final_answer}\n\n📌 <b>المصادر</b>\n{sources}"

        # لو الإجابة طويلة: ابعتها Chunked
        if len(final_answer) > 3500:
            if status_message_id:
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=status_message_id)
                except Exception:
                    pass
            ids = await self._send_long_text(chat_id, context, final_answer, reply_markup=back_markup)
            st = self.state_mgr.get(chat_id)
            st.result_message_ids.extend(ids)
            return True

        # تعديل رسالة الحالة لو موجودة
        if status_message_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text=final_answer,
                    reply_markup=back_markup,
                    parse_mode=C.ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                st = self.state_mgr.get(chat_id)
                st.result_message_ids.append(status_message_id)
                return True
            except Exception:
                pass

        # إرسال رسالة جديدة لو التعديل فشل
        out = await context.bot.send_message(
            chat_id=chat_id,
            text=final_answer,
            reply_markup=back_markup,
            parse_mode=C.ParseMode.HTML,
            disable_web_page_preview=True,
        )
        st = self.state_mgr.get(chat_id)
        st.result_message_ids.append(out.message_id)
        return True

    # ---------- /start ----------
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        import time
        chat_id = update.effective_chat.id

        # ✅ whitelist (اختياري)
        if not self._guard_chat_whitelist(chat_id):
            await context.bot.send_message(chat_id=chat_id, text="❌ هذا البوت غير مفعّل لهذا القروب.")
            return ConversationHandler.END

        uid = update.effective_user.id
        name = (update.effective_user.full_name or "").strip()
        uname = (update.effective_user.username or "") and f"@{update.effective_user.username}" or ""
        self.state_mgr.track_user_seen(uid, name=name, username=uname)
        logger.info("/start called chat_id=%s", chat_id)
        state = self.state_mgr.get(chat_id)
        now = time.time()
        if now - getattr(state, "last_start_ts", 0.0) < 2.0:
            return self.ui.PAGE_HOME
        state.last_start_ts = now

        self._reset_expect_flags(state)

        is_admin = self._is_admin(update.effective_user.id)
        text = self.ui.render_home_text()
        home_kb = self._build_home_menu(is_admin=is_admin)

        try:
            out = await update.message.reply_text(
                text,
                reply_markup=home_kb,
                parse_mode=C.ParseMode.HTML,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.exception("start: reply_text failed — trying bot.send_message; err=%s", e)
            out = await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=home_kb,
                parse_mode=C.ParseMode.HTML,
                disable_web_page_preview=True,
            )

        state.result_message_ids.append(out.message_id)
        logger.info("/start finished → PAGE_HOME")
        return self.ui.PAGE_HOME

    # ---------- callback router ----------
    async def callback_router(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = update.effective_chat.id

        # ✅ whitelist (اختياري)
        if not self._guard_chat_whitelist(chat_id):
            await query.answer("غير مسموح بهذا القروب", show_alert=True)
            return ConversationHandler.END

        data = (query.data or "")
        if data.startswith("home:") or data.startswith("quick:") or data == "back_to_home":
            return await self.buttons_in_home(update, context)
        if data == "back_to_search_same_page":
            return await self.buttons_in_replies(update, context)
        return await self.buttons_in_search(update, context)

    # ---------- home ----------
    async def buttons_in_home(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        chat_id = update.effective_chat.id
        state = self.state_mgr.get(chat_id)
        data = query.data
        logger.info("buttons_in_home HIT data=%s chat_id=%s", data, chat_id)

        is_admin = self._is_admin(update.effective_user.id)

        def _home_kb():
            return self._build_home_menu(is_admin=is_admin)

        if data == "home:quick":
            self._reset_expect_flags(state)
            state.expecting_quick_query = True
            txt = (
                "✍️ اكتب عبارة البحث (سطر واحد)، وبعطيك أفضل 10 نتائج مباشرة.\n\n"
                "مثال: <code>احد يعرف الدكتور الي يدرس مادة الرسم الهندسي ايش اسمه؟</code>"
            )
            try:
                await query.edit_message_text(
                    txt, reply_markup=self._quick_prompt_kb(is_admin), parse_mode=C.ParseMode.HTML
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id, text=txt, reply_markup=self._quick_prompt_kb(is_admin), parse_mode=C.ParseMode.HTML
                )
            return self.ui.PAGE_HOME

        if data == "home:advanced":
            if not is_admin:
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME

            self._reset_expect_flags(state)
            txt = "اختر فلاتر البحث من الأزرار ثم ابدأ البحث."
            try:
                await query.edit_message_text(
                    txt, reply_markup=self.ui.build_main_menu(state), parse_mode=C.ParseMode.HTML
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id, text=txt, reply_markup=self.ui.build_main_menu(state), parse_mode=C.ParseMode.HTML
                )
            return self.ui.PAGE_SEARCH

        if data == "home:about":
            guide = self.ui.render_about_text()
            ids = await self._send_long_text(
                chat_id, context, guide, reply_markup=self._build_home_menu(is_admin)
            )
            state.result_message_ids.extend(ids)
            return self.ui.PAGE_HOME

        if data == "back_to_home":
            self._reset_expect_flags(state)
            try:
                await query.edit_message_text(
                    self.ui.render_home_text(), reply_markup=_home_kb(), parse_mode=C.ParseMode.HTML
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id, text=self.ui.render_home_text(), reply_markup=_home_kb(), parse_mode=C.ParseMode.HTML
                )
            return self.ui.PAGE_HOME

        # quick actions
        if data == "quick:new":
            self._reset_expect_flags(state)
            state.expecting_quick_query = True
            txt = "📝 اكتب عبارة البحث الجديدة:"
            try:
                await query.edit_message_text(
                    txt, reply_markup=self._quick_prompt_kb(is_admin), parse_mode=C.ParseMode.HTML
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id, text=txt, reply_markup=self._quick_prompt_kb(is_admin), parse_mode=C.ParseMode.HTML
                )
            return self.ui.PAGE_HOME

        if data == "quick:to_advanced":
            # ممنوع لغير الأدمن
            if not is_admin:
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            txt = "تمام! افتحنا لك البحث المتقدم. اضبط الفلاتر واضغط «ابدأ البحث»."
            try:
                await query.edit_message_text(
                    txt, reply_markup=self.ui.build_main_menu(state), parse_mode=C.ParseMode.HTML
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id, text=txt, reply_markup=self.ui.build_main_menu(state), parse_mode=C.ParseMode.HTML
                )
            return self.ui.PAGE_SEARCH

        return self.ui.PAGE_HOME

    # ---------- البحث السريع ----------
    async def _run_quick_search_and_show(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE, state: ChatState):
        q = getattr(state, "quick_query", "") or ""
        if not q.strip():
            out = await context.bot.send_message(
                chat_id=chat_id,
                text="اكتب عبارة البحث أولًا.",
                reply_markup=self._quick_prompt_kb(is_admin=False),
                parse_mode=C.ParseMode.HTML,
            )
            state.result_message_ids.append(out.message_id)
            return

        # تنظيف أي رسائل نتائج سابقة
        self._delete_messages_safely(context.bot, chat_id, state.result_message_ids)
        state.result_message_ids.clear()

        # رسالة حالة
        status = await context.bot.send_message(
            chat_id=chat_id, text="⏳ عزيزي الطالب… جاري البحث …", parse_mode=C.ParseMode.HTML
        )

        # الردود فقط + TopK=100 + صفحة 10
        state.only_replies = True
        state.top_k = 100
        state.page_size = 10
        state.query = q  # نخزن الاستعلام في نفس الحقل المستخدم لعرض صفحة النتائج

        # تحسين الاستعلام (خصوصًا أسئلة القوائم)
        q_search = self._rewrite_query_for_search(q)

        flt = SearchFilters(
            only_with_replies=True,
            date_filter=None,
            date_range=None,
            keyword=None,
            only_with_contact=False,
        )

        async with self._sem:
            results = await self.search_engine.search(query=q_search, top_k=state.top_k, flt=flt)

            # لو ما فيش نتائج، جرّب الاستعلام الأصلي كخطة بديلة
            if (not results) and (q_search != q):
                results = await self.search_engine.search(query=q, top_k=state.top_k, flt=flt)

        # ✅ OpenAI: إجابة نهائية مباشرة بدل صفحة النتائج
        answered = await self._maybe_openai_answer(
            question=q,
            results=results or [],
            chat_id=chat_id,
            context=context,
            status_message_id=status.message_id,
            back_markup=self._quick_prompt_kb(is_admin=False),
        )
        if answered:
            return

        if not results:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status.message_id,
                    text="❌ ما لقيت نتائج. جرّب تصيغ العبارة بشكل أدق أو كلمة ثانية.",
                    reply_markup=self._quick_prompt_kb(is_admin=False),
                    parse_mode=C.ParseMode.HTML,
                )
                state.result_message_ids.append(status.message_id)
            except Exception:
                out = await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ ما لقيت نتائج. جرّب تصيغ العبارة بشكل أدق أو كلمة ثانية.",
                    reply_markup=self._quick_prompt_kb(is_admin=False),
                    parse_mode=C.ParseMode.HTML,
                )
                state.result_message_ids.append(out.message_id)
            return

        # خزّن النتائج في last_results (عشان نستخدم صفحة النتائج + فتح الردود)
        state.last_results = results or []
        state.total_pages = self.ui.compute_total_pages(len(state.last_results), state.page_size)
        state.current_page = 0

        # اعرض صفحة النتائج القياسية (مع زر 🧠 توليد رد ذكي)
        page_text = self.ui.render_search_page_text(state)
        kb = self.ui.build_search_page_keyboard(state)
        kb = self._with_smart_button(kb)

        if len(page_text) > 3500:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status.message_id)
            except Exception:
                pass
            ids = await self._send_long_text(chat_id, context, page_text, reply_markup=kb)
            state.result_message_ids.extend(ids)
            return

        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status.message_id,
                text=page_text,
                reply_markup=kb,
                parse_mode=C.ParseMode.HTML,
                disable_web_page_preview=True,
            )
            state.result_message_ids.append(status.message_id)
        except Exception:
            out = await context.bot.send_message(
                chat_id=chat_id,
                text=page_text,
                reply_markup=kb,
                parse_mode=C.ParseMode.HTML,
                disable_web_page_preview=True,
            )
            state.result_message_ids.append(out.message_id)

    # ---------- inline mode ----------
    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = self._strip_text(update.inline_query.query)
        if not q:
            await update.inline_query.answer([], cache_time=2, is_personal=True)
            return

        # ✅ whitelist (اختياري) — لو فعالة، امنع الاستخدام خارج القروبات المسموحة
        try:
            chat_id = update.effective_chat.id
            if not self._guard_chat_whitelist(chat_id):
                await update.inline_query.answer([], cache_time=2, is_personal=True)
                return
        except Exception:
            # inline_query قد لا يوفّر دائمًا chat واضح؛ نتجاهل الفحص في هذه الحالة
            pass

        flt = SearchFilters()
        top_k = min(self.cfg.top_k_default, 10)
        async with self._sem:
            results = await self.search_engine.search(query=q, top_k=top_k, flt=flt)

        articles = []
        from .utils import mask_sensitive, highlight_html
        for item in results[:10]:
            seed = item.get("seed", {}) or {}
            date_s = seed.get("date_str", "") or ""
            author = seed.get("user", "") or "مستخدم"
            link = self.ui._tg_link(seed)

            text_raw = self._strip_text(seed.get("message"))
            text_raw = mask_sensitive(text_raw)
            snippet_raw = (text_raw[:220] + "…") if len(text_raw) > 220 else text_raw
            snippet = highlight_html(snippet_raw, q)

            title = (text_raw[:64] + "…") if len(text_raw) > 64 else (text_raw or f"{author} — {date_s}")
            desc = f"{author} — {date_s}"
            body = (f"🧠 <i>{html.escape(q)}</i>\n"
                    f"🗓️ {html.escape(date_s)}\n"
                    f"👤 {html.escape(author)}\n"
                    f"{snippet}")
            if link:
                body += f'\n🔗 <a href="{html.escape(link)}">فتح الرسالة</a>'

            articles.append(
                InlineQueryResultArticle(
                    id=str(uuid4()),
                    title=title,
                    description=desc,
                    input_message_content=InputTextMessageContent(body, parse_mode=C.ParseMode.HTML)
                )
            )

        await update.inline_query.answer(articles, cache_time=2, is_personal=True)

    # ---------- callbacks في واجهة البحث المتقدم + الذكاء ----------
    async def buttons_in_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        chat_id = update.effective_chat.id
        state = self.state_mgr.get(chat_id)
        data = query.data
        logger.info("buttons_in_search HIT data=%s chat_id=%s", data, chat_id)

        def _try_edit_text(text: str, rm=None):
            async def _runner():
                try:
                    await query.edit_message_text(text, reply_markup=rm, parse_mode=C.ParseMode.HTML)
                except BadRequest as e:
                    logger.debug("edit_message_text skipped (not modified?): %s", e)
                except Exception as e:
                    logger.debug("edit_message_text failed: %s", e)
            return asyncio.create_task(_runner())

        def _try_edit_markup(markup):
            async def _runner():
                try:
                    await query.edit_message_reply_markup(markup)
                except BadRequest as e:
                    logger.debug("edit_message_reply_markup skipped (not modified?): %s", e)
                except Exception as e:
                    logger.debug("edit_message_reply_markup failed: %s", e)
            return asyncio.create_task(_runner())

        # ===== "🧠 توليد رد ذكي" =====
        if data == "gen_ai_reply":
            # ✅ لو OpenAI مفعّل: ولّد رد مباشر من آخر نتائج
            q_text = (state.query or getattr(state, "quick_query", "") or "").strip()
            if getattr(self.cfg, "use_openai_answer", False) and self.answerer and state.last_results:
                try:
                    await query.answer("⏳ جاري توليد الرد…", show_alert=False)
                except Exception:
                    pass

                await self._maybe_openai_answer(
                    question=q_text,
                    results=state.last_results,
                    chat_id=chat_id,
                    context=context,
                    status_message_id=None,
                    back_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 رجوع", callback_data="back_to_home")]]),
                )
                return self.ui.PAGE_SEARCH

            # (Fallback) النظام القديم: إرسال طلب للأدمن
            req_id = str(uuid4())
            user_id = update.effective_user.id
            user_name = (update.effective_user.full_name or "").strip()
            user_username = (update.effective_user.username or "") and f"@{update.effective_user.username}" or ""
            origin_chat_title = getattr(update.effective_chat, "title", "") or "خاص/مجموعة"
            q_text = (state.query or getattr(state, "quick_query", "") or "").strip()

            self.smart_requests[req_id] = {
                "req_id": req_id,
                "ts": datetime.utcnow().timestamp(),
                "ts_h": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                "origin_chat_id": chat_id,
                "origin_chat_title": origin_chat_title,
                "student_user_id": user_id,
                "student_name": user_name,
                "student_username": user_username,
                "query_text": q_text,
            }

            # أبلغ الطالب
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="🤖 تم استدعاء الذكاء الاصطناعي للرد على استفسارك من محادثات القروب… انتظر قليلاً وسيصلك الرد.",
                    parse_mode=C.ParseMode.HTML,
                )
            except Exception:
                pass

            # أبلغ كل الأدمن
            admins = list(self.cfg.admin_ids or [])
            if not admins:
                await query.answer("لا يوجد مشرفون مسجلون.", show_alert=True)
                return self.ui.PAGE_SEARCH

            notif = (
                "📥 <b>طلب رد ذكي جديد</b>\n"
                f"• المجموعة/القروب: {html.escape(origin_chat_title)} (id={chat_id})\n"
                f"• الطالب: {html.escape(user_name)} {html.escape(user_username)} (id={user_id})\n"
                f"• الوقت: {html.escape(self.smart_requests[req_id]['ts_h'])}\n"
                f"• الاستعلام:\n<code>{html.escape(q_text or '—')}</code>\n"
            )
            admin_kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✍️ رد الآن", callback_data=f"smart:reply:{req_id}"),
                    InlineKeyboardButton("🗑️ تجاهل", callback_data=f"smart:dismiss:{req_id}"),
                ],
                [InlineKeyboardButton("🛠️ لوحة الأدمن", callback_data="admin:dashboard")]
            ])
            for aid in admins:
                try:
                    await context.bot.send_message(
                        chat_id=aid, text=notif, reply_markup=admin_kb, parse_mode=C.ParseMode.HTML
                    )
                except Exception as e:
                    logger.debug("notify admin %s failed: %s", aid, e)

            await query.answer("تم إرسال الطلب للمشرفين.", show_alert=False)
            return self.ui.PAGE_SEARCH

        # ===== معالجة أزرار طلبات الذكاء عند الأدمن =====
        if data.startswith("smart:"):
            parts = data.split(":")
            if len(parts) >= 3:
                action = parts[1]
                req_id = parts[2]
            else:
                return self.ui.PAGE_SEARCH

            req = self.smart_requests.get(req_id)
            if not req:
                await query.answer("هذا الطلب غير موجود أو تم التعامل معه.", show_alert=True)
                return self.ui.PAGE_SEARCH

            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH

            if action == "dismiss":
                self.smart_requests.pop(req_id, None)
                await query.answer("تم تجاهل الطلب.", show_alert=False)
                try:
                    await query.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                return self.ui.PAGE_SEARCH

            if action == "reply":
                admin_state = self.state_mgr.get(update.effective_chat.id)
                admin_state.expecting_admin_smart_reply = True
                admin_state.pending_smart_req_id = req_id
                await query.answer("أرسل الرد الآن (أول رسالة سترسل إلى الطالب).", show_alert=True)
                try:
                    await query.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=(
                        "✍️ اكتب الرد الذكي الآن.\n"
                        f"• المجموعة: {html.escape(req.get('origin_chat_title',''))}\n"
                        f"• استفسار الطالب:\n<code>{html.escape(req.get('query_text') or '—')}</code>"
                    ),
                    parse_mode=C.ParseMode.HTML
                )
                return self.ui.PAGE_SEARCH

        # ===== لوحة الأدمن =====
        if data == "admin:dashboard":
            uid = update.effective_user.id
            if not (self.cfg.admin_ids and uid in self.cfg.admin_ids):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH

            a = getattr(self.state_mgr, "admin_stats", None)
            users_count = len(a.unique_users) if a else 0
            searches_count = sum(len(v) for v in getattr(a, "user_searches", {}).values()) if a else 0

            recent_users = []
            if a and a.user_profile:
                by_seen = sorted(a.user_profile.items(), key=lambda kv: kv[1].get("last_seen", 0), reverse=True)[:10]
                for uid_i, prof in by_seen:
                    recent_users.append({
                        "id": uid_i,
                        "name": prof.get("name") or "—",
                        "username": prof.get("username") or "",
                    })

            recent_queries = []
            if a and a.user_searches:
                for uid_i, items in a.user_searches.items():
                    for it in items[-20:]:
                        recent_queries.append({
                            "ts": it.get("t", 0.0),
                            "ts_h": datetime.fromtimestamp(it.get("t", 0.0)).strftime("%Y-%m-%d %H:%M"),
                            "chat": uid_i,
                            "mode": it.get("mode", ""),
                            "q": it.get("q", ""),
                        })
                recent_queries.sort(key=lambda r: r["ts"], reverse=True)
                recent_queries = recent_queries[:10]

            bot_name = getattr(self.cfg, "bot_name", "") or "Search Bot"
            uni_name = getattr(self.cfg, "university_name", "") or "-"
            multibot_id = getattr(self.cfg, "multibot_id", "") or "-"

            smart_n = len(self.smart_requests)

            lines = []
            lines.append("🛠️ <b>لوحة الأدمن</b>")
            lines.append("")
            lines.append(f"• Bot: {html.escape(bot_name)}")
            lines.append(f"• University: {html.escape(uni_name)}")
            lines.append(f"• MULTIBOT_ID: {html.escape(str(multibot_id))}")
            lines.append("")
            lines.append(f"👥 المستخدمون الفريدون: {users_count}")
            lines.append(f"🔎 عدد عمليات البحث: {searches_count}")
            lines.append(f"🤖 طلبات الرد الذكي المعلّقة: {smart_n}")
            lines.append("")

            if recent_users:
                lines.append("<b>آخر مستخدمين شوهدوا:</b>")
                for u in recent_users:
                    uid_s = str(u.get("id", ""))
                    nm = u.get("name") or "—"
                    un = u.get("username") or ""
                    lines.append(f"• {html.escape(nm)} {html.escape(un)} (id={html.escape(uid_s)})")
                lines.append("")

            if recent_queries:
                lines.append("<b>آخر عمليات البحث:</b>")
                for r in recent_queries:
                    ts_h = r.get("ts_h", "")
                    ch = str(r.get("chat", ""))
                    mode = r.get("mode", "")
                    q2 = r.get("q", "") or ""
                    q_disp = (q2[:70] + "…") if len(q2) > 70 else q2
                    lines.append(f"• [{html.escape(ts_h)}] chat={html.escape(ch)} mode={html.escape(mode)} — {html.escape(q_disp)}")

            txt = "\n".join(lines) or "لا توجد بيانات بعد."

            adm_kb_rows = [
                [InlineKeyboardButton(f"📬 طلبات الرد الذكي ({smart_n})", callback_data="admin:smart_list")],
                [InlineKeyboardButton("العودة للرئيسية", callback_data="back_to_home")],
            ]
            adm_kb = InlineKeyboardMarkup(adm_kb_rows)

            try:
                await query.edit_message_text(
                    txt,
                    parse_mode=C.ParseMode.HTML,
                    reply_markup=adm_kb
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=txt,
                    parse_mode=C.ParseMode.HTML,
                    reply_markup=adm_kb
                )
            return self.ui.PAGE_HOME

        if data == "admin:smart_list":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH

            if not self.smart_requests:
                await query.edit_message_text(
                    "لا توجد طلبات رد ذكي معلّقة حالياً.",
                    parse_mode=C.ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("رجوع", callback_data="admin:dashboard")]])
                )
                return self.ui.PAGE_SEARCH

            reqs = sorted(self.smart_requests.values(), key=lambda r: r.get("ts", 0), reverse=True)[:10]
            lines = ["📬 <b>طلبات الرد الذكي</b>", ""]
            kb_rows: List[List[InlineKeyboardButton]] = []
            for r in reqs:
                rid = r["req_id"]
                title = f"{r.get('origin_chat_title','')} — {r.get('student_name','')}"
                lines.append(f"• [{r.get('ts_h','')}] {html.escape(title)}")
                lines.append(f"  س: <code>{html.escape(r.get('query_text') or '—')}</code>")
                kb_rows.append([
                    InlineKeyboardButton(f"✍️ رد ({rid[:8]})", callback_data=f"smart:reply:{rid}"),
                    InlineKeyboardButton("🗑️ تجاهل", callback_data=f"smart:dismiss:{rid}")
                ])
            kb_rows.append([InlineKeyboardButton("🔙 رجوع", callback_data="admin:dashboard")])
            await query.edit_message_text(
                "\n".join(lines),
                parse_mode=C.ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(kb_rows)
            )
            return self.ui.PAGE_SEARCH

        # تنقّل للرئيسية / مزج أوضاع
        if data == "back_to_home":
            self._reset_expect_flags(state)
            is_admin = self._is_admin(update.effective_user.id)
            _try_edit_text(self.ui.render_home_text(), rm=self._build_home_menu(is_admin))
            return self.ui.PAGE_HOME
        if data == "quick:new":
            self._reset_expect_flags(state)
            state.expecting_quick_query = True
            _try_edit_text("📝 اكتب عبارة البحث الجديدة:", rm=self._quick_prompt_kb(self._is_admin(update.effective_user.id)))
            return self.ui.PAGE_HOME
        if data == "quick:to_advanced":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            _try_edit_text("جاهز! اضبط الفلاتر واضغط «ابدأ البحث».", rm=self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        # ===== (أزرار الفلاتر للأدمن فقط) =====
        if data == "toggle_only_replies":
            state.only_replies = not state.only_replies
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "toggle_only_contact":
            state.only_with_contact = not getattr(state, "only_with_contact", False)
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "qf:last7":
            today = datetime.utcnow().date()
            start = today - timedelta(days=6)
            end = today
            state.date = None
            state.date_range = ((start.year, start.month, start.day), (end.year, end.month, end.day))
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "qf:last30":
            today = datetime.utcnow().date()
            start = today - timedelta(days=29)
            end = today
            state.date = None
            state.date_range = ((start.year, start.month, start.day), (end.year, end.month, end.day))
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "qf:last365":
            today = datetime.utcnow().date()
            start = today - timedelta(days=364)
            end = today
            state.date = None
            state.date_range = ((start.year, start.month, start.day), (end.year, end.month, end.day))
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "qf:all":
            state.date = None
            state.date_range = None
            await query.answer("تم إلغاء فلتر التاريخ ✅")
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "disable_date":
            state.date = None
            await query.answer("تم تعطيل التاريخ ✅")
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "disable_daterange":
            state.date_range = None
            await query.answer("تم تعطيل النطاق ✅")
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        # ===== التقويم =====
        if data == "ask_date":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            state.expecting_date = True
            today = datetime.utcnow()
            await query.edit_message_text(
                "📅 اختر اليوم من التقويم:",
                reply_markup=self.ui.build_calendar(today.year, today.month),
                parse_mode=C.ParseMode.HTML,
            )
            return self.DATE_PICKER

        if data == "ask_date_range":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            state.expecting_date_range = True
            today = datetime.utcnow()
            await query.edit_message_text(
                "🗓️ اختر تاريخ البداية للبحث باستخدام التقويم:",
                reply_markup=self.ui.build_calendar(today.year, today.month),
                parse_mode=C.ParseMode.HTML,
            )
            return self.DATE_PICKER_RANGE_START

        if data == "calendar_back":
            self._reset_expect_flags(state)
            await query.edit_message_text(
                "تم الرجوع دون اختيار تاريخ.",
                reply_markup=self.ui.build_main_menu(state),
                parse_mode=C.ParseMode.HTML,
            )
            return self.ui.PAGE_SEARCH

        if data.startswith("prevmonth:") or data.startswith("nextmonth:"):
            try:
                year, month = map(int, data.split(":")[1].split("-"))
                await query.edit_message_reply_markup(self.ui.build_calendar(year, month))
            except Exception as e:
                logger.debug("calendar month nav failed: %s", e)
            return self.ui.PAGE_SEARCH

        if data.startswith("setdate:"):
            try:
                date_str = data.split(":")[1]
                selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception as e:
                logger.debug("setdate parse failed: %s", e)
                return self.ui.PAGE_SEARCH

            if state.expecting_date:
                state.date = (selected_date.year, selected_date.month, selected_date.day)
                state.date_range = None
                self._reset_expect_flags(state)
                await query.edit_message_text(
                    f"✅ تم ضبط التاريخ على: {selected_date.strftime('%d/%m/%Y')}",
                    reply_markup=self.ui.build_main_menu(state),
                    parse_mode=C.ParseMode.HTML,
                )
                return self.ui.PAGE_SEARCH

            if state.expecting_date_range:
                if not getattr(state, "date_range_start", None):
                    state.date_range_start = selected_date
                    await query.edit_message_text(
                        f"🗓️ اختر تاريخ النهاية (بعد {selected_date.strftime('%d/%m/%Y')}) باستخدام التقويم:",
                        reply_markup=self.ui.build_calendar(selected_date.year, selected_date.month),
                        parse_mode=C.ParseMode.HTML,
                    )
                    return self.DATE_PICKER_RANGE_END
                else:
                    start = state.date_range_start
                    end = selected_date
                    if end < start:
                        start, end = end, start
                    state.date_range = (
                        (start.year, start.month, start.day),
                        (end.year, end.month, end.day),
                    )
                    self._reset_expect_flags(state)
                    state.date = None
                    await query.edit_message_text(
                        f"✅ تم اختيار الفترة: {start.strftime('%d/%m/%Y')} → {end.strftime('%d/%m/%Y')}",
                        reply_markup=self.ui.build_main_menu(state),
                        parse_mode=C.ParseMode.HTML,
                    )
                    return self.ui.PAGE_SEARCH

            return self.ui.PAGE_SEARCH

        # ===== بقية الأزرار =====
        if data == "ask_nprobe":
            # أدمن فقط
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH

            state.expecting_nprobe = True
            cur, mx = self._faiss_probe_limits()
            cur_s = str(cur) if cur is not None else "غير مضبوط"
            mx_s = f" / الأقصى: {mx}" if mx is not None else ""

            txt = (
                f"🧮 اكتب قيمة nprobe (عدد موجب) (الحالي: {cur_s}{mx_s})\n"
                "ملاحظة: كل ما زاد nprobe زادت الدقة… وأيضًا يزيد وقت البحث."
            )
            try:
                await query.edit_message_text(
                    txt,
                    reply_markup=self.ui.build_main_menu(state),
                    parse_mode=C.ParseMode.HTML
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=txt,
                    reply_markup=self.ui.build_main_menu(state),
                    parse_mode=C.ParseMode.HTML
                )
            return self.ui.PAGE_SEARCH

        if data == "ask_topk":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            state.expecting_topk = True
            _try_edit_text(
                f"🔢 اكتب عدد النتائج TopK (حالياً: {state.top_k}). مثال: 100",
                rm=self.ui.build_main_menu(state),
            )
            return self.ui.PAGE_SEARCH

        if data == "ask_pagesize":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            state.expecting_pagesize = True
            _try_edit_text(
                f"📄 اكتب حجم الصفحة (من {self.cfg.page_size_min} إلى {self.cfg.page_size_max})، الحالي: {state.page_size}.",
                rm=self.ui.build_main_menu(state),
            )
            return self.ui.PAGE_SEARCH

        if data == "ask_keyword":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            state.expecting_keyword = True
            _try_edit_text(
                "🔑 اكتب الكلمة المهمة للفلترة داخل نص الرسالة.\n"
                "• لفصل أكثر من كلمة: استخدم | مثل: فيزياء|مواد\n"
                "• للاستبعاد: -كلمة مثل: -إعلانات",
                rm=self.ui.build_main_menu(state),
            )
            return self.ui.PAGE_SEARCH

        if data == "start_search":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_HOME
            self._reset_expect_flags(state)
            state.expecting_query = True
            txt = (
                "✍️ اكتب استعلام البحث (جملة قصيرة تعبّر عن اللي تدور عليه).\n"
                "مثال: <code>احد يعرف الدكتور الي يدرس مادة الرسم الهندسي ايش اسمه؟</code>"
            )
            _try_edit_text(txt, rm=self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "save_query":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            if state.query:
                state.saved_query = {
                    "query": state.query,
                    "only_replies": state.only_replies,
                    "only_with_contact": getattr(state, "only_with_contact", False),
                    "date": state.date,
                    "date_range": state.date_range,
                    "keyword": state.keyword,
                    "top_k": state.top_k,
                    "page_size": state.page_size,
                }
                _try_edit_text("💾 تم حفظ الاستعلام والإعدادات الحالية.", rm=self.ui.build_main_menu(state))
            else:
                _try_edit_text("ℹ️ لا يوجد استعلام حالي لحفظه.", rm=self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "rerun_saved":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            saved = state.saved_query
            if not saved:
                _try_edit_text("❌ لا يوجد بحث محفوظ.", rm=self.ui.build_main_menu(state))
                return self.ui.PAGE_SEARCH
            state.query = saved["query"]
            state.only_replies = saved["only_replies"]
            state.only_with_contact = saved.get("only_with_contact", False)
            state.date = saved["date"]
            state.date_range = saved["date_range"]
            state.keyword = saved["keyword"]
            state.top_k = saved["top_k"]
            state.page_size = min(max(saved["page_size"], self.cfg.page_size_min), self.cfg.page_size_max)
            await self._run_search_and_show(query, context, chat_id, state, from_message=False, keep_page=0)
            return self.ui.PAGE_SEARCH

        if data == "new_search":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            self._reset_expect_flags(state)
            state.expecting_query = True
            state.query = ""
            await query.answer("اكتب كلمات البحث الجديدة…")
            _try_edit_text("📝 اكتب كلمات البحث الجديدة:", rm=self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "pin_filters":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            state.pinned = {
                "only_replies": state.only_replies,
                "only_with_contact": getattr(state, "only_with_contact", False),
                "date": state.date,
                "date_range": state.date_range,
                "keyword": state.keyword,
                "top_k": state.top_k,
                "page_size": state.page_size,
            }
            await query.answer("تم تثبيت الفلاتر كافتراضية ✅")
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "apply_pinned":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            if getattr(state, "pinned", None):
                pf = state.pinned
                state.only_replies = pf.get("only_replies", state.only_replies)
                state.only_with_contact = pf.get("only_with_contact", getattr(state, "only_with_contact", False))
                state.date = pf.get("date")
                state.date_range = pf.get("date_range")
                state.keyword = pf.get("keyword")
                state.top_k = pf.get("top_k", state.top_k)
                state.page_size = pf.get("page_size", state.page_size)
                await query.answer("تم تطبيق الافتراضيات 📥")
            else:
                await query.answer("لا يوجد فلاتر مثبتة بعد")
            _try_edit_markup(self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        if data == "refresh_page":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            if not state.query:
                await query.answer("اكتب استعلام أولًا")
                return self.ui.PAGE_SEARCH
            prev_page = state.current_page
            await self._run_search_and_show(query, context, chat_id, state, from_message=False, keep_page=prev_page)
            return self.ui.PAGE_SEARCH

        if data == "reset_filters":
            if not self._is_admin(update.effective_user.id):
                await query.answer("هذا الخيار للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            new_state = self.state_mgr.reset(chat_id)
            self._reset_expect_flags(new_state)
            _try_edit_text("♻️ تم مسح الفلاتر وإعادة الضبط.", rm=self.ui.build_main_menu(new_state))
            return self.ui.PAGE_SEARCH

        if data == "back_to_menu":
            _try_edit_text("🏠 رجعناك للقائمة الرئيسية (وضع الفلاتر).", rm=self.ui.build_main_menu(state))
            return self.ui.PAGE_SEARCH

        # تنقّل بين الصفحات بالسهمين
        if data in ("nav:prev", "nav:next"):
            if data == "nav:prev" and state.current_page > 0:
                state.current_page -= 1
            if data == "nav:next" and (state.current_page + 1) < state.total_pages:
                state.current_page += 1
            text = self.ui.render_search_page_text(state)
            kb = self.ui.build_search_page_keyboard(state)
            kb = self._with_smart_button(kb)

            if len(text) > 3500:
                try:
                    await query.message.delete()
                except Exception:
                    pass
                ids = await self._send_long_text(chat_id, context, text, reply_markup=kb)
                state.result_message_ids.extend(ids)
                return self.ui.PAGE_SEARCH

            try:
                await query.edit_message_text(text, reply_markup=kb, parse_mode=C.ParseMode.HTML)
            except Exception:
                out = await context.bot.send_message(
                    chat_id=chat_id, text=text, reply_markup=kb, parse_mode=C.ParseMode.HTML
                )
                state.result_message_ids.append(out.message_id)
            return self.ui.PAGE_SEARCH

        # زر مباشر للانتقال إلى صفحة معينة page:N
        if data.startswith("page:"):
            try:
                page = int(data.split(":")[1])
            except Exception:
                page = state.current_page
            if state.total_pages:
                state.current_page = min(max(page, 0), max(state.total_pages - 1, 0))
            text = self.ui.render_search_page_text(state)
            kb = self.ui.build_search_page_keyboard(state)
            kb = self._with_smart_button(kb)

            if len(text) > 3500:
                try:
                    await query.message.delete()
                except Exception:
                    pass
                ids = await self._send_long_text(chat_id, context, text, reply_markup=kb)
                state.result_message_ids.extend(ids)
                return self.ui.PAGE_SEARCH

            try:
                await query.edit_message_text(text, reply_markup=kb, parse_mode=C.ParseMode.HTML)
            except Exception:
                out = await context.bot.send_message(
                    chat_id=chat_id, text=text, reply_markup=kb, parse_mode=C.ParseMode.HTML
                )
                state.result_message_ids.append(out.message_id)
            return self.ui.PAGE_SEARCH

        if data == "export_json":
            if not self._is_admin(update.effective_user.id):
                await query.answer("للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            await self._export_results(state, query, context, chat_id, fmt="json")
            return self.ui.PAGE_SEARCH

        if data == "export_csv":
            if not self._is_admin(update.effective_user.id):
                await query.answer("للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            await self._export_results(state, query, context, chat_id, fmt="csv")
            return self.ui.PAGE_SEARCH

        if data == "export_html":
            if not self._is_admin(update.effective_user.id):
                await query.answer("للمشرفين فقط", show_alert=True)
                return self.ui.PAGE_SEARCH
            await self._export_results(state, query, context, chat_id, fmt="html")
            return self.ui.PAGE_SEARCH

        if data == "noop":
            return self.ui.PAGE_SEARCH

        # ===== show:idx — صفحة الردود =====
        if data.startswith("show:"):
            try:
                idx = int(data.split(":")[1])
            except Exception:
                return self.ui.PAGE_SEARCH

            if idx < 0 or idx >= len(state.last_results):
                _try_edit_text("❌ هذا العنصر غير متاح.", rm=self.ui.build_search_page_keyboard(state))
                return self.ui.PAGE_SEARCH

            item = state.last_results[idx]
            try:
                if not item.get("replies"):
                    seed = item.get("seed", {}) or {}
                    sid = seed.get("id", "")
                    q_vec = await self._encode_query_safely(state.query or "")
                    replies = await self.search_engine._smart_replies(
                        q_vec=q_vec,
                        seed_id=sid,
                        max_depth=max(self.cfg.max_depth, 3),
                        max_replies=self.cfg.max_replies,
                        keyword=state.keyword,
                    )
                    item["replies"] = replies
                    if replies:
                        item["best_reply"] = replies[0]
                    state.last_results[idx] = item
            except Exception as e:
                logger.warning("failed to recompute replies for %s: %s", item.get('seed', {}).get('id', '?'), e)

            text = self.ui.render_replies_page_text(item, state.query)

            try:
                await query.message.delete()
            except Exception:
                pass

            if len(text) > 3500:
                ids = await self._send_long_text(
                    chat_id,
                    context,
                    text,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🔙 رجوع لنتائج البحث", callback_data="back_to_search_same_page")]]
                    ),
                )
                state.reply_page_message_ids.extend(ids)
            else:
                out = await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=InlineKeyboardMarkup(
                        [[InlineKeyboardButton("🔙 رجوع لنتائج البحث", callback_data="back_to_search_same_page")]]
                    ),
                    parse_mode=C.ParseMode.HTML,
                )
                state.reply_page_message_ids.append(out.message_id)

            state.last_page_before_replies = state.current_page
            return self.ui.PAGE_REPLIES

        if data == "back_to_search_same_page":
            try:
                await query.message.delete()
            except Exception:
                pass
            return await self.redraw_search_page(chat_id, context)

        logger.info("buttons_in_search: unhandled callback data=%s", data)
        return self.ui.PAGE_SEARCH

    #--------------------------------------------------------------------------------------------------------------
    async def redraw_search_page(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
        state = self.state_mgr.get(chat_id)
        text = self.ui.render_search_page_text(state)
        reply_markup = self.ui.build_search_page_keyboard(state)
        reply_markup = self._with_smart_button(reply_markup)

        if len(text) > 3500:
            ids = await self._send_long_text(chat_id, context, text, reply_markup=reply_markup)
            state.result_message_ids.extend(ids)
        else:
            out = await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=C.ParseMode.HTML,
            )
            state.result_message_ids.append(out.message_id)
        return self.ui.PAGE_SEARCH

    # ---------- export helpers ----------
    async def _export_results(self, state: ChatState, query_obj, context, chat_id: int, fmt: str = "json"):
        """
        ✅ تعديل: التصدير يشمل الردود نفسها تحت كل استفسار (ليس فقط العدد)
        - يحسب replies تلقائيًا لو غير موجودة وقت التصدير (باستخدام _smart_replies)
        - JSON: replies قائمة objects
        - CSV: replies تُخزّن كسلسلة JSON
        - HTML: يعرض الردود داخل الجدول (اختياري)
        """
        if not state.last_results:
            try:
                await query_obj.edit_message_text(
                    "❌ لا توجد نتائج لتصديرها.",
                    reply_markup=self.ui.build_search_page_keyboard(state),
                    parse_mode=C.ParseMode.HTML,
                )
            except Exception:
                pass
            return

        include_replies = bool(getattr(self.cfg, "export_include_replies", True))
        max_replies = int(getattr(self.cfg, "export_max_replies", 20))

        # حضّر embedding للاستعلام مرة واحدة (لـ smart replies)
        q_text = (state.query or "").strip()
        q_vec = None
        if include_replies and q_text:
            q_vec = await self._encode_query_safely(q_text)

        rows: List[Dict[str, Any]] = []
        for item in state.last_results:
            seed = item.get("seed", {}) or {}
            sid = seed.get("id", "")
            link = self.ui._tg_link(seed)

            # ✅ احسب replies لو مش موجودة وعايز تضمّنها في التصدير
            if include_replies and sid and not item.get("replies"):
                try:
                    replies = await self.search_engine._smart_replies(
                        q_vec=q_vec,
                        seed_id=sid,
                        max_depth=max(getattr(self.cfg, "max_depth", 3), 3),
                        max_replies=max_replies,
                        keyword=state.keyword,
                    )
                    item["replies"] = replies
                    if replies:
                        item["best_reply"] = replies[0]
                except Exception as e:
                    logger.debug("export: compute replies failed for %s: %s", sid, e)
                    item["replies"] = []

            replies = item.get("replies", []) or []

            # صياغة replies للتصدير
            exp_replies = []
            if include_replies:
                for depth, r in replies[:max_replies]:
                    exp_replies.append({
                        "depth": depth,
                        "id": r.get("id"),
                        "date_str": r.get("date_str"),
                        "user": r.get("user"),
                        "username": r.get("username"),
                        "message": r.get("message"),
                        "link": self.ui._tg_link(r) or "",
                    })

            base_row = {
                "id": seed.get("id"),
                "date_str": seed.get("date_str"),
                "user": seed.get("user"),
                "username": seed.get("username"),
                "message": seed.get("message"),
                "year": seed.get("year"),
                "month": seed.get("month"),
                "day": seed.get("day"),
                "link": link or "",
                "replies_count": len(replies),
            }

            if include_replies:
                base_row["replies"] = exp_replies
                base_row["best_reply"] = (exp_replies[0] if exp_replies else None)

            rows.append(base_row)

        ts = self._now_str()

        if fmt == "json":
            buf = io.BytesIO(json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"))
            fname = f"search_results_{ts}.json"
            await context.bot.send_document(chat_id=chat_id, document=InputFile(buf, filename=fname), caption="📤 تصدير JSON")

        elif fmt == "csv":
            import io as _io
            sio = _io.StringIO()

            # CSV لا يدعم nested list جيدًا → نخلي replies كـ JSON string
            def _csv_val(v):
                if isinstance(v, (dict, list)):
                    return json.dumps(v, ensure_ascii=False)
                if v is None:
                    return ""
                return str(v).replace("\n", " ").strip()

            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(sio, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: _csv_val(r.get(k)) for k in fieldnames})

            buf = io.BytesIO(sio.getvalue().encode("utf-8"))
            fname = f"search_results_{ts}.csv"
            await context.bot.send_document(chat_id=chat_id, document=InputFile(buf, filename=fname), caption="📤 تصدير CSV")

        else:  # html
            html_rows = [
                "<table border=1 cellpadding=6 cellspacing=0>",
                "<tr><th>date</th><th>user</th><th>text</th><th>link</th><th>replies</th></tr>"
            ]

            for r in rows:
                link_cell = f'<a href="{html.escape(r["link"])}">open</a>' if r.get("link") else ""
                text_cell = html.escape((r.get("message") or "")[:300])
                user_cell = html.escape((r.get("user") or ""))
                date_cell = html.escape((r.get("date_str") or ""))

                rep_cell = ""
                if include_replies:
                    reps = r.get("replies") or []
                    parts = []
                    for rep in reps[:max_replies]:
                        rep_msg = html.escape((rep.get("message") or "")[:220])
                        rep_user = html.escape((rep.get("user") or ""))
                        rep_date = html.escape((rep.get("date_str") or ""))
                        rep_link = rep.get("link") or ""
                        rep_link_html = f' <a href="{html.escape(rep_link)}">open</a>' if rep_link else ""
                        parts.append(f"<div>↳ <b>{rep_user}</b> ({rep_date}): {rep_msg}{rep_link_html}</div>")
                    rep_cell = "".join(parts) if parts else ""
                else:
                    rep_cell = html.escape(str(r.get("replies_count", 0)))

                html_rows.append(
                    f"<tr><td>{date_cell}</td><td>{user_cell}</td><td>{text_cell}</td><td>{link_cell}</td><td>{rep_cell}</td></tr>"
                )

            content = "\n".join(html_rows + ["</table>"])
            buf = io.BytesIO(content.encode("utf-8"))
            fname = f"search_results_{ts}.html"
            await context.bot.send_document(
                chat_id=chat_id,
                document=InputFile(buf, filename=fname),
                caption="📤 تصدير HTML"
            )

        try:
            await query_obj.edit_message_reply_markup(self.ui.build_search_page_keyboard(state))
        except Exception:
            pass

    # ---------- الرسائل النصية ----------
    async def text_in_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id

        # ✅ whitelist (اختياري)
        if not self._guard_chat_whitelist(chat_id):
            await update.message.reply_text("❌ هذا البوت غير مفعّل لهذا القروب.")
            return ConversationHandler.END

        state = self.state_mgr.get(chat_id)

        # ✅ قصّ مدخلات المستخدم
        raw_text = update.message.text or ""
        max_len = getattr(self.cfg, "max_user_text_len", 2000)
        if len(raw_text) > max_len:
            raw_text = raw_text[:max_len]
        text = self._strip_text(raw_text)

        # ⛔️ منع تمرير نفس الرسالة مرة ثانية كاستعلام
        if getattr(state, "suppress_next_text", False):
            state.suppress_next_text = False
            return self.ui.PAGE_SEARCH

        logger.info(
            "text_in_search chat_id=%s text=%r flags: quick=%s q=%s, date=%s, range=%s, topk=%s, pagesize=%s, keyword=%s, admin_smart=%s",
            chat_id,
            text,
            getattr(state, "expecting_quick_query", False),
            state.expecting_query,
            state.expecting_date,
            state.expecting_date_range,
            state.expecting_topk,
            state.expecting_pagesize,
            state.expecting_keyword,
            getattr(state, "expecting_admin_smart_reply", False),
        )

        # ✍️ لو الإدمن في وضع "رد ذكي": أول رسالة تُرسل للطالب (بدون "(من المشرف)" وبدون توقيع)
        if self._is_admin(update.effective_user.id) and getattr(state, "expecting_admin_smart_reply", False):
            state.expecting_admin_smart_reply = False
            req_id = getattr(state, "pending_smart_req_id", None)
            state.pending_smart_req_id = None

            req = self.smart_requests.pop(req_id, None) if req_id else None
            if not req:
                await update.message.reply_text("⚠️ الطلب غير موجود/أُغلق.")
                return self.ui.PAGE_SEARCH

            student_chat_id = req.get("origin_chat_id")
            q_text = req.get("query_text") or ""

            try:
                await context.bot.send_message(
                    chat_id=student_chat_id,
                    text=(
                        "🧠 <b>رد الذكاء الاصطناعي</b>\n"
                        f"• الاستفسار:\n<code>{html.escape(q_text)}</code>\n\n"
                        f"• الرد:\n{html.escape(text)}"
                    ),
                    parse_mode=C.ParseMode.HTML
                )
            except Exception as e:
                logger.debug("send smart reply to student failed: %s", e)
                await update.message.reply_text("❌ تعذّر إرسال الرد للطالب.")
                return self.ui.PAGE_SEARCH

            await update.message.reply_text("✅ تم إرسال الرد للطالب.")
            return self.ui.PAGE_SEARCH

        if not self.state_mgr.check_rate_limit(chat_id):
            await update.message.reply_text("⚠️ تجاوزت حد الطلبات مؤقتًا. جرب بعد شوي.")
            return self.ui.PAGE_SEARCH

        from .utils import parse_user_date, parse_date_range

        # البحث السريع
        if getattr(state, "expecting_quick_query", False):
            state.expecting_quick_query = False
            state.quick_query = text
            await self._run_quick_search_and_show(chat_id, context, state)
            return self.ui.PAGE_HOME

        # اختيار التاريخ/النطاق من التقويم
        if state.expecting_date:
            state.expecting_date = False
            if text.lower() == "تعطيل":
                state.date = None
                msg = await update.message.reply_text("📅 تم تعطيل فلتر التاريخ (يوم واحد).", reply_markup=self.ui.build_main_menu(state))
                state.result_message_ids.append(msg.message_id)
                return self.ui.PAGE_SEARCH
            parsed = parse_user_date(text)
            if not parsed:
                msg = await update.message.reply_text("❌ صيغة التاريخ غير صحيحة. استخدم YYYY-MM-DD أو DD/MM/YYYY.", reply_markup=self.ui.build_main_menu(state))
                state.result_message_ids.append(msg.message_id)
                return self.ui.PAGE_SEARCH
            state.date = parsed
            state.date_range = None
            msg = await update.message.reply_text(f"✅ تم ضبط التاريخ على: {parsed[2]:02d}/{parsed[1]:02d}/{parsed[0]}", reply_markup=self.ui.build_main_menu(state))
            state.result_message_ids.append(msg.message_id)
            return self.ui.PAGE_SEARCH

        if state.expecting_date_range:
            state.expecting_date_range = False
            if text.lower() == "تعطيل":
                state.date_range = None
                state.date_range_start = None
                state.date_range_end = None
                msg = await update.message.reply_text("🗓️ تم تعطيل فلتر نطاق التاريخ.", reply_markup=self.ui.build_main_menu(state))
                state.result_message_ids.append(msg.message_id)
                return self.ui.PAGE_SEARCH
            parsed = parse_date_range(text)
            if not parsed:
                msg = await update.message.reply_text("❌ صيغة نطاق التاريخ غير صحيحة. مثال: 2024-01-01..2024-12-31", reply_markup=self.ui.build_main_menu(state))
                state.result_message_ids.append(msg.message_id)
                return self.ui.PAGE_SEARCH
            state.date_range = parsed
            (y1, m1, d1), (y2, m2, d2) = parsed
            state.date_range_start = None
            state.date_range_end = None
            state.date = None
            msg = await update.message.reply_text(f"✅ تم ضبط النطاق: {d1:02d}/{m1:02d}/{y1} → {d2:02d}/{m2:02d}/{y2}", reply_markup=self.ui.build_main_menu(state))
            state.result_message_ids.append(msg.message_id)
            return self.ui.PAGE_SEARCH

        if getattr(state, "expecting_nprobe", False):
            state.expecting_nprobe = False
            raw = (text or "").strip()
            try:
                val = int(raw)
                if val <= 0:
                    raise ValueError()

                try:
                    import faiss  # type: ignore
                    idx = getattr(self.search_engine.index, "index", None)
                    if isinstance(idx, faiss.IndexIVF):
                        nlist = int(idx.nlist)
                        applied = min(val, nlist)
                        idx.nprobe = int(applied)
                        try:
                            self.cfg.nprobe = int(idx.nprobe)
                        except Exception:
                            pass
                        msg = await update.message.reply_text(
                            f"✅ تم ضبط nprobe على: {applied} (الأقصى: {nlist})",
                            reply_markup=self.ui.build_main_menu(state),
                        )
                        state.result_message_ids.append(msg.message_id)
                    else:
                        msg = await update.message.reply_text(
                            "ℹ️ الفهرس الحالي ليس IVF؛ إعداد nprobe غير متاح (FLAT/HNSW).",
                            reply_markup=self.ui.build_main_menu(state),
                        )
                        state.result_message_ids.append(msg.message_id)
                except Exception:
                    msg = await update.message.reply_text(
                        "⚠️ تعذّر الوصول إلى الفهرس لضبط nprobe.",
                        reply_markup=self.ui.build_main_menu(state),
                    )
                    state.result_message_ids.append(msg.message_id)
            except Exception:
                msg = await update.message.reply_text(
                    "❌ قيمة غير صالحة. اكتب عددًا صحيحًا موجبًا لـ nprobe.",
                    reply_markup=self.ui.build_main_menu(state),
                )
                state.result_message_ids.append(msg.message_id)
            return self.ui.PAGE_SEARCH

        if state.expecting_topk:
            state.expecting_topk = False
            try:
                k = int(text)
                if k <= 0 or k > 1000:
                    raise ValueError()
                state.top_k = k
                msg = await update.message.reply_text(f"✅ تم ضبط عدد النتائج TopK على: {k}", reply_markup=self.ui.build_main_menu(state))
                state.result_message_ids.append(msg.message_id)
                state.suppress_next_text = True
            except Exception:
                msg = await update.message.reply_text("❌ أدخل رقم صحيح أكبر من 0 (و≤ 1000).", reply_markup=self.ui.build_main_menu(state))
                state.result_message_ids.append(msg.message_id)
                state.suppress_next_text = True
            return self.ui.PAGE_SEARCH

        if state.expecting_pagesize:
            state.expecting_pagesize = False
            try:
                s = int(text)
                if s < self.cfg.page_size_min or s > self.cfg.page_size_max:
                    raise ValueError()
                state.page_size = s

                state.suppress_next_text = True

                self._delete_messages_safely(context.bot, chat_id, state.result_message_ids)
                state.result_message_ids.clear()

                if state.last_results:
                    state.total_pages = self.ui.compute_total_pages(len(state.last_results), state.page_size)
                    state.current_page = min(state.current_page, max(state.total_pages - 1, 0))
                    page_text = self.ui.render_search_page_text(state)
                    kb = self.ui.build_search_page_keyboard(state)
                    kb = self._with_smart_button(kb)
                    if len(page_text) > 3500:
                        ids = await self._send_long_text(chat_id, context, page_text, reply_markup=kb)
                        state.result_message_ids.extend(ids)
                    else:
                        out = await context.bot.send_message(
                            chat_id=chat_id,
                            text=page_text,
                            reply_markup=kb,
                            parse_mode=C.ParseMode.HTML,
                        )
                        state.result_message_ids.append(out.message_id)
                else:
                    msg = await update.message.reply_text(
                        f"✅ تم ضبط حجم الصفحة على: {s}",
                        reply_markup=self.ui.build_main_menu(state)
                    )
                    state.result_message_ids.append(msg.message_id)

            except Exception:
                msg = await update.message.reply_text(
                    f"❌ أدخل رقم بين {self.cfg.page_size_min} و {self.cfg.page_size_max}.",
                    reply_markup=self.ui.build_main_menu(state)
                )
                state.result_message_ids.append(msg.message_id)
                state.suppress_next_text = True
            return self.ui.PAGE_SEARCH

        if state.expecting_keyword:
            state.expecting_keyword = False
            if text.lower() == "تعطيل":
                state.keyword = None
                msg = await update.message.reply_text("🔑 تم تعطيل فلتر الكلمة.", reply_markup=self.ui.build_main_menu(state))
                state.result_message_ids.append(msg.message_id)
                return self.ui.PAGE_SEARCH
            state.keyword = text
            msg = await update.message.reply_text(f"✅ تم ضبط كلمة الفلترة على: {text}", reply_markup=self.ui.build_main_menu(state))
            state.result_message_ids.append(msg.message_id)
            state.suppress_next_text = True
            return self.ui.PAGE_SEARCH

        # ✍️ الحالة الأساسية: انتظار الاستعلام (الوضع المتقدم)
        if state.expecting_query:
            state.expecting_query = False
            state.query = text
            await self._run_search_and_show(update, context, chat_id, state, from_message=True, keep_page=0)
            return self.ui.PAGE_SEARCH

        return self.ui.PAGE_SEARCH

    # ---------- تشغيل بحث + عرض (متقدم) ----------
    async def _run_search_and_show(
        self,
        src,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        state: ChatState,
        from_message: bool = False,
        keep_page: Optional[int] = None,
    ):
        self._delete_messages_safely(context.bot, chat_id, state.result_message_ids)
        state.result_message_ids.clear()

        status_msg = None
        if from_message and getattr(src, "message", None):
            try:
                status_msg = await src.message.reply_text("⏳ عزيزي الطالب… جاري البحث …", parse_mode=C.ParseMode.HTML)
            except Exception:
                status_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ عزيزي الطالب… جاري البحث …", parse_mode=C.ParseMode.HTML)
        else:
            try:
                status_msg = await src.edit_message_text("⏳ عزيزي الطالب… جاري البحث …", parse_mode=C.ParseMode.HTML)
            except Exception:
                status_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ عزيزي الطالب… جاري البحث …", parse_mode=C.ParseMode.HTML)

        flt = SearchFilters(
            only_with_replies=state.only_replies,
            date_filter=state.date,
            date_range=state.date_range,
            keyword=state.keyword,
            only_with_contact=getattr(state, "only_with_contact", False),
        )
        logger.info(
            "run_search: q=%r top_k=%s only_replies=%s only_contact=%s date=%s range=%s keyword=%r",
            state.query, state.top_k, state.only_replies, getattr(state, "only_with_contact", False),
            state.date, state.date_range, state.keyword
        )

        try:
            uid = src.effective_user.id
        except Exception:
            uid = None
        if uid:
            self.state_mgr.track_search(uid, state.query or "", mode="adv")

        async with self._sem:
            results = await self.search_engine.search(query=state.query or "", top_k=state.top_k, flt=flt)

        # ✅ OpenAI: رد نهائي (لو مفعّل)
        answered = await self._maybe_openai_answer(
            question=state.query or "",
            results=results or [],
            chat_id=chat_id,
            context=context,
            status_message_id=getattr(status_msg, "message_id", None),
            back_markup=self.ui.build_main_menu(state),
        )
        if answered:
            return

        state.last_results = results
        state.total_pages = self.ui.compute_total_pages(len(results or []), state.page_size) if results else 0
        if keep_page is not None and state.total_pages:
            state.current_page = min(max(keep_page, 0), max(state.total_pages - 1, 0))
        else:
            state.current_page = 0

        logger.info("run_search: got %d results (page_size=%d, total_pages=%d)", len(results or []), state.page_size, state.total_pages)

        if not results:
            final_text = "❌ لا توجد نتائج مطابقة.\nجرّب تعديل الاستعلام أو الفلاتر."
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text=final_text,
                    reply_markup=self.ui.build_main_menu(state),
                    parse_mode=C.ParseMode.HTML,
                )
                state.result_message_ids.append(status_msg.message_id)
            except Exception:
                out = await context.bot.send_message(
                    chat_id=chat_id, text=final_text, reply_markup=self.ui.build_main_menu(state), parse_mode=C.ParseMode.HTML
                )
                state.result_message_ids.append(out.message_id)
            return

        page_text = self.ui.render_search_page_text(state)
        keyboard = self.ui.build_search_page_keyboard(state)
        keyboard = self._with_smart_button(keyboard)

        if len(page_text) > 3500:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            except Exception:
                pass
            ids = await self._send_long_text(chat_id, context, page_text, reply_markup=keyboard)
            state.result_message_ids.extend(ids)
            return

        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text=page_text,
                reply_markup=keyboard,
                parse_mode=C.ParseMode.HTML,
            )
            state.result_message_ids.append(status_msg.message_id)
        except Exception:
            out = await context.bot.send_message(
                chat_id=chat_id,
                text=page_text,
                reply_markup=keyboard,
                parse_mode=C.ParseMode.HTML,
            )
            state.result_message_ids.append(out.message_id)

    # ---------- replies callbacks ----------
    async def buttons_in_replies(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        chat_id = update.effective_chat.id
        state = self.state_mgr.get(chat_id)
        data = query.data

        if data == "back_to_search_same_page":
            self._delete_messages_safely(context.bot, chat_id, state.reply_page_message_ids)
            state.reply_page_message_ids.clear()
            if not state.last_results:
                is_admin = self._is_admin(update.effective_user.id)
                out = await context.bot.send_message(
                    chat_id=chat_id,
                    text="🏠 لا توجد نتائج حالية. رجعناك للقائمة.",
                    reply_markup=self._build_home_menu(is_admin),
                    parse_mode=C.ParseMode.HTML,
                )
                state.result_message_ids.append(out.message_id)
                return self.ui.PAGE_SEARCH

            text = self.ui.render_search_page_text(state)
            kb = self.ui.build_search_page_keyboard(state)
            kb = self._with_smart_button(kb)

            if len(text) > 3500:
                ids = await self._send_long_text(chat_id, context, text, reply_markup=kb)
                state.result_message_ids.extend(ids)
            else:
                out = await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=kb,
                    parse_mode=C.ParseMode.HTML,
                )
                state.result_message_ids.append(out.message_id)
            return self.ui.PAGE_SEARCH

        return self.ui.PAGE_REPLIES

    # ---------- misc ----------
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        state = self.state_mgr.get(chat_id)
        self._delete_messages_safely(context.bot, chat_id, state.result_message_ids)
        self._delete_messages_safely(context.bot, chat_id, state.reply_page_message_ids)
        state.result_message_ids.clear()
        state.reply_page_message_ids.clear()
        await update.message.reply_text("❌ تم إلغاء العملية.")
        return ConversationHandler.END

    async def cmd_reindex(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if self.cfg.admin_ids and uid not in self.cfg.admin_ids:
            await update.message.reply_text("❌ هذا الأمر للمشرفين فقط.")
            return
        await update.message.reply_text("🔄 جاري إعادة تحميل الملفات...")
        try:
            # لو متوفر: await self.search_engine.reload()
            await update.message.reply_text("✅ تمت إعادة التحميل بنجاح.")
        except Exception as e:
            logger.exception("reindex failed")
            await update.message.reply_text(f"❌ فشل إعادة التحميل: {e}")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        if self.cfg.admin_ids and uid not in self.cfg.admin_ids:
            await update.message.reply_text("❌ هذا الأمر للمشرفين فقط.")
            return

        total_msgs = len(getattr(self.search_engine.repo, "metas", []))
        index_dim = getattr(self.search_engine.index, "index_dim", None)
        cache_size = len(getattr(getattr(self.search_engine.model, "cache", None), "data", [])) if getattr(self.search_engine.model, "cache", None) else 0
        model_name = getattr(self.search_engine.model, "loaded_model_name", None)

        # معلومات FAISS
        kind = "Unknown"
        ntotal = None
        nlist = None
        nprobe_runtime = None
        hnsw_ef_search = None
        hnsw_M = None

        try:
            import faiss  # type: ignore
            idx = getattr(self.search_engine.index, "index", None)
            if idx is not None:
                ntotal = getattr(idx, "ntotal", None)

                # تحديد النوع
                if isinstance(idx, faiss.IndexIVF):
                    kind = "IVF"
                    try:
                        nlist = int(getattr(idx, "nlist", 0))
                    except Exception:
                        nlist = None
                    try:
                        nprobe_runtime = int(getattr(idx, "nprobe", None))
                    except Exception:
                        nprobe_runtime = None

                elif hasattr(idx, "hnsw"):  # HNSW
                    kind = "HNSW"
                    try:
                        hnsw = getattr(idx, "hnsw", None)
                        if hnsw is not None:
                            hnsw_ef_search = int(getattr(hnsw, "efSearch", None))
                            hnsw_M = int(getattr(hnsw, "M", None))
                    except Exception:
                        pass
                else:
                    kind = "FLAT"
        except Exception:
            pass

        # سطر nprobe حسب النوع
        if kind == "IVF":
            nprobe_line = f"NPROBE (cfg): {self.cfg.nprobe}"
            if nprobe_runtime is not None:
                nprobe_line += f" — (runtime): {nprobe_runtime}"
            if nlist is not None:
                nprobe_line += f" / nlist: {nlist}"
        elif kind == "HNSW":
            parts = []
            if hnsw_ef_search is not None:
                parts.append(f"efSearch: {hnsw_ef_search}")
            if hnsw_M is not None:
                parts.append(f"M: {hnsw_M}")
            nprobe_line = "HNSW " + (" / ".join(parts) if parts else "(بدون تفاصيل)")
        else:
            nprobe_line = "ℹ️ الفهرس FLAT (بحث شامل على كل المتجهات)."

        # هل BM25 مفعّل من الإعدادات
        bm25_on = getattr(self.cfg, "enable_bm25", True)
        emb_w = getattr(self.cfg, "emb_weight", 0.6)
        bm_w = getattr(self.cfg, "bm25_weight", 0.4)

        await update.message.reply_text(
            "📊 بيانات النظام:\n"
            f"Messages: {total_msgs}\n"
            f"Index kind: {kind}\n"
            f"Index ntotal: {ntotal}\n"
            f"Index dim: {index_dim}\n"
            f"{nprobe_line}\n"
            f"Cache size: {cache_size}\n"
            f"Model: {model_name}\n"
            f"BM25: {'ON' if bm25_on else 'OFF'} (emb_w={emb_w}, bm25_w={bm_w})"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        tl = self.ui.render_tldr_text()
        await update.message.reply_text(
            tl,
            reply_markup=self.ui.build_help_keyboard(),
            parse_mode=C.ParseMode.HTML
        )

    async def cmd_guide(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        guide = self.ui.render_about_text()
        ids = await self._send_long_text(update.effective_chat.id, context, guide, reply_markup=self.ui.build_about_keyboard())
        # ✅ تعديل بسيط: نخزن الرسائل في state علشان delete/تنضيف
        st = self.state_mgr.get(update.effective_chat.id)
        st.result_message_ids.extend(ids)

    async def cmd_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("search_bot_secure v3.7 — Quick-only for users, admin-only advanced, Smart Reply (no signature) + OpenAI Answer (optional)")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error("⚠️ Error: %s", context.error)
        try:
            import traceback
            traceback.print_exception(type(context.error), context.error, context.error.__traceback__)
        except Exception:
            pass
