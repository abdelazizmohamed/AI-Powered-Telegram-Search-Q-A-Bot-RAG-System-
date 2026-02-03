# -*- coding: utf-8 -*-
import os
import re
import html
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    # Optional dependency. If the package isn't installed, we only fail when the
    # feature is actually enabled/used.
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class OpenAIAnswerer:
    """
    Answerer مقيد بالمصدر:
    - بياخد سؤال المستخدم + نتائج البحث (messages)
    - يجاوب فقط من المراجع اللي اتبعتتله
    - لو مش موجود: يقول "غير موجود في الداتا" (داخليًا) ثم نطبّق فورمات العرض المطلوب.
    - بيرجع نص HTML آمن لتيليجرام (ParseMode.HTML)
      * ممنوع أي HTML من الموديل — بنعمل escape بالكامل.
    """

    NO_ANSWER_MSG = "لقد بحثت في الذاكرة ولم اجد إجابة على استفسارك"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ModuleNotFoundError(
                "The 'openai' package is not installed. Install it (pip install openai) "
                "or disable USE_OPENAI_ANSWER in your environment."
            )
        api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")
        self.client = OpenAI(api_key=api_key)
        self.model = (model or "gpt-4o-mini").strip()

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _clean(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s

    @staticmethod
    def _truncate(s: str, n: int) -> str:
        s = (s or "")
        return s[:n] + ("…" if len(s) > n else "")

    @staticmethod
    def _safe(s: str) -> str:
        # Escape لأي HTML عشان ParseMode.HTML ما يتكسرش
        return html.escape(s or "")

    @staticmethod
    def _normalize_headings(t: str) -> str:
        """Normalize common alternative headings produced by the model."""
        t = (t or "")
        t = t.replace("✅ إجابة مختصرة", "✅ الإجابة")
        t = t.replace("🧾 تفاصيل", "🧠 الشرح")
        return t

    @staticmethod
    def _dedupe_consecutive_lines(s: str) -> str:
        """Remove consecutive duplicate lines and extra blank lines."""
        lines = (s or "").splitlines()
        out: List[str] = []
        for ln in lines:
            raw = ln.rstrip()
            stripped = raw.strip()

            # collapse multiple blank lines
            if not stripped:
                if out and out[-1] == "":
                    continue
                out.append("")
                continue

            # drop consecutive duplicates (ignoring surrounding spaces)
            if out and out[-1].strip() == stripped:
                continue
            out.append(raw)

        return "\n".join(out).strip()

    @staticmethod
    def _is_no_data(s: str) -> bool:
        """Detect the no-answer condition."""
        t = (s or "").strip()
        if not t:
            return True
        if "غير موجود في الداتا" in t:
            return True
        # Sometimes the model writes our user-facing phrasing directly.
        if OpenAIAnswerer.NO_ANSWER_MSG in t:
            return True
        if "لم اجد" in t and "إجابة" in t and "الذاكرة" in t:
            return True
        if "لا توجد مراجع" in t:
            return True
        return False

    def _split_answer_and_more(self, model_text: str) -> Tuple[str, str]:
        """Extract (answer, more) from model output (supports old/new formats)."""
        t = self._clean(model_text)
        # Remove any accidental "sources" tail
        t = re.sub(r"🔗\s*مصادر.*$", "", t, flags=re.DOTALL).strip()
        t = self._normalize_headings(t)

        # New format: <answer>\n\n🧠 مزيد من المعلومات\n<more>
        if "🧠 مزيد من المعلومات" in t:
            left, right = t.split("🧠 مزيد من المعلومات", 1)
            answer = left.strip()
            more = right.strip()
            # Drop any stray old headings in the answer part
            answer_lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
            answer_lines = [ln for ln in answer_lines if ln not in {"✅ الإجابة", "🧠 الشرح"}]
            answer = "\n".join(answer_lines).strip()
            return (answer, more)

        # Old format: ✅ الإجابة ...\n\n🧠 الشرح ...
        lines = [ln.rstrip() for ln in t.splitlines()]
        ans_i = None
        exp_i = None
        for i, ln in enumerate(lines):
            if ans_i is None and ln.strip() == "✅ الإجابة":
                ans_i = i
                continue
            if exp_i is None and ln.strip() in {"🧠 الشرح", "🧠 مزيد من المعلومات"}:
                exp_i = i

        if ans_i is not None and exp_i is not None and exp_i > ans_i:
            answer = "\n".join(lines[ans_i + 1 : exp_i]).strip()
            more = "\n".join(lines[exp_i + 1 :]).strip()
            return (answer, more)

        # Fallback: treat whole output as answer only.
        # We can mirror it later under "🧠 مزيد من المعلومات" if needed.
        t2 = t.strip()
        return (t2, "")

    def _format_final_output(self, model_text: str) -> str:
        """Return the final Telegram-ready HTML-safe output with the requested format."""
        answer, more = self._split_answer_and_more(model_text)

        # No-answer condition -> fixed text in both sections
        if self._is_no_data(answer) or self._is_no_data(more):
            msg = self.NO_ANSWER_MSG
            final_plain = f"{msg}\n\n🧠 مزيد من المعلومات\n{msg}"
            return self._safe(final_plain)

        answer = (answer or "").strip()
        more = (more or "").strip()

        # Remove repeated lines that make the UX noisy.
        answer = self._dedupe_consecutive_lines(answer)
        more = self._dedupe_consecutive_lines(more)

        # If one side is empty, mirror the other
        if not answer and more:
            answer = more
        if not more and answer:
            more = answer

        final_plain = f"{answer}\n\n🧠 مزيد من المعلومات\n{more}"
        return self._safe(final_plain)

    # -----------------------------
    # Build references
    # -----------------------------
    def build_refs(
        self,
        results: List[Dict[str, Any]],
        max_items: int = 8,
        max_chars_each: int = 800
    ) -> str:
        """
        يرجع نص المراجع اللي هنبعته للموديل (بدون روابط — عشان نفضل مقيدين بالمصدر فقط)
        """
        chunks: List[str] = []

        for i, item in enumerate((results or [])[:max_items], start=1):
            seed = item.get("seed", {}) or {}

            msg = self._clean(seed.get("message") or "")
            user = (seed.get("user") or seed.get("username") or "").strip()
            date_str = (seed.get("date_str") or "").strip()

            # ضيف أفضل رد لو موجود (بيحسن الإجابة)
            best_reply_text = ""
            try:
                br = item.get("best_reply")
                if isinstance(br, (list, tuple)) and len(br) == 2 and isinstance(br[1], dict):
                    br_meta = br[1]
                    br_msg = self._clean(br_meta.get("message") or "")
                    if br_msg:
                        best_reply_text = f"\n\n[أفضل رد]\n{br_msg}"
            except Exception:
                pass

            # ضيف بعض الردود الأخرى (إن وُجدت) — بدون روابط
            try:
                reps = item.get("replies") or []
                shown = 0
                extra_parts = []
                for rr in reps:
                    if shown >= 3:
                        break
                    if isinstance(rr, (list, tuple)) and len(rr) >= 2 and isinstance(rr[1], dict):
                        r_meta = rr[1]
                    elif isinstance(rr, dict):
                        r_meta = rr
                    else:
                        continue
                    r_msg = self._clean(r_meta.get("message") or "")
                    if not r_msg:
                        continue
                    extra_parts.append(r_msg)
                    shown += 1
                if extra_parts:
                    best_reply_text = (best_reply_text or "") + "\n\n[ردود أخرى]\n" + "\n---\n".join(extra_parts)
            except Exception:
                pass

            msg = self._truncate(msg, max_chars_each)
            if best_reply_text:
                best_reply_text = self._truncate(best_reply_text, max_chars_each)

            chunks.append(f"[{i}] ({date_str}) {user}\n{msg}{best_reply_text}".strip())

        return "\n\n---\n\n".join(chunks).strip()

    # -----------------------------
    # Main (non-streaming)
    # -----------------------------
    def answer(
        self,
        question: str,
        results: List[Dict[str, Any]],
        max_items: int = 8,
        max_chars: int = 800,
    ) -> str:
        question = (question or "").strip()
        if not question:
            return self._safe("❌ سؤال فارغ.")

        refs_text = self.build_refs(results, max_items=max_items, max_chars_each=max_chars)
        if not refs_text:
            # دي رسالة داخلية ثابتة (مش من الموديل)
            return self._format_final_output("غير موجود في الداتا.")

        # Prompt صارم: بدون مصادر + ممنوع HTML/روابط
        system = (
            "أنت مساعد يجيب فقط من (المراجع) المرفقة.\n"
            "قواعد صارمة:\n"
            "1) ممنوع استخدام أي معرفة خارج المراجع.\n"
            "2) ممنوع الافتراض أو التخمين.\n"
            "3) لو المعلومة غير موجودة صراحة في المراجع: لا تخترع.\n"
            "4) ممنوع استخدام أي وسوم HTML أو روابط داخل النص.\n"
            "5) التزم فورمات الإخراج EXACT بدون أي أقسام إضافية:\n\n"
            "✅ الإجابة\n"
            "<إجابة مختصرة> أو: غير موجود في الداتا.\n\n"
            "🧠 مزيد من المعلومات\n"
            "<نقاط قصيرة (3-7) توضح الاستدلال من المراجع فقط> أو: غير موجود في الداتا.\n"
        )

        user = (
            f"السؤال:\n{question}\n\n"
            f"المراجع (المصدر الوحيد):\n{refs_text}\n\n"
            "اكتب الإجابة بالعربي وبنفس الفورمات المطلوبة."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.exception("OpenAI answer failed: %s", e)
            return self._safe("⚠️ حصل خطأ أثناء توليد الإجابة.")

        return self._format_final_output(content)

    # -----------------------------
    # Main (streaming)
    # -----------------------------
    def stream_answer(
        self,
        question: str,
        results: List[Dict[str, Any]],
        max_items: int = 8,
        max_chars: int = 800,
    ):
        """
        Stream model output as text deltas (no HTML) using chat.completions streaming.
        Important:
        - دي التدفّقات خام (بدون HTML).
        - التنسيق النهائي المطلوب (مزيد من المعلومات...) بيتم في الهاندلر بعد التجميع،
          أو تقدر تستخدم _format_final_output لو مش هتعمل preview حيّة.
        """
        question = (question or "").strip()
        if not question:
            yield "❌ سؤال فارغ."
            return

        refs_text = self.build_refs(results, max_items=max_items, max_chars_each=max_chars)
        if not refs_text:
            # خليها بنفس المنطق: نطلع "غير موجود في الداتا" (والهاندلر يحولها للصيغة النهائية)
            yield "غير موجود في الداتا."
            return
        system = (
            "أنت مساعد يجيب فقط من (المراجع) المرفقة.\n"
            "قواعد صارمة:\n"
            "1) ممنوع استخدام أي معرفة خارج المراجع.\n"
            "2) ممنوع الافتراض أو التخمين.\n"
            "3) لو المعلومة غير موجودة صراحة في المراجع: لا تخترع.\n"
            "4) ممنوع استخدام أي وسوم HTML أو روابط داخل النص.\n"
            "5) التزم فورمات الإخراج EXACT بدون أي أقسام إضافية:\n\n"
            "✅ الإجابة\n"
            "<إجابة مختصرة> أو: غير موجود في الداتا.\n\n"
            "🧠 مزيد من المعلومات\n"
            "<نقاط قصيرة (3-7) توضح الاستدلال من المراجع فقط> أو: غير موجود في الداتا.\n"
        )

        user = (
            f"السؤال:\n{question}\n\n"
            f"المراجع (المصدر الوحيد):\n{refs_text}\n\n"
            "اكتب الإجابة بالعربي وبنفس الفورمات المطلوبة."
        )

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                stream=True,
            )
            for event in stream:
                try:
                    delta = event.choices[0].delta.content
                except Exception:
                    delta = None
                if delta:
                    yield delta
            return
        except Exception as e:
            logger.warning("OpenAI streaming failed, falling back to non-streaming: %s", e)

        # Fallback: non-streaming
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = self._clean(content)
            content = re.sub(r"🔗\s*مصادر.*$", "", content, flags=re.DOTALL).strip()
            yield content
        except Exception as e:
            logger.exception("OpenAI answer failed: %s", e)
            yield "⚠️ حصل خطأ أثناء توليد الإجابة."
