# -*- coding: utf-8 -*-
import re, html
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ------------------ Patterns ------------------
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")
USERNAME_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]{4,})\b")

# ------------------ Arabic normalization ------------------
AR_DIAC = dict.fromkeys(map(ord, "\u0617\u0618\u0619\u061A\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670\u0640"), None)
AR_STOP = {
    "من","في","على","الى","إلى","عن","أن","إن","كان","كانت","يكون","هذا","هذه","ذلك","تلك",
    "ثم","أو","أمام","مع","بين","حتى","بعد","قبل","كل","أي","اذا","إذا","ما","لم","لن","لا","قد","لقد",
}

# ------------------ Basic utils ------------------
def mask_sensitive(text: str) -> str:
    """أخفِ أرقام الهواتف مع إظهار آخر 4 أرقام فقط."""
    def _mask(m: re.Match[str]) -> str:
        s = re.sub(r"[^0-9]", "", m.group(0))
        if len(s) <= 4:
            return "****"
        return "****" + s[-4:]
    return PHONE_RE.sub(_mask, text)

def build_date_str(m: Dict[str, Any]) -> str:
    """تركيب نص زمني موحّد من الحقول المتاحة داخل الميتا."""
    try:
        if m.get("date_iso_local"):
            s = m["date_iso_local"].replace("T", " ").replace("Z", "")
            return s
        y, mo, d = m.get("year", 0), m.get("month", 0), m.get("day", 0)
        hh, mm, ss = m.get("hour", 0), m.get("minute", 0), m.get("second", 0)
        if y and mo and d:
            return f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}:{ss:02d}"
    except Exception:
        pass
    return ""

def parse_user_date(s: str) -> Optional[Tuple[int, int, int]]:
    s = s.strip()
    m = re.match(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            _ = datetime(y, mo, d); return (y, mo, d)
        except Exception:
            return None
    m = re.match(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$", s)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            _ = datetime(y, mo, d); return (y, mo, d)
        except Exception:
            return None
    return None

def parse_date_range(s: str) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    s = s.strip().replace(" ", "")
    s = s.replace("إلى", "..").replace("to", "..")
    if ".." not in s: return None
    left, right = s.split("..", 1)
    d1 = parse_user_date(left); d2 = parse_user_date(right)
    if not d1 or not d2: return None
    from datetime import datetime as _dt
    if _dt(*d1) > _dt(*d2):
        d1, d2 = d2, d1
    return (d1, d2)

def normalize_ar(text: str) -> str:
    """تطبيع عربي خفيف: إزالة التشكيل، توحيد الألف/الياء/التاء المربوطة، تنظيف الرموز."""
    if not text: return ""
    t = text.translate(AR_DIAC)
    t = t.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    t = t.replace("ى","ي").replace("ة","ه")
    t = re.sub(r"[^\w@#\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize_ar(text: str) -> List[str]:
    """توكنايز عربي مبسط + إزالة ستوب ووردز الشائعة."""
    if not text: return []
    t = normalize_ar(text)
    toks = [w for w in t.split() if w and (len(w) > 1) and (w not in AR_STOP)]
    return toks

def safe_truncate(text: str, max_chars: int) -> str:
    """قصّ آمن بدون كسر الهايلت أو الإيموجيز (قصّ بسيط على مستوى الحروف)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"

# ------------------ Query terms extraction ------------------
def extract_terms(query: str) -> List[str]:
    """استخرج الكلمات المفتاحية من الكواري (بدون السالب -term) وبترتيب تنازلي حسب الطول."""
    if not query: return []
    raw = re.split(r"[\s|]+", query.strip())
    terms: List[str] = []
    for t in raw:
        if not t or t.startswith("-"): continue
        if len(t) < 2: continue
        terms.append(t)
    terms = sorted(set(terms), key=len, reverse=True)
    return terms

# ------------------ Telegram link helpers ------------------
def make_tg_link(chat_id: Optional[int], message_id: Optional[int]) -> Optional[str]:
    """
    يبني رابط t.me/c/<internal_chat_id>/<message_id> لو الداتا متاحة.
    ملاحظة: في السوبرجروب/القنوات يكون chat_id سالب ويبدأ بـ -100.
    """
    if chat_id is None or message_id is None:
        return None
    try:
        cid = int(chat_id)
        mid = int(message_id)
    except Exception:
        return None
    # تحويل -100xxxxxxxxxx إلى xxxxxxxxxx
    if cid < 0:
        cid = abs(cid)
        if str(cid).startswith("100"):
            cid = int(str(cid)[3:])
    return f"https://t.me/c/{cid}/{mid}"

# ------------------ Highlighting ------------------
_HL_START = "\x00HL\x00"
_HL_END   = "\x00/HL\x00"

def _mark_terms(raw: str, terms: List[str]) -> str:
    """علِّم مواضع الكلمات في النص الخام باستخدام placeholders ثم نهرب HTML لاحقًا."""
    if not raw or not terms:
        return raw or ""
    out = raw
    for term in terms[:8]:  # حد معقول لعدم تضخيم النص
        try:
            # IGNORECASE + لا نتلاعب بالـ HTML هنا (نشتغل على النص الخام)
            pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
            out = pattern.sub(_HL_START + r"\g<0>" + _HL_END, out)
        except re.error:
            continue
    return out

def _first_mark_span(s: str) -> Optional[Tuple[int, int]]:
    """إرجاع إندكس أول span مُعلّم لتحديد سنِبت حوالينه."""
    start = s.find(_HL_START)
    if start == -1:
        return None
    end = s.find(_HL_END, start + len(_HL_START))
    if end == -1:
        return None
    return (start, end + len(_HL_END))

def highlight_html(text: str, query: str, max_chars: int = 0, surround: int = 90) -> str:
    """
    هايلايت HTML آمن:
      1) نعلّم على النص الخام place-holders.
      2) نقصّ سنِبت حوالين أول تطابق لو max_chars > 0.
      3) نهرب HTML كامل.
      4) نستبدل placeholders بعلامات <b>.
    """
    if not text:
        return ""
    if not query or not query.strip():
        t = mask_sensitive(text)
        t = html.escape(t)
        return t

    terms = extract_terms(query)
    if not terms:
        t = mask_sensitive(text)
        t = html.escape(t)
        return t

    raw = mask_sensitive(text)
    marked = _mark_terms(raw, terms)

    # قصّ سنبت حول أول تطابق إن لزم
    snippet = marked
    if max_chars and len(marked) > max_chars:
        span = _first_mark_span(marked)
        if span:
            s, e = span
            # نحاول نأخذ جزء حوالين أول تطابق
            center = (s + e) // 2
            left = max(0, center - surround)
            right = min(len(marked), center + surround)
            snippet = ("…" if left > 0 else "") + marked[left:right] + ("…" if right < len(marked) else "")
        else:
            snippet = safe_truncate(marked, max_chars)

    # 1) نهرب HTML
    escaped = html.escape(snippet)

    # 2) رجّع الـ <b> بدل placeholders
    escaped = escaped.replace(html.escape(_HL_START), "<b>").replace(html.escape(_HL_END), "</b>")

    return escaped
