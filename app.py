# -*- coding: utf-8 -*-
"""search_bot.app

Main application bootstrap for the Telegram search bot.

This file was cleaned up to remove accidental duplicated blocks/classes that
caused maintenance headaches and could lead to subtle runtime issues.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    InlineQueryHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from .config import Config
from .handlers import Handlers
from .index import FaissIndex
from .model import EmbeddingModel
from .repository import DataRepository
from .search import Reranker, SearchEngine
from .state import StateManager
from .ui import UIBuilder

# ✅ OpenAI Answerer (اختياري)
try:
    from .openai_answerer import OpenAIAnswerer
except Exception:
    OpenAIAnswerer = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BotApp:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

        # Thread pool for IO/CPU bound tasks (loading, embedding, faiss, etc.)
        self.executor = ThreadPoolExecutor(max_workers=self.cfg.thread_workers)

        # Core services
        self.repo = DataRepository(self.cfg, self.executor)
        self.findex = FaissIndex(self.cfg, self.executor)
        self.emodel = EmbeddingModel(self.cfg, self.executor)
        self.reranker = Reranker(self.cfg, self.executor)

        # UI & State & Search
        self.ui = UIBuilder(self.cfg)
        self.state_mgr = StateManager(self.cfg)
        self.search_engine = SearchEngine(self.cfg, self.repo, self.findex, self.emodel, self.reranker)

        # ✅ OpenAI Answerer (اختياري)
        self.answerer = self._build_openai_answerer()

        # Telegram handlers
        self.handlers = Handlers(
            self.cfg,
            self.ui,
            self.state_mgr,
            self.search_engine,
            answerer=self.answerer,
        )

    def _build_openai_answerer(self):
        """Build OpenAIAnswerer if enabled; otherwise return None safely."""
        try:
            use_flag = bool(getattr(self.cfg, "use_openai_answer", False))
            if not use_flag:
                logger.info("OpenAI answers disabled (use_openai_answer=False).")
                return None

            if OpenAIAnswerer is None:
                logger.warning("OpenAIAnswerer import failed — OpenAI answers disabled.")
                return None

            api_key = (
                getattr(self.cfg, "openai_api_key", "")
                or os.environ.get("OPENAI_API_KEY", "")
            ).strip()
            if not api_key:
                logger.warning("OPENAI_API_KEY missing — OpenAI answers disabled.")
                return None

            model = (
                getattr(self.cfg, "openai_model", "")
                or getattr(self.cfg, "openai_chat_model", "")
                or os.environ.get("OPENAI_CHAT_MODEL", "")
                or "gpt-4o-mini"
            ).strip()

            logger.info("OpenAI answers enabled ✅ (model=%s)", model)
            return OpenAIAnswerer(api_key=api_key, model=model)

        except Exception as e:
            logger.warning("Failed to init OpenAIAnswerer — disabled. err=%s", e)
            return None

    async def load_resources(self):
        """Load repository, FAISS index, embedding model and reranker."""
        await self.repo.load()
        await self.findex.load()
        await self.emodel.load(self.findex.index_dim)
        await self.reranker.load()

        # Warn on mismatch between rows and FAISS index size
        try:
            if self.findex.index is not None and hasattr(self.findex.index, "ntotal"):
                ntotal = self.findex.index.ntotal
                if len(self.repo.metas) != ntotal:
                    logger.warning(
                        "Count mismatch: rows=%d vs index.ntotal=%d — تأكد من نفس المصدر والترتيب",
                        len(self.repo.metas),
                        ntotal,
                    )
        except Exception:
            pass

    def build_app(self):
        if not self.cfg.token:
            raise RuntimeError("QIMAH_BOT_TOKEN مفقود — ضع التوكن في ENV")

        req = HTTPXRequest(
            connect_timeout=20,
            read_timeout=20,
            write_timeout=20,
            pool_timeout=10,
        )
        return (
            ApplicationBuilder()
            .token(self.cfg.token)
            .request(req)
            .concurrent_updates(True)
            .build()
        )

    def conversation_handler(self) -> ConversationHandler:
        """Conversation flow: text is handled; all buttons are routed globally."""
        return ConversationHandler(
            entry_points=[CommandHandler("start", self.handlers.start)],
            states={
                self.ui.PAGE_HOME: [
                    MessageHandler(filters.TEXT & (~filters.COMMAND), self.handlers.text_in_search),
                ],
                self.ui.PAGE_SEARCH: [
                    MessageHandler(filters.TEXT & (~filters.COMMAND), self.handlers.text_in_search),
                ],
                self.ui.PAGE_REPLIES: [],
                self.handlers.DATE_PICKER: [],
                self.handlers.DATE_PICKER_RANGE_START: [],
                self.handlers.DATE_PICKER_RANGE_END: [],
            },
            fallbacks=[
                CommandHandler("start", self.handlers.start),
                CommandHandler("cancel", self.handlers.cancel),
            ],
            per_chat=True,
            per_user=True,
            per_message=False,
            name="search_replies_paged_flow",
            allow_reentry=True,
        )

    def run(self):
        logger.info("Starting bot — preparing event loop and loading resources...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.load_resources())
        except Exception:
            logger.exception("Failed to load resources")
            loop.close()
            raise

        app = self.build_app()

        # ✅ /start always works
        app.add_handler(CommandHandler("start", self.handlers.start, block=False), group=0)

        # ✅ Global callback router for all inline keyboards
        app.add_handler(CallbackQueryHandler(self.handlers.callback_router), group=0)

        # Conversation + other handlers
        app.add_handler(self.conversation_handler(), group=1)

        # Inline + commands
        app.add_handler(InlineQueryHandler(self.handlers.inline_query))
        app.add_handler(CommandHandler("reindex", self.handlers.cmd_reindex))
        app.add_handler(CommandHandler("stats", self.handlers.cmd_stats))
        app.add_handler(CommandHandler("version", self.handlers.cmd_version))
        app.add_handler(CommandHandler("help", self.handlers.cmd_help))
        app.add_handler(CommandHandler("guide", self.handlers.cmd_guide))

        # Global text fallback
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handlers.text_in_search), group=2)

        # Error handler
        app.add_error_handler(self.handlers.error_handler)

        logger.info("🤖 Bot is running (polling)...")
        try:
            app.run_polling(close_loop=False)
        finally:
            try:
                self.executor.shutdown(wait=True)
            finally:
                loop.close()
