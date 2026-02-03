# -*- coding: utf-8 -*-
import os, asyncio, logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from .config import Config

logger = logging.getLogger(__name__)

# استيراد faiss بعد قراءة OMP threads من الكونفج
try:
    import faiss
except Exception as e:
    raise

class FaissIndex:
    def __init__(self, cfg: Config, executor: ThreadPoolExecutor):
        self.cfg = cfg
        self.executor = executor
        self.index: Optional["faiss.Index"] = None
        self.index_dim: Optional[int] = None

    def resolve_index_path(self, ipath: str) -> str:
        if ipath and os.path.isfile(ipath):
            return ipath
        default = self.cfg.default_index_path
        if os.path.isfile(default):
            logger.warning("INDEX_PATH not found/empty: %s — using default %s", ipath, default)
            return default
        raise FileNotFoundError(f"INDEX_PATH not found: {ipath} and default {default} not found")

    def _load_index_sync(self):
        # ضبط عدد خيوط FAISS (لو مدعوم)
        try:
            faiss.omp_set_num_threads(self.cfg.faiss_omp_threads)
            logger.info("FAISS omp_set_num_threads=%s", self.cfg.faiss_omp_threads)
        except Exception:
            pass
        resolved = self.resolve_index_path(self.cfg.index_path)
        logger.info("Reading FAISS index from: %s", resolved)
        return faiss.read_index(resolved)

    async def load(self):
        loop = asyncio.get_running_loop()
        index = await loop.run_in_executor(self.executor, self._load_index_sync)
        self.index = index
        try:
            ivf = faiss.extract_index_ivf(self.index)
            if ivf is not None:
                ivf.nprobe = self.cfg.nprobe
                logger.info("FAISS index loaded, set nprobe=%s", self.cfg.nprobe)
        except Exception:
            pass
        self.index_dim = self.index.d
        logger.info("FAISS index loaded (dim=%s, ntotal=%s)", self.index_dim, getattr(self.index, "ntotal", "N/A"))
