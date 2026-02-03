# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import Any

class LRUCache:
    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self.data: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.max_size:
            self.data.popitem(last=False)
