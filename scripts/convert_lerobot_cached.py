#!/usr/bin/env python3
"""Run the delivered converter with per-file NumPy array caching.

Accepts the same arguments as 转换LeRobot数据集_rebot.py. The delivered
converter, frame values, statistics, and LeRobot writer remain unchanged.
Arrays are released with their NpzFile instead of accumulating across files.
"""
from pathlib import Path
import runpy

import numpy as np


def main():
    original = np.lib.npyio.NpzFile.__getitem__

    def cached_getitem(self, key):
        cache = self.__dict__.setdefault("_rebot_array_cache", {})
        if key not in cache:
            cache[key] = original(self, key)
        return cache[key]

    np.lib.npyio.NpzFile.__getitem__ = cached_getitem
    try:
        runpy.run_path(
            str(Path(__file__).with_name("转换LeRobot数据集_rebot.py")),
            run_name="__main__",
        )
    finally:
        np.lib.npyio.NpzFile.__getitem__ = original


if __name__ == "__main__":
    main()
