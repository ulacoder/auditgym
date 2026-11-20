"""Защита процессов оценщика от процесса политики (тот же пользователь ОС).

* harden() — запрещает чтение /proc/<pid>/environ, /proc/<pid>/mem, /proc/<pid>/fd и ptrace
  этого процесса другими процессами того же пользователя (Linux, PR_SET_DUMPABLE=0).
  Секрет оценщика и каналы с эталоном/итогами таким образом недоступны политике.
* LineReader — чтение JSON-строк из канала с тайм-аутом (зависшая политика не вешает харнесс).
* package_digest() — хеш кода оценщика; харнесс проверяет, что политика его не изменила.
"""
from __future__ import annotations

import hashlib
import os
import select
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent


def harden() -> bool:
    """True, если процесс удалось сделать недампируемым (Linux)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_DUMPABLE = 4
        return libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) == 0
    except Exception:
        return False


def package_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(PKG.rglob("*.py")):
        h.update(str(p.relative_to(PKG)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


class Timeout(Exception):
    pass


class LineReader:
    """Построчное чтение из файлового дескриптора с крайним сроком (time.monotonic())."""

    MAX_LINE = 64 * 1024 * 1024

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = b""

    def readline(self, deadline: float | None = None) -> bytes | None:
        """Строка без '\\n'; None при EOF; Timeout, если срок истёк."""
        import time
        while b"\n" not in self.buf:
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise Timeout()
                r, _, _ = select.select([self.fd], [], [], left)
                if not r:
                    raise Timeout()
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                return None
            self.buf += chunk
            if len(self.buf) > self.MAX_LINE:
                raise ValueError("line too long")
        line, self.buf = self.buf.split(b"\n", 1)
        return line
