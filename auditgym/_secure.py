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


POLICY_USER_DEFAULT = "auditgym-policy"


def sandbox_spec(user: str | None = None):
    """Если харнесс запущен от root и есть отдельный пользователь политики — запускать политику
    от его имени (так в Docker-образе). Иначе None: политика работает под тем же пользователем."""
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        return None
    import pwd
    name = user or os.environ.get("AUDITGYM_POLICY_USER", POLICY_USER_DEFAULT)
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        return None
    if pw.pw_uid == 0:
        return None
    return {"user": name, "uid": pw.pw_uid, "gid": pw.pw_gid}


def fresh_home(spec) -> str:
    """Новый пустой HOME (700) для одного процесса политики."""
    import tempfile
    home = tempfile.mkdtemp(prefix="auditgym-policy-")
    os.chown(home, spec["uid"], spec["gid"])
    os.chmod(home, 0o700)
    return home


def purge_user_files(uid: int, dirs=("/tmp", "/var/tmp", "/dev/shm")) -> int:
    """Удалить всё, что пользователь политики оставил в общих временных каталогах."""
    import shutil
    import stat
    n = 0
    for d in dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            path = os.path.join(d, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if st.st_uid != uid:
                continue
            try:
                if stat.S_ISDIR(st.st_mode):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
                n += 1
            except OSError:
                pass
    return n


def sandbox_preexec():
    """В дочернем процессе политики: своя сессия и лимиты ресурсов."""
    import resource
    os.setsid()
    for lim, val in ((resource.RLIMIT_AS, 3 << 30), (resource.RLIMIT_NPROC, 128),
                     (resource.RLIMIT_FSIZE, 256 << 20), (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass


def kill_user_processes(uid: int) -> int:
    """Убить все процессы пользователя политики (демоны, пережившие эпизод). Только для отдельного uid."""
    import signal
    n = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/status") as fh:
                uid_line = next(l for l in fh if l.startswith("Uid:"))
            if int(uid_line.split()[1]) == uid:
                os.kill(int(d), signal.SIGKILL)
                n += 1
        except (OSError, StopIteration, ValueError):
            pass
    return n
