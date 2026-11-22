"""judge_adapter.py — единый адаптер жюри FinTech для AuditGym.

Интерфейс из «Определений проверок FinTech» (genesis-exe.com/fintech-checks.html, 20.11.2026):

    generate_episode(seed, config, output_dir) -> None
    make_env() -> объект с reset(seed, config) и step(action)
    score_episode(episode_dir, findings, inspection_steps) -> dict

Рабочая директория — корень репозитория (в Docker-образе это /app). Адаптер только переводит
оболочку действий и форму ответов; смысл задачи, данные и оценщик — те же, что в auditgym/.

Каталог эпизода (generate_episode пишет, reset/score_episode читают):
    manifest.json  policy.json  ledger.csv  invoices.jsonl  payments.jsonl  fx.csv   — открытые данные
    truth.json  provenance.json  ledger_before.csv                                  — закрытая часть
Таблицы: ledger, invoices, payments, fx (канонические имена жюри совпадают с нашими).
ID записей: ledger.row_id, invoices.doc_id, payments.doc_id, fx — ключ "DATE:SRC:TGT".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditgym.env import AuditEnv  # noqa: E402
from auditgym.episode_io import export_episode, load_episode  # noqa: E402
from auditgym.generator import generate  # noqa: E402
from auditgym.matcher import grade, tau_from_policy, visible_ids  # noqa: E402

TABLES = ("ledger", "invoices", "payments", "fx")
ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")
TIME_LIMIT_S = 60.0
_TRAIN_INFO_KEYS = ("outcome", "valid", "tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps")


# ----------------------------------------------------------------------------- генерация

def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Обёртка над генератором команды: эпизод с seed и config (ключи — auditgym/config.py,
    например {"errors": {"fx_conversion": 1}, "interactions": {...}}) в каталог output_dir."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    cfg = dict(config or {})
    if "team_config" in cfg:                  # допускаем и оболочку {"mode": ..., "team_config": {...}}
        cfg = dict(cfg["team_config"] or {})
    ep = generate(seed, cfg)
    export_episode(ep, output_dir, output_dir)


# ----------------------------------------------------------------------------- среда

def make_env():
    return JudgeEnv()


class JudgeEnv:
    """reset(seed, config) -> (observation, info); step(action) -> (observation, reward, terminated,
    truncated, info). config: {"mode": "closed", "episode_dir": ..., "team_config": {...}} или
    {"mode": "train", "team_config": {...}} (+ необязательный episode_dir)."""

    def __init__(self):
        self._env = None
        self._mode = None

    def reset(self, seed=None, config=None):
        config = dict(config or {})
        mode = config.get("mode", "train")
        if mode not in ("closed", "train"):
            raise ValueError("mode must be 'closed' or 'train'")
        team_config = config.get("team_config") or {}
        episode_dir = config.get("episode_dir")
        # closed: награда политике всегда 0, info пуст; train: терминальная награда доступна
        self._env = AuditEnv(mode="eval" if mode == "closed" else "train", time_limit_s=TIME_LIMIT_S)
        self._mode = mode
        if episode_dir:
            ep = load_episode(episode_dir, episode_dir)       # именно этот эпизод, ID не меняются
            obs, info = self._env.reset_episode(ep)
        else:
            if mode == "closed":
                raise ValueError("closed mode needs episode_dir")
            obs, info = self._env.reset(seed, team_config)
        first = {"task": obs["task"], "manifest": obs["manifest"], "policy": obs["policy"],
                 "remaining_steps": obs["steps_remaining"]}
        return first, {"episode_id": info["episode_id"], "env_version": info["env_version"], "mode": mode}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("call reset first")
        obs, reward, terminated, truncated, info = self._env.step(_to_native(action))
        out = {"result": _result(obs.get("tool"), obs.get("result")), "error": obs.get("error"),
               "remaining_steps": obs.get("steps_remaining", 0)}
        if self._mode == "closed":
            return out, 0.0, bool(terminated), bool(truncated), {}
        g = info.get("grading", info) if (terminated or truncated) else {}
        safe = {k: g[k] for k in _TRAIN_INFO_KEYS if k in g}     # только счётчики, без ID эталона
        return out, float(reward), bool(terminated), bool(truncated), safe


def _to_native(action):
    """{"name": ..., "args": {...}} -> внутренний {"tool": ..., "args": {...}}."""
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except (ValueError, RecursionError):
            return action                     # среда вернёт ограниченную ошибку и спишет шаг
    if isinstance(action, dict) and "name" in action and "tool" not in action:
        native = {"tool": action["name"], "args": action.get("args", {})}
        extra = set(action) - {"name", "args"}
        for k in extra:                       # лишние ключи оболочки -> среда вернёт invalid_action
            native[k] = action[k]
        return native
    return action


def _result(tool, res):
    if res is None:
        return None
    if tool == "list_records":
        return {"records": res["records"], "next_cursor": res["next_cursor"]}
    if tool == "get_record":
        return res["record"]
    if tool == "get_document":
        return res["document"]
    if tool == "calculator":
        return res["value"]                   # десятичная строка, например "0.3"
    return res                                # submit -> {"accepted": true}


# ----------------------------------------------------------------------------- оценщик

def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    """Закрытый вызов оценщика. Валидный submit -> {tp, fp, fn, precision, recall, f1, return};
    невалидный -> return=-1, остальные поля null, причина пишется в stderr (журнал оценщика)."""
    ep = load_episode(episode_dir, episode_dir)
    if isinstance(findings, str):
        try:
            findings = json.loads(findings)
        except (ValueError, RecursionError):
            findings = None
    if type(inspection_steps) is not int or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative integer")
    g = grade(findings, ep.truth, inspection_steps, visible=visible_ids(ep.tables), tau=tau_from_policy(ep.policy))
    if not g["valid"]:
        sys.stderr.write(f"auditgym score_episode: invalid submit: {g.get('error')}\n")
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1.0}
    return {"tp": g["tp"], "fp": g["fp"], "fn": g["fn"], "precision": g["precision"], "recall": g["recall"],
            "f1": g["f1"], "return": g["reward"]}
