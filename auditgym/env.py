"""RL-среда аудита: JSON-совместимый контракт в стиле Gymnasium.

reset(seed, config) -> (observation, info)
step(action)        -> (observation, reward, terminated, truncated, info)

Действие: {"tool": <имя>, "args": {...}} (dict или JSON-строка).
Инструменты: list_records, get_record, get_document, calculator, submit.

Режимы:
  train — терминальная награда и метрики возвращаются политике;
  eval  — политика всегда получает reward=0 и пустой info, результат хранится
          у оценщика (grader_result()); для настоящей изоляции используйте
          `python -m auditgym serve` (оценщик в отдельном процессе).
"""
from __future__ import annotations

import copy
import json
import time

from . import calculator
from .generator import TASK_TEXT, Episode, generate
from .matcher import grade
from .version import ENV_VERSION, SCHEMA_VERSION

TABLES = ("ledger", "invoices", "payments", "fx")
PAGE_LIMIT = 50
MAX_STEPS = 100
TOOLS = [
    {"name": "list_records", "args": {"table": "ledger|invoices|payments|fx", "cursor": "string|null",
                                       "limit": "int 1..50 (default 50)"}},
    {"name": "get_record", "args": {"table": "ledger|invoices|payments|fx",
                                     "id": "row_id | invoice_id | payment_id | 'DATE:SRC:TGT' for fx"}},
    {"name": "get_document", "args": {"doc_id": "doc_id of an invoice or payment"}},
    {"name": "calculator", "args": {"expression": "arithmetic on decimal numbers only"}},
    {"name": "submit", "args": {"findings": "{episode_id, findings:[...]} — ends the episode"}},
]
_ID_FIELD = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id"}


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message[:200]


def _fx_id(r):
    return f"{r['date']}:{r['source_currency']}:{r['target_currency']}"


def _is_submit(action) -> bool:
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except (ValueError, RecursionError):
            return False
    return isinstance(action, dict) and action.get("tool") == "submit"


class AuditEnv:
    metadata = {"env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION}

    def __init__(self, mode: str = "train", max_steps: int = MAX_STEPS, time_limit_s: float | None = 60.0,
                 recorder=None, clock=time.monotonic):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode, self.max_steps, self.time_limit_s = mode, int(max_steps), time_limit_s
        self.recorder, self._clock = recorder, clock
        self._ep: Episode | None = None
        self._done = True

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, config: dict | None = None, options: dict | None = None):
        """reset(seed, config). Совместимо с Gymnasium: config можно передать как options={"config": ...}.
        seed=None выбирает случайный seed (эпизод тогда невоспроизводим). Особый config
        {"episode_dir": <открытая папка>, "grader_dir": <закрытая папка>} загружает эпизод с диска."""
        if config is None and options:
            config = options.get("config", {k: v for k, v in options.items() if k != "config"})
        config = dict(config or {})
        if "episode_dir" in config:
            from .episode_io import load_episode
            return self._start(load_episode(config["episode_dir"], config.get("grader_dir") or config["episode_dir"]))
        if seed is None:
            import secrets
            seed = secrets.randbits(62)
        ep = generate(int(seed), config)
        return self._start(ep)

    def reset_episode(self, ep: Episode):
        """Запуск на заранее сгенерированном эпизоде (используется оценщиком)."""
        return self._start(ep)

    def step(self, action):
        if self._ep is None:
            raise RuntimeError("call reset() first")
        if self._done:
            obs = self._obs(None, None, ToolError("episode_finished", "episode already finished; call reset()"))
            return obs, 0.0, self._terminated, self._truncated, {}
        tool = None
        # время проверяется до счёта шага: действие после тайм-аута не выполняется и шагом не считается
        if self.time_limit_s is not None and self._clock() - self._t0 > self.time_limit_s:
            return self._finish_truncated(action, "time_limit")
        self._steps += 1
        try:
            tool, args = self._parse(action)
            if tool == "submit":
                return self._submit(action, args)
            result = self._dispatch(tool, args)
            err = None
        except ToolError as e:
            if _is_submit(action):
                # submit с неверной структурой действия = неверная отправка: эпизод завершается, награда −1
                return self._submit(action, {"findings": None})
            result, err = None, e
        if self._steps >= self.max_steps:
            return self._finish_truncated(action, "step_limit", tool, result, err)
        obs = self._obs(tool, result, err)
        self._record(action, obs, 0.0, False, False)
        return obs, 0.0, False, False, {}

    def abandon(self, why: str = "abandoned"):
        """Для оценщика: незавершённый эпизод закрывается как truncated с пустым ответом,
        все сделанные шаги учитываются. Ничего не делает, если эпизод уже завершён."""
        if self._ep is None or self._done:
            return None
        self._finish_truncated(None, why)
        return self.grader_result()

    def grader_result(self) -> dict | None:
        """Только для оценщика/харнесса: итог оценки последнего эпизода."""
        return copy.deepcopy(self._result)

    # ------------------------------------------------------------------ внутреннее
    def _start(self, ep: Episode):
        self._ep = ep
        self._tables = ep.tables
        self._index = {t: {r[_ID_FIELD[t]]: r for r in ep.tables[t]} for t in _ID_FIELD}
        self._index["fx"] = {_fx_id(r): r for r in ep.tables["fx"]}
        # документы можно искать и по doc_id
        self._index["invoices"].update({r["doc_id"]: r for r in ep.tables["invoices"]})
        self._index["payments"].update({r["doc_id"]: r for r in ep.tables["payments"]})
        self._docs = {r["doc_id"]: ("invoice", r) for r in ep.tables["invoices"]}
        self._docs.update({r["doc_id"]: ("payment", r) for r in ep.tables["payments"]})
        self._steps = 0
        self._done, self._terminated, self._truncated = False, False, False
        self._result = None
        self._t0 = self._clock()
        obs = {
            "task": TASK_TEXT,
            "manifest": copy.deepcopy(ep.manifest),
            "policy": copy.deepcopy(ep.policy),
            "tools": copy.deepcopy(TOOLS),
            "page_limit": PAGE_LIMIT,
            "steps_remaining": self.max_steps,
        }
        info = {"episode_id": ep.episode_id, "env_version": ENV_VERSION, "schema_version": SCHEMA_VERSION,
                "mode": self.mode}
        self._record(None, obs, 0.0, False, False, step=0)
        return obs, info

    def _parse(self, action):
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except (ValueError, RecursionError):
                raise ToolError("invalid_action", "action is not valid JSON") from None
        if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
            raise ToolError("invalid_action", "action must be an object with a string 'tool'")
        if set(action) - {"tool", "args"}:
            raise ToolError("invalid_action", "action may only contain 'tool' and 'args'")
        args = action.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ToolError("invalid_args", "'args' must be an object")
        if action["tool"] not in {t["name"] for t in TOOLS}:
            raise ToolError("unknown_tool", "unknown tool")
        return action["tool"], args

    def _table(self, args):
        t = args.get("table")
        if t not in TABLES:
            raise ToolError("unknown_table", "table must be one of ledger, invoices, payments, fx")
        return t

    def _dispatch(self, tool, args):
        if tool == "list_records":
            if set(args) - {"table", "cursor", "limit"}:
                raise ToolError("invalid_args", "unexpected arguments")
            t = self._table(args)
            limit = args.get("limit", PAGE_LIMIT)
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= PAGE_LIMIT:
                raise ToolError("invalid_args", f"limit must be an integer 1..{PAGE_LIMIT}")
            cur = args.get("cursor")
            rows = self._tables[t]
            if cur is None:
                off = 0
            else:
                try:
                    ct, co = str(cur).split("@", 1)
                    off = int(co)
                    assert ct == t and 0 <= off <= len(rows) and str(off) == co
                except Exception:
                    raise ToolError("invalid_cursor", "cursor is not valid for this table") from None
            page = rows[off:off + limit]
            nxt = off + len(page)
            return {"table": t, "records": copy.deepcopy(page), "total": len(rows),
                    "next_cursor": f"{t}@{nxt}" if nxt < len(rows) else None}
        if tool == "get_record":
            if set(args) - {"table", "id"}:
                raise ToolError("invalid_args", "unexpected arguments")
            t = self._table(args)
            rid = args.get("id")
            if not isinstance(rid, str) or rid not in self._index[t]:
                raise ToolError("unknown_id", "no record with this id in this table")
            return {"table": t, "record": copy.deepcopy(self._index[t][rid])}
        if tool == "get_document":
            if set(args) - {"doc_id"}:
                raise ToolError("invalid_args", "unexpected arguments")
            d = args.get("doc_id")
            if not isinstance(d, str) or d not in self._docs:
                raise ToolError("unknown_id", "no document with this doc_id")
            kind, rec = self._docs[d]
            return {"doc_type": kind, "document": copy.deepcopy(rec)}
        if tool == "calculator":
            if set(args) - {"expression"}:
                raise ToolError("invalid_args", "unexpected arguments")
            try:
                return {"value": calculator.evaluate(args.get("expression"))}
            except calculator.CalcError as e:
                raise ToolError("calculator_error", str(e)) from None
        raise ToolError("unknown_tool", "unknown tool")

    def _submit(self, action, args):
        sub = args.get("findings") if set(args) == {"findings"} else None
        if isinstance(sub, str):
            try:
                sub = json.loads(sub)
            except (ValueError, RecursionError):
                sub = None
        res = grade(sub, self._ep.truth, inspection_steps=self._steps - 1)
        res["outcome"] = "submitted"
        self._diagnose(res)
        self._result = res
        self._done, self._terminated = True, True
        msg = {"accepted": res["valid"]}   # без подсказок о правильности
        obs = self._obs("submit", msg, None if res["valid"] else ToolError("invalid_submission",
                                                                          "submission rejected: invalid structure"))
        reward, info = self._reward_info(res)
        self._record(action, obs, reward, True, False)
        return obs, reward, True, False, info

    def _finish_truncated(self, action, why, tool=None, result=None, err=None):
        res = grade({"episode_id": self._ep.episode_id, "findings": []}, self._ep.truth,
                    inspection_steps=self._steps)
        res["outcome"] = why
        self._diagnose(res)
        self._result = res
        self._done, self._truncated = True, True
        if why in ("time_limit", "abandoned"):
            tool, result, err = None, None, ToolError(why, "episode time limit exceeded" if why == "time_limit"
                                                      else "episode abandoned before submit")
        obs = self._obs(tool, result, err)
        reward, info = self._reward_info(res)
        self._record(action, obs, reward, False, True)
        return obs, reward, False, True, info

    def _diagnose(self, res):
        """Только для журнала оценщика: типы эталонных ошибок и к чему относятся ложные выводы."""
        ep = self._ep
        res["episode_id"] = ep.episode_id
        res["issue_types"] = {t["issue_id"]: t["type"] for t in ep.truth["issues"]}
        res["interactions"] = {t["issue_id"]: t["interaction"] for t in ep.truth["issues"] if t.get("interaction")}
        kind_of: dict[str, str] = {}
        for d in (ep.provenance or {}).get("distractors", []):
            for k, v in d.items():
                if k == "kind":
                    continue
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str):
                        kind_of.setdefault(x, d["kind"])
        known = set(self._index["invoices"]) | set(self._index["payments"])
        for u in res.get("unmatched_findings", []):
            ids = u.get("affected_ids", [])
            if u["reason"] in ("duplicate_of_matched", "amount", "evidence", "currency"):
                u["attribution"] = "real_issue_wrong_details" if u["reason"] != "duplicate_of_matched" else "repeat"
            elif any(i in kind_of for i in ids):
                u["attribution"] = "distractor:" + next(kind_of[i] for i in ids if i in kind_of)
            elif ids and all(i in known for i in ids):
                u["attribution"] = "clean_record"
            else:
                u["attribution"] = "unknown_id"

    def _reward_info(self, res):
        if self.mode == "train":
            keys = ("valid", "tp", "fp", "fn", "precision", "recall", "f1", "reward", "inspection_steps", "outcome")
            return res["reward"], {"grading": {k: res[k] for k in keys}}
        return 0.0, {}

    def _obs(self, tool, result, err):
        remaining = 0 if self._done else max(0, self.max_steps - self._steps)
        return {"step": getattr(self, "_steps", 0), "steps_remaining": remaining,
                "tool": tool, "result": result,
                "error": None if err is None else {"code": err.code, "message": err.message}}

    def _record(self, action, obs, reward, terminated, truncated, step=None):
        if self.recorder is None:
            return
        try:
            act = action if not isinstance(action, str) else json.loads(action)
            json.dumps(act)
        except (ValueError, TypeError, RecursionError):
            act = {"unparseable_action": True}
        self.recorder({"episode_id": self._ep.episode_id, "step": self._steps if step is None else step,
                       "action": act, "observation": obs, "reward": reward,
                       "terminated": terminated, "truncated": truncated})
