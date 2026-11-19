"""Пример внешнего адаптера политики для жюри.

Запуск:  python -m auditgym run --policy examples.my_policy:SilentPolicy --seed 9001
         python -m auditgym evaluate --split holdout_example --policies examples.my_policy:SilentPolicy
"""


class SilentPolicy:
    """Сразу отправляет пустой ответ (молчание)."""

    def reset(self, observation, info):
        self.episode_id = observation["manifest"]["episode_id"]

    def act(self, observation):
        return {"tool": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": []}}}


class TruthOracle:
    """Пример oracle жюри: эталон передаёт харнесс оценщика (grader_only), а не среда."""
    grader_only = True

    def __init__(self, truth_provider):
        self.truth_provider = truth_provider

    def reset(self, observation, info):
        self.episode_id = observation["manifest"]["episode_id"]

    def act(self, observation):
        t = self.truth_provider(self.episode_id)
        fs = [{"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
               "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "oracle"}
              for i in t["issues"]]
        return {"tool": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": fs}}}
