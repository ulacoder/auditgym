# AuditGym для жюри: как проверить среду своими политиками и seed

Всё ниже работает без сети, на Python 3.11+ из корня репозитория (или в контейнере).

## 0. Секрет оценщика (обязательно для закрытого прогона)

```bash
export AUDITGYM_SECRET='<любая длинная случайная строка>'
```

Секрет подмешивается во все потоки случайности, публичные ID и `episode_id`. Без него агент,
у которого есть исходный код, может перебором найти небольшой seed по manifest или ID и
восстановить эталон. Открытые сплиты (train/dev/holdout_example) и `configs/FINGERPRINT` собраны
с публичным секретом по умолчанию (`auditgym-open-v1`); это значение для закрытого прогона не
принимается. При вашем секрете тот же seed даёт другую, но так же воспроизводимую компанию.

Передавайте секрет прямо в окружение процесса `evaluate`/`serve` (например, `docker run -e
AUDITGYM_SECRET=... auditgym:1.1.0 python -m auditgym evaluate ...` или `export` + `exec`), а не
через долгоживущие обёртки: процессы оценщика защищены от чтения `/proc`, а чужая оболочка — нет.

## 1. Закрытые seed и конфигурации

Файл сплита (формат как у `configs/dev.json`):

```json
{"config_version": "jury-v1", "episodes": [
  {"name": "h00", "seed": 918273, "config": {"errors": {"fx_conversion": 1, "period_cutoff": 1},
                                             "interactions": {"partial_fx_payment_at_boundary": 1},
                                             "id_salt": "x7", "order_seed": 42}}
]}
```

Ключи `config` описаны в `auditgym/config.py` (`DEFAULT_CONFIG`). Неизвестный ключ (в том числе
вложенный), дробное или булево число ошибок, неверный пояс, валюта, будущий период или
противоречивые `error_options` — сразу `ValueError`. Если в файле есть `env_version`, он должен
совпадать с версией среды (сейчас `1.1.0`).

```bash
python -m auditgym evaluate --split /path/hidden.json            # 4 базовые политики + наш аудитор
python -m auditgym export --splits /path/hidden.json --out /tmp/x # выгрузить эпизоды на диск
python -m auditgym evaluate --split /path/hidden.json --time-limit 60  # секунд на эпизод (по умолчанию 60)
```

## 2. Свои политики (адаптеры)

Политика — любой класс с двумя методами:

```python
class MyPolicy:
    def reset(self, observation: dict, info: dict) -> None: ...
    def act(self, observation: dict) -> dict:          # действие, см. schemas/action.schema.json
        return {"tool": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}}
```

```bash
python -m auditgym evaluate --split /path/hidden.json --policies mypkg.mod:MyPolicy oracle
python -m auditgym run --policy mypkg.mod:MyPolicy --seed 918273 --config '{"errors":{"duplicate_invoice":1}}'
```

Ваш oracle: класс с `grader_only = True` и конструктором `__init__(self, truth_provider)`.
Эталон он получит, только если вы явно разрешите это оценщику:
`--oracle-policies mypkg.mod:MyOracle`. Атрибут, объявленный самим адаптером, доверия не даёт.
Пример обоих видов: `examples/my_policy.py`. В `run` правило то же: `--oracle-policies`.

Поведение при сбоях политики: исключение, `SystemExit`, падение процесса или мусор вместо ответа
— неверная отправка (−1); зависание дольше лимита — процесс убивается, эпизод `truncated` с
пустым ответом; `print()` в политике уходит в stderr и протокол не ломает. Следующий эпизод
идёт с новым процессом политики, другие политики не затронуты.

## 3. Изоляция: оценщик в отдельном процессе

`evaluate` запускает три процесса: оценщик (`serve`), политику (`policy-worker`) и харнесс.

- Seed, config и секрет идут оценщику по анонимному каналу, итоги — обратно в харнесс по
  отдельному каналу. Временных файлов нет, в командной строке процессов закрытых данных нет.
- Харнесс и оценщик недампируемы (`PR_SET_DUMPABLE=0`): процесс политики не читает их
  `/proc/<pid>/environ`, память и дескрипторы. В окружении политики секрета нет.
- `reset` закрытого эпизода — только по индексу и только один раз; брошенный эпизод закрывается
  пустым ответом с учётом сделанных шагов.
- Политика в режиме eval получает награду 0 и пустой info.
- Хеш кода пакета `auditgym` проверяется после каждой политики; изменение кода останавливает
  прогон с ошибкой.
- Ограничение: политика работает под тем же пользователем ОС и может сигналом завершить процесс
  оценщика — это засчитывается ей как −1 на оставшихся эпизодах. Для полной изоляции ОС
  запускайте политику под отдельным пользователем или в отдельном контейнере.

Агент на любом языке может говорить с сервером напрямую, по JSON-строкам через stdin/stdout
(протокол описан в `auditgym/server.py`):

```bash
AUDITGYM_SECRET=... python -m auditgym serve --mode eval --episodes hidden.json --grader-log grader.jsonl
> {"op":"reset","index":0}
> {"op":"step","action":{"tool":"get_document","args":{"doc_id":"D-..."}}}
```

## 4. Свои преобразования файлов

Выгрузите эпизод, измените файлы (переставьте строки, переименуйте ID согласованно в открытых
файлах и `truth.json`) и подайте папку в среду:

```python
env = AuditEnv(mode="eval")
obs, info = env.reset(config={"episode_dir": "ep/public", "grader_dir": "ep/grader"})
```

Или: `python -m auditgym run --policy oracle --config '{"episode_dir":"...","grader_dir":"..."}'`.
Эпизод сплита с тем же `episode_id`, что у выгрузки: `python -m auditgym run --policy oracle --split dev --index 6`.

## 5. Отдельная проверка ответа

Оценить готовый `findings.json` по `truth.json`:

```bash
python -m auditgym grade --truth grader/truth.json --findings findings.json --steps 12
```

## 6. Независимая проверка данных

```bash
python3 independent_check/verify.py              # не импортирует пакет auditgym; все сплиты в episodes/
python3 independent_check/verify.py --root /tmp/x # ваш экспорт (export --splits ... --out /tmp/x)
python3 independent_check/selftest_mutations.py  # 19 видов порчи данных ловятся проверкой
```

## 7. Правила неверной структуры (награда −1, эпизод завершается)

Все пункты ниже дают −1:

- `findings` — не объект `{episode_id, findings}`, лишние ключи, чужой `episode_id`;
- `findings` не список или содержит больше 30 выводов;
- у вывода не ровно 6 ключей из `schemas/findings.schema.json`;
- `type` не из 4 типов;
- `claimed_amount_minor` не целое (bool тоже нельзя; `1000.0` считается целым, как в JSON Schema);
- `currency` не из 3 символов;
- `affected_ids` пустой, длиннее 30 или не список строк;
- `evidence_ids` длиннее 10 или не список строк;
- `explanation` длиннее 1000 символов.

## 8. Нормализация и допуски (код: `auditgym/matcher.py`)

- **Тип:** strip + lower. **ID и валюта:** strip + upper.
- **affected_ids** сравниваются как множество.
- **Сумма** сравнивается по модулю, допуск `max(1, ceil(0.0005 × expected))` минимальных единиц.
- **Доказательства:** множество `evidence_ids` должно содержать хотя бы один полный допустимый
  набор. Ключ курса в доказательствах записывается как `DATE:SRC:TGT`.
- **Неоднозначность:** максимальное паросочетание (алгоритм Куна); при равенстве выигрывает
  более ранний вывод. Повторы и несопоставленные выводы считаются FP.
