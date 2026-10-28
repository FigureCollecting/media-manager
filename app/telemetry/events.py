"""A CLI's output as lines of the shape (LG-U11).

EventLines: a command that already printed one JSON object per event
(mm-report, mask_batch) prints one log line per event instead, ``event``
being ``app.<name>`` and the object's other fields the line's fields (a
nested one as one-line JSON). The scalar fields of the events named in
``summarize``, and their ``counts``, are what job.end carries.

report_lines: a command that printed one JSON document (the masks CLI)
prints its records one line each and the rest on job.end: a list of records
becomes one ``app.<key>`` line per record (a record that is not an object
under ``value``) and its count on job.end; a list of numbers stays one
value; an object holding such a list is read the same way, its keys joined
with ``_``; anything else goes on job.end as it is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .log import Logger
from .redact import to_snake_case

_SCALARS = (str, int, float, bool)


class EventLines:
    def __init__(self, log: Logger, summarize: Iterable[str] = ()) -> None:
        self.log = log
        self.summarize = frozenset(summarize)
        self.summary: dict[str, Any] = {}

    def __call__(self, payload: Mapping[str, Any]) -> None:
        fields = dict(payload)
        name = str(fields.pop("event", "line"))
        fields.pop("at", None)  # the line has its own time
        self.log.info("", event=f"app.{to_snake_case(name)}", **fields)
        if name not in self.summarize:
            return
        for key, value in fields.items():
            if key == "counts" and isinstance(value, Mapping):
                self.summary.update(value)
            elif value is None or isinstance(value, _SCALARS):
                self.summary[key] = value


def _records(value: Any) -> bool:
    return isinstance(value, list) and not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in value
    )


def _holds_records(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        _records(v) or _holds_records(v) for v in value.values()
    )


def report_lines(log: Logger, report: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Write ``report``'s records as lines; return what job.end carries."""
    summary: dict[str, Any] = {}
    for key, value in report.items():
        name = to_snake_case(f"{prefix}{key}")
        if _records(value):
            for record in value:
                fields = dict(record) if isinstance(record, Mapping) else {"value": record}
                log.info("", event=f"app.{name}", **fields)
            summary[name] = len(value)
        elif _holds_records(value):
            summary.update(report_lines(log, value, f"{name}_"))
        else:
            summary[name] = value
    return summary
