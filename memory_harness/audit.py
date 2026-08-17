from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence

from memory_harness.contracts import AuditEvent


class JsonlAuditSink:
    def __init__(self, path: pathlib.Path | str) -> None:
        self.path = pathlib.Path(path)

    def write(self, events: Sequence[AuditEvent]) -> None:
        if not events:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(event.to_json(), sort_keys=True) + "\n")
