"""[DEFERRED] Deferred.

Stage gate: see docs/progress.md.
Do not implement until prior V4-Mini stages pass.
Paper notes: paper_notes/ced.md
"""

DEFERRED = True
STAGE_NAME = "ced"


def not_ready(msg: str | None = None) -> None:
    raise NotImplementedError(
        msg
        or f"{STAGE_NAME} is deferred until earlier research stages complete "
           f"(docs/progress.md)."
    )
