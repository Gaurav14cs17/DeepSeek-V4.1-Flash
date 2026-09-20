"""[DEFERRED] Deferred.

Stage gate: see docs/progress.md.
Do not implement until prior V4-Mini stages pass.
Paper notes: paper_notes/multimodal.md
"""

DEFERRED = True
STAGE_NAME = "vision_encoder"


def not_ready(msg: str | None = None) -> None:
    raise NotImplementedError(
        msg
        or f"{STAGE_NAME} is deferred until earlier research stages complete "
           f"(docs/progress.md)."
    )
