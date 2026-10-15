"""media.v1 messages built from the media DB's records, every field through
the contract's field rule (app.spine.field_rules).

A builder raises ``FieldRefused`` for a value the spine would refuse where
the contract gives no way to leave it out (an identity, a required
grounding); an optional value the spine would refuse is left out and named
in ``Built.warnings``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from . import field_rules as fr
from .media_v1 import media_pb2

M = TypeVar("M")

#: DerivativeReport.width / height: a value outside this range is unknown.
_DIMENSION = range(1, 2**31)


@dataclass(frozen=True)
class CopyRecord:
    """A clean copy as the media DB records it (a ``sanitized`` public object
    and the appearance its mask rows measured WITHOUT a mask)."""

    sha256: str
    mime: str
    width: int
    height: int
    generated_at: dt.datetime
    thumbhash: str | None = None
    dominant_color: str | None = None
    #: Grounding on the copy's own alpha channel: a transparent copy only.
    own_grounding: Mapping[str, float | None] | None = None


@dataclass(frozen=True)
class MaskRecord:
    """A mask the media DB publishes over a clean copy, at one pipeline version."""

    clean_sha256: str
    pipeline_version: str
    mask_sha256: str
    mime: str
    generated_at: dt.datetime
    grounding: Mapping[str, float | None]
    thumbhash: str | None = None
    dominant_color: str | None = None


@dataclass
class Built(Generic[M]):
    message: M
    warnings: list[str] = field(default_factory=list)


def _grounding(values: Mapping[str, float | None]) -> media_pb2.Grounding:
    return media_pb2.Grounding(**fr.grounding(values))


def derivative_report(
    product_image_id: str, copy: CopyRecord, *, pipeline_version: str
) -> Built[media_pb2.DerivativeReport]:
    """A DERIVED report of ``copy`` for the depiction ``product_image_id``."""
    warnings: list[str] = []
    message = media_pb2.DerivativeReport(
        product_image_id=fr.uuid_identity("product_image_id", product_image_id),
        derivative_sha256=fr.sha_bytes("derivative_sha256", copy.sha256),
        mime=fr.derivative_mime(copy.mime),
        derive_kind="sanitized",
        pipeline_version=fr.identity("pipeline_version", pipeline_version),
        generated_at=fr.identity("generated_at", fr.instant(copy.generated_at)),
        outcome=media_pb2.DERIVE_OUTCOME_DERIVED,
    )
    if copy.width in _DIMENSION:
        message.width = copy.width
    if copy.height in _DIMENSION:
        message.height = copy.height
    thumb, why = fr.thumbhash(copy.thumbhash)
    warnings += why
    if thumb is not None:
        message.thumbhash = thumb
    color, why = fr.dominant_color(copy.dominant_color)
    warnings += why
    if color is not None:
        message.dominant_color = color
    if copy.own_grounding is not None:
        try:
            message.grounding.CopyFrom(_grounding(copy.own_grounding))
        except fr.FieldRefused as exc:
            warnings.append(f"own grounding dropped: {exc}")
    return Built(message, warnings)


def failed_report(
    product_image_id: str,
    *,
    pipeline_version: str,
    generated_at: dt.datetime,
    failure_reason: str,
) -> Built[media_pb2.DerivativeReport]:
    """A FAILED report: no derivative fields, the reason cleaned and cut to 1 KB."""
    reason, why = fr.failure_reason(failure_reason)
    message = media_pb2.DerivativeReport(
        product_image_id=fr.uuid_identity("product_image_id", product_image_id),
        derive_kind="sanitized",
        pipeline_version=fr.identity("pipeline_version", pipeline_version),
        generated_at=fr.identity("generated_at", fr.instant(generated_at)),
        outcome=media_pb2.DERIVE_OUTCOME_FAILED,
        failure_reason=reason,
    )
    return Built(message, [f"failure_reason: {w}" for w in why])


def mask_report(mask: MaskRecord) -> Built[media_pb2.MaskReport]:
    """A ReportMask of a published mask, with its grounding and its appearance
    with the mask applied."""
    warnings: list[str] = []
    message = media_pb2.MaskReport(
        derivative_sha256=fr.sha_bytes("derivative_sha256", mask.clean_sha256),
        pipeline_version=fr.identity("pipeline_version", mask.pipeline_version),
        mask_sha256=fr.sha_bytes("mask_sha256", mask.mask_sha256),
        mime=fr.mask_mime(mask.mime),
        generated_at=fr.identity("generated_at", fr.instant(mask.generated_at)),
        grounding=_grounding(mask.grounding),
    )
    thumb, why = fr.thumbhash(mask.thumbhash)
    warnings += why
    if thumb is not None:
        message.thumbhash = thumb
    color, why = fr.dominant_color(mask.dominant_color)
    warnings += why
    if color is not None:
        message.dominant_color = color
    return Built(message, warnings)


def withdrawal(
    mask_sha256: str, *, reason: str | None, withdrawn_at: dt.datetime
) -> Built[media_pb2.MaskWithdrawal]:
    """WithdrawMask, restore=false: the mask is withdrawn, why, and since when."""
    text, why = fr.withdraw_reason(reason)
    message = media_pb2.MaskWithdrawal(
        mask_sha256=fr.sha_bytes("mask_sha256", mask_sha256),
        reason=text,
        withdrawn_at=fr.identity("withdrawn_at", fr.instant(withdrawn_at)),
        restore=False,
    )
    return Built(message, [f"reason: {w}" for w in why])


def restore(mask_sha256: str) -> media_pb2.MaskWithdrawal:
    """WithdrawMask, restore=true: the mask is published again."""
    return media_pb2.MaskWithdrawal(
        mask_sha256=fr.sha_bytes("mask_sha256", mask_sha256), restore=True
    )
