"""MM-REPORT-1: the vendored media.v1 proto is pinned by tag and sha-256, and
the stubs generated from it in the build (scripts/gen_spine_stubs.py) speak
the contract's service and messages."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from app.spine import contract

REPO = Path(__file__).resolve().parent.parent


def test_the_vendored_proto_is_the_tagged_file() -> None:
    data = (REPO / contract.PROTO_PATH).read_bytes()
    assert hashlib.sha256(data).hexdigest() == contract.PROTO_SHA256
    # SHOW1: 0.12.0, the lease's order, scope and cursor and four PendingImage
    # fields (fc-ingest-contract feat/media-lease-order-0120; the tag lands with
    # its merge). The commit is not repeated here: the squash merge re-pins it in
    # app/spine/contract.py alone, the bytes and their sha-256 unchanged.
    assert contract.PROTO_TAG == "v0.12.0"
    assert re.fullmatch(r"[0-9a-f]{40}", contract.PROTO_COMMIT)
    assert contract.verify_vendored_proto(REPO) == data


def test_a_changed_proto_is_refused(tmp_path: Path) -> None:
    target = tmp_path / contract.PROTO_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes((REPO / contract.PROTO_PATH).read_bytes() + b"\n")
    with pytest.raises(contract.ProtoMismatch, match="sha-256"):
        contract.verify_vendored_proto(tmp_path)


def test_a_missing_proto_is_refused(tmp_path: Path) -> None:
    with pytest.raises(contract.ProtoMismatch, match="missing"):
        contract.verify_vendored_proto(tmp_path)


def test_the_generated_stubs_name_the_contract_service() -> None:
    from app.spine.media_v1 import media_pb2, media_pb2_grpc

    service = media_pb2.DESCRIPTOR.services_by_name["SpineMedia"]
    assert service.full_name == "media.v1.SpineMedia"
    assert sorted(m.name for m in service.methods) == [
        "LeasePendingImages",
        "ReportDerivative",
        "ReportMask",
        "ReportTiers",
        "WithdrawMask",
    ]
    assert media_pb2.DESCRIPTOR.package == "media.v1"
    assert hasattr(media_pb2_grpc, "SpineMediaStub")
    report = media_pb2.DerivativeReport(product_image_id="x")
    assert not report.HasField("derivative_sha256")


def test_the_stubs_carry_the_0120_lease_order_scope_and_provenance() -> None:
    from app.spine.media_v1 import media_pb2

    request = media_pb2.LeasePendingImagesRequest(
        order=media_pb2.LEASE_ORDER_NEWEST_FIRST, scope=media_pb2.LEASE_SCOPE_SHOWABLE
    )
    assert request.order == 2 and request.scope == 2
    image = media_pb2.PendingImage(
        product_id="p", codec="identity", first_seen_at="2026-10-31T00:00:00.000000Z"
    )
    assert not image.HasField("origin_url")
    image.origin_url = "https://img.example.test/a.jpg"
    assert image.HasField("origin_url")


def test_the_stubs_carry_the_0120_lease_cursor() -> None:
    """LeasePendingImagesRequest.after = 7, a LeaseCursor of the two halves of
    the lease's sort key, both plain strings: presence is the message's."""
    from app.spine.media_v1 import media_pb2

    after = media_pb2.LeasePendingImagesRequest.DESCRIPTOR.fields_by_name["after"]
    assert after.number == 7 and after.message_type.full_name == "media.v1.LeaseCursor"
    cursor = media_pb2.LeaseCursor.DESCRIPTOR
    assert [(f.name, f.number) for f in cursor.fields] == [
        ("first_seen_at", 1),
        ("product_image_id", 2),
    ]
    request = media_pb2.LeasePendingImagesRequest()
    assert not request.HasField("after")
    request.after.SetInParent()
    assert request.HasField("after")  # present though empty: the spine refuses that
