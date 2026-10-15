"""MM-REPORT-1: the vendored media.v1 proto is pinned by tag and sha-256, and
the stubs generated from it in the build (scripts/gen_spine_stubs.py) speak
the contract's service and messages."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from app.spine import contract

REPO = Path(__file__).resolve().parent.parent


def test_the_vendored_proto_is_the_tagged_file() -> None:
    data = (REPO / contract.PROTO_PATH).read_bytes()
    assert hashlib.sha256(data).hexdigest() == contract.PROTO_SHA256
    assert contract.PROTO_TAG == "v0.10.0"
    assert contract.PROTO_COMMIT == "e06b1f61f941c1ac65f9876064ac4d6a8f71993d"
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
        "WithdrawMask",
    ]
    assert media_pb2.DESCRIPTOR.package == "media.v1"
    assert hasattr(media_pb2_grpc, "SpineMediaStub")
    report = media_pb2.DerivativeReport(product_image_id="x")
    assert not report.HasField("derivative_sha256")
