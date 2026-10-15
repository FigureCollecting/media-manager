"""media-manager's side of the spine's media.v1 SpineMedia contract (MM-REPORT-1).

``contract`` pins the vendored proto, ``field_rules`` applies the contract's
field rule before a call is made, ``media_client`` is the gRPC client, and
``replay`` reads the media database and reports it (``app.cli.mask_report``).
The stubs in ``media_v1`` are generated in the build (``scripts/gen_spine_stubs.py``)
from ``proto/media/v1/media.proto``, never committed.
"""
