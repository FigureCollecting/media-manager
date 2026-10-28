"""media-manager's ONE seam onto the estate's logging and trace shape (LG-U11).

Python port of the SHAPE of fc-shared 1.8.0's node-only server subpaths, as
fc-aggregation's src/telemetry.ts uses them (LG-ADOPT-1):

  log         the fixed-key JSON line on stdout, redaction (redact), and
              Python's logging routed into it (configure).
  tracing     a real TracerProvider with NO exporter: spans record with real
              ids, so every line carries trace_id, and nothing leaves the
              process; W3C traceparent helpers; the propagation allowlist;
              run_job, the job model: a run is a root span with
              job.start/job.end, each item of it a new root trace linked
              to the run (fc-shared server/job), every line with run_id.
  http        one http.in line per request (ASGI), continuing the caller's
              traceparent.
  grpc_client a CLIENT span per call and its traceparent on the wire, for
              cluster-internal targets only.
  celery      the worker's logging, and a trace per task continuing the
              enqueuer's.
"""

from .log import Logger, configure, process_logger, service_name, use_logger
from .tracing import NULL_ITEM, ItemScope, JobRun, current_traceparent, run_job, start_tracing

__all__ = [
    "NULL_ITEM",
    "ItemScope",
    "JobRun",
    "Logger",
    "configure",
    "current_traceparent",
    "process_logger",
    "run_job",
    "service_name",
    "start_tracing",
    "use_logger",
]
