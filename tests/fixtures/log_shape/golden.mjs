// Regenerates fc-shared-1.8.0-golden.jsonl: the lines fc-shared 1.8.0's own
// server logger writes for the cases tests/test_telemetry_log.py replays.
//   FC_SHARED_NODE_MODULES=<a node_modules holding @figurecollecting/fc-shared 1.8.0,
//   @opentelemetry/api and @opentelemetry/context-async-hooks> node golden.mjs > fc-shared-1.8.0-golden.jsonl
const NM = process.env.FC_SHARED_NODE_MODULES;
if (!NM) throw new Error('set FC_SHARED_NODE_MODULES');
const { createLogger } = await import(`${NM}/@figurecollecting/fc-shared/dist/server/log.mjs`);
const api = await import(`${NM}/@opentelemetry/api/build/src/index.js`);
const { AsyncLocalStorageContextManager } = await import(`${NM}/@opentelemetry/context-async-hooks/build/src/index.js`);
const { context, trace, ROOT_CONTEXT } = api.default ?? api;
context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());

const TRACE_ID = '4bf92f3577b34da6a3ce929d0e0e4736';
const SPAN_ID = '00f067aa0ba902b7';
const inSpan = (fn) =>
  context.with(trace.setSpan(ROOT_CONTEXT, trace.wrapSpanContext({ traceId: TRACE_ID, spanId: SPAN_ID, traceFlags: 1, isRemote: true })), fn);

const out = [];
const log = createLogger({ service: 'media-api', version: '0.1.0', env: {}, sink: (line) => out.push(line) });
const jobLog = createLogger({ service: 'mm-report', version: '0.1.0', env: { JOB_NAME: 'mm-report-29312345' }, sink: (line) => out.push(line) });

inSpan(() => log.info({ event: 'rpc.in', call: 'media.v1.SpineMedia/ReportDerivative', code: 'ok', duration_ms: 41.23,
  peer: 'media-report.media.serviceaccount.identity.linkerd.app.mesh.estate', site: 'orzgk', claims: 23 }, ''));
inSpan(() => jobLog.info({ event: 'job.end', code: 'ok', duration_ms: 671000, stores: 27, enqueued: 412 }, 'pass finished'));
log.info({ site: 'orzgk' });
inSpan(() => log.error({ err: new TypeError('boom') }, 'failed hard'));
inSpan(() => log.warn({ imageUrl: 'https://u:p@cdn.example/a.jpg?sig=1#f', password: 'x', note: 'use Bearer abc.def now',
  detail: { a: 1, token: 't', nested: { href: 'https://h.example/x?y=1' } }, itemId: 7, ratio: 0.5, ok: true, gone: null },
  'see https://x.example/p?token=1 now'));
inSpan(() => log.info({ event: 'http.in', call: 'GET /d/{sha256}', code: '200', duration_ms: 1.25, peer: 'unmeshed' }, ''));
inSpan(() => log.warn({ event: 'http.in', call: 'GET <unmatched>', code: 404, duration_ms: 0.04, peer: 'invalid', aborted: false }, ''));
log.info({ event: 'app.log', form: 'user=a&password=b&code=c', header: 'Cookie: a=b', json: '{"token":"t","n":1}', HTTPStatus: 'x', '9lives': 1 }, 'multi\nline');
for (const line of out) console.log(line);
