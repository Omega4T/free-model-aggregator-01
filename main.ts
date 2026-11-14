// free-model-aggregator-01 entry: a quality-ranked, budget-aware free-model router.
//
// The server reads the trial's brief, probes every provider's own catalog route, registers one catalog,
// then serves the OpenAI Chat Completions API and the Anthropic Messages API. Each request is answered by
// the highest-rated FREE model that meets it, and a provider that refuses (402, 429, 5xx) is failed over
// to the next-best candidate instead of taking the request down.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import { rankCandidates, recordCall, recordFailure, type Candidate } from './lib/select.ts';
import { anthropicResponseToOpenAI, anthropicStreamToOpenAI, openAIRequestToAnthropic } from './lib/anthropic.ts';
import { geminiResponseToOpenAI, geminiStreamToOpenAI, openAIRequestToGemini } from './lib/gemini.ts';
import type { ProviderDirectoryEntry } from './lib/providers.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
// A request may fail over to the next-best candidate, but every upstream call spends budget, so the
// fan-out is capped: past this many attempts a client gets the last refusal instead of more spend.
const MAX_ATTEMPTS = 7;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// Every free model on an answered provider, with the quality table's rating where it has a row. A wire the
// request cannot travel on is left out: a JSON-mode request on the OpenAI wire needs a wire that carries
// response_format, so only an openai-wire provider may answer it.
function candidates(requires: Requirement, allowAnthropicWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({
        provider: probe.provider.id,
        wire: probe.provider.wire,
        model: model.id,
        rating: row === null ? 0 : row.rating_milli,
      });
    }
  }
  return rankCandidates(out).filter((c) => c.wire === 'openai' || (allowAnthropicWire && c.wire !== 'openai'));
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

function providerOf(id: string) {
  return brief.directory.providers.find((p) => p.id === id);
}

// One call to an openai-wire provider's Chat Completions route.
async function callOpenAI(providerId: string, body: Json): Promise<Response> {
  const provider = providerOf(providerId);
  if (provider === undefined) throw new Error(`unknown provider ${providerId}`);
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  recordCall(providerId);
  return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
}

// One call to an anthropic-wire provider's Messages route.
async function callAnthropic(provider: ProviderDirectoryEntry, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'anthropic-version': '2023-06-01',
    ...authHeaders(provider.auth, provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  recordCall(provider.id);
  return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(body) });
}

// One call to a gemini-wire provider, on the vendor's own generateContent paths.
async function callGemini(provider: ProviderDirectoryEntry, model: string, body: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  const route = stream ? `/v1beta/models/${model}:streamGenerateContent?alt=sse` : `/v1beta/models/${model}:generateContent`;
  recordCall(provider.id);
  return fetch(`${base}${route}`, { method: 'POST', headers, body: JSON.stringify(body) });
}

// A refusal a failover should answer with another model: 402, 429 or any 5xx.
function refuses(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

// POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that takes it, on
// either upstream wire.
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const jsonMode = isRecord(body.response_format) && body.response_format.type !== undefined;
  const list = candidates(requires, !jsonMode);
  if (list.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');

  let lastStatus = 0;
  let lastText = '';
  let attempts = 0;
  for (const candidate of list) {
    if (attempts >= MAX_ATTEMPTS) break;
    const provider = providerOf(candidate.provider);
    if (provider === undefined) continue;
    attempts += 1;
    let upstream: Response;
    try {
      if (candidate.wire === 'openai') {
        const outgoing: Json = { ...body, model: candidate.model };
        if (body.stream === true) outgoing.stream_options = { include_usage: true };
        upstream = await callOpenAI(candidate.provider, outgoing);
      } else if (candidate.wire === 'gemini') {
        upstream = await callGemini(provider, candidate.model, openAIRequestToGemini(body, candidate.model), body.stream === true);
      } else {
        upstream = await callAnthropic(provider, openAIRequestToAnthropic({ ...body, model: candidate.model }, candidate.model));
      }
    } catch (err) {
      recordFailure(candidate.provider);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (refuses(upstream.status)) {
      recordFailure(candidate.provider);
      lastStatus = upstream.status;
      lastText = (await upstream.text()).slice(0, 500);
      continue;
    }
    if (body.stream === true && upstream.ok && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (candidate.wire === 'openai') {
        await pipeStream(upstream.body, res);
      } else if (candidate.wire === 'gemini') {
        for (const frame of await geminiStreamToOpenAI(upstream.body, candidate.model)) res.write(frame);
        res.end();
      } else {
        for (const frame of await anthropicStreamToOpenAI(upstream.body, candidate.model)) res.write(frame);
        res.end();
      }
      return;
    }
    if (candidate.wire === 'openai') {
      const text = await upstream.text();
      res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return;
    }
    const answer: unknown = await upstream.json();
    if (candidate.wire === 'gemini') sendJson(res, 200, geminiResponseToOpenAI(isRecord(answer) ? answer : {}, candidate.model));
    else sendJson(res, 200, anthropicResponseToOpenAI(isRecord(answer) ? answer : {}, candidate.model));
    return;
  }
  sendError(res, 'openai', lastStatus === 0 ? 503 : lastStatus, 'upstream_error', lastText === '' ? 'every free candidate refused this request' : lastText);
}

// POST /v1/messages: an Anthropic-shaped client, rewritten to the best free openai-wire model and back.
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const list = candidates(requirementsFor(body), true).filter((c) => c.wire !== 'gemini');
  if (list.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');

  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  let lastStatus = 0;
  let lastText = '';
  let attempts = 0;
  for (const candidate of list) {
    if (attempts >= MAX_ATTEMPTS) break;
    attempts += 1;
    const provider = providerOf(candidate.provider);
    if (provider === undefined) continue;
    let upstream: Response;
    try {
      if (candidate.wire === 'anthropic') {
        upstream = await callAnthropic(provider, { ...body, model: candidate.model });
      } else {
        const upstreamBody = anthropicToOpenAI(body, candidate.model);
        if (body.stream === true) upstreamBody.stream = true;
        upstream = await callOpenAI(candidate.provider, upstreamBody);
      }
    } catch (err) {
      recordFailure(candidate.provider);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (refuses(upstream.status)) {
      recordFailure(candidate.provider);
      lastStatus = upstream.status;
      lastText = (await upstream.text()).slice(0, 500);
      continue;
    }
    if (body.stream === true && upstream.ok && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (candidate.wire === 'anthropic') await pipeStream(upstream.body, res);
      else {
        for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
        res.end();
      }
      return;
    }
    const answer: unknown = await upstream.json();
    if (candidate.wire === 'anthropic') sendJson(res, 200, isRecord(answer) ? answer : {});
    else sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
    return;
  }
  sendError(res, 'anthropic', lastStatus === 0 ? 503 : lastStatus, 'api_error', lastText === '' ? 'every free candidate refused this request' : lastText);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return models(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return chat(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return messages(req, res);
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}