// The Gemini-wire upstream: an OpenAI Chat Completions request as a generateContent request, the answer as
// a Chat Completions object, and the event stream as Chat Completions chunks.
//
// Pure functions of their arguments: no clock and no random source, so a replay answers the same bytes.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function partsOf(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const match = /^data:([^;,]+);base64,(.*)$/s.exec(block.image_url.url);
      if (match !== null) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
    }
  }
  if (parts.length === 0) parts.push({ text: '' });
  return parts;
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One OpenAI Chat Completions request as a Gemini generateContent request that names `model`. */
export function openAIRequestToGemini(body: Json, model: string): Json {
  void model;
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system' || role === 'developer') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      const name = typeof raw.name === 'string' ? raw.name : 'tool';
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: textOf(content) } } }] });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
      continue;
    }
    contents.push({ role: 'user', parts: partsOf(content) });
  }

  const generationConfig: Json = {};
  const max = typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  if (max !== undefined) generationConfig.maxOutputTokens = max;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') generationConfig.stopSequences = [stop];
  else if (Array.isArray(stop)) generationConfig.stopSequences = stop.filter((s): s is string => typeof s === 'string');
  if (isRecord(body.response_format) && body.response_format.type !== undefined) generationConfig.responseMimeType = 'application/json';

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const declarations: Json[] = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      const fn = tool.function;
      if (typeof fn.name !== 'string') continue;
      const entry: Json = { name: fn.name };
      if (typeof fn.description === 'string') entry.description = fn.description;
      if (isRecord(fn.parameters)) entry.parameters = fn.parameters;
      declarations.push(entry);
    }
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

const STOP: Record<string, string> = {
  STOP: 'stop',
  MAX_TOKENS: 'length',
  SAFETY: 'content_filter',
  RECITATION: 'content_filter',
  PROHIBITED_CONTENT: 'content_filter',
};

function partsOfAnswer(answer: Json): Json[] {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  return Array.isArray(content.parts) ? content.parts.filter(isRecord) : [];
}

function finishOf(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return STOP[String(first.finishReason ?? 'STOP')] ?? 'stop';
}

function usageOf(answer: Json): { prompt: number; completion: number } {
  const usage = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    prompt: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
    completion: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
  };
}

/** A generateContent answer as an OpenAI Chat Completions answer. */
export function geminiResponseToOpenAI(answer: Json, model: string): Json {
  const parts = partsOfAnswer(answer);
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (typeof part.text === 'string') text += part.text;
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
      });
    }
  }
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = usageOf(answer);
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: toolCalls.length > 0 ? 'tool_calls' : finishOf(answer) }],
    usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion },
  };
}

/** A Gemini event stream as Chat Completions chunks, in the OpenAI wire's order. */
export async function geminiStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): Promise<string[]> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  const toolCalls: Json[] = [];
  let stop = 'stop';
  let usage = { prompt: 0, completion: 0 };

  const takeFrame = (frame: string): void => {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let event: unknown;
      try {
        event = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(event)) continue;
      for (const part of partsOfAnswer(event)) {
        if (typeof part.text === 'string') text += part.text;
        if (isRecord(part.functionCall)) {
          toolCalls.push({
            id: `call_${toolCalls.length}`,
            type: 'function',
            function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
          });
        }
      }
      const fin = finishOf(event);
      if (fin !== 'stop' || toolCalls.length > 0) stop = toolCalls.length > 0 ? 'tool_calls' : fin;
      const u = usageOf(event);
      if (u.prompt > 0) usage.prompt = u.prompt;
      if (u.completion > 0) usage.completion = u.completion;
    }
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        takeFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') takeFrame(buffer);
  } finally {
    reader.releaseLock();
  }

  const chunks: Json[] = [];
  const base = { id: 'chatcmpl-gemini-stream', object: 'chat.completion.chunk', created: 1767225600, model };
  chunks.push({ ...base, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] });
  if (text !== '') chunks.push({ ...base, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] });
  if (toolCalls.length > 0) {
    chunks.push({
      ...base,
      choices: [
        {
          index: 0,
          delta: { tool_calls: toolCalls.map((call, index) => ({ index, id: isRecord(call) ? call.id : `call_${index}`, type: 'function', function: isRecord(call) && isRecord(call.function) ? call.function : { name: '', arguments: '' } })) },
          finish_reason: null,
        },
      ],
    });
  }
  chunks.push({ ...base, choices: [{ index: 0, delta: {}, finish_reason: toolCalls.length > 0 ? 'tool_calls' : stop }] });
  chunks.push({ ...base, choices: [], usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion } });
  const out = chunks.map((chunk) => sseEvent(null, chunk));
  out.push('data: [DONE]\n\n');
  return out;
}
