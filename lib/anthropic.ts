// The Anthropic-wire upstream: an OpenAI Chat Completions request on the way out, an Anthropic Messages
// answer on the way back, and the Messages event stream turned into Chat Completions chunks.
//
// Nothing here reads the clock or a random source, so the same request and the same provider answer give
// the same bytes on a replay.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function imageBlock(part: Json): Json | null {
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (match !== null) return { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  return null;
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input, as the wire allows.
    }
  }
  return {};
}

/** One OpenAI Chat Completions request as an Anthropic Messages request that names `model`. */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
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
      const text = textOf(content);
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }],
      });
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const fn = isRecord(call.function) ? call.function : {};
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${blocks.length}`,
          name: typeof fn.name === 'string' ? fn.name : '',
          input: argumentsOf(fn.arguments),
        });
      }
      if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    const blocks: Json[] = [];
    if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = imageBlock(part);
        if (block !== null) blocks.push(block);
      }
    }
    messages.push({ role: 'user', content: blocks.length === 0 ? [{ type: 'text', text: '' }] : blocks });
  }

  const out: Json = {
    model,
    messages,
    max_tokens: typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : typeof body.max_tokens === 'number' ? body.max_tokens : 4096,
  };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') out.stop_sequences = [stop];
  else if (Array.isArray(stop)) out.stop_sequences = stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools: Json[] = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      const fn = tool.function;
      if (typeof fn.name !== 'string') continue;
      const entry: Json = { name: fn.name, input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } };
      if (typeof fn.description === 'string') entry.description = fn.description;
      tools.push(entry);
    }
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'auto') out.tool_choice = { type: 'auto' };
  } else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

const STOP: Record<string, string> = {
  end_turn: 'stop',
  stop_sequence: 'stop',
  max_tokens: 'length',
  tool_use: 'tool_calls',
  refusal: 'content_filter',
};

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function anthropicResponseToOpenAI(answer: Json, model: string): Json {
  const raw = Array.isArray(answer.content) ? answer.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of raw) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : `call_${toolCalls.length}`,
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const message: Json = { role: 'assistant', content: text === '' && toolCalls.length > 0 ? null : text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: typeof answer.id === 'string' ? answer.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: STOP[String(answer.stop_reason ?? 'end_turn')] ?? 'stop' }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * An Anthropic Messages event stream as Chat Completions chunks: the role first, then one chunk per text or
 * tool-call delta, then the finish chunk, the usage chunk and [DONE], which is the order the OpenAI wire
 * sends them in.
 */
export async function anthropicStreamToOpenAI(stream: ReadableStream<Uint8Array>, model: string): Promise<string[]> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  const toolCalls: Json[] = [];
  let stop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;

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
      const type = event.type;
      if (type === 'message_start' && isRecord(event.message)) {
        const usage = isRecord(event.message.usage) ? event.message.usage : {};
        if (typeof usage.input_tokens === 'number') inputTokens = usage.input_tokens;
        if (typeof usage.output_tokens === 'number') outputTokens = usage.output_tokens;
      }
      if (type === 'content_block_start' && isRecord(event.content_block) && event.content_block.type === 'tool_use') {
        toolCalls.push({
          id: typeof event.content_block.id === 'string' ? event.content_block.id : `call_${toolCalls.length}`,
          type: 'function',
          function: { name: typeof event.content_block.name === 'string' ? event.content_block.name : '', arguments: '' },
        });
      }
      if (type === 'content_block_delta' && isRecord(event.delta)) {
        if (event.delta.type === 'text_delta' && typeof event.delta.text === 'string') text += event.delta.text;
        if (event.delta.type === 'input_json_delta' && typeof event.delta.partial_json === 'string' && toolCalls.length > 0) {
          const last = toolCalls[toolCalls.length - 1];
          const fn = isRecord(last.function) ? last.function : {};
          fn.arguments = `${typeof fn.arguments === 'string' ? fn.arguments : ''}${event.delta.partial_json}`;
          last.function = fn;
        }
      }
      if (type === 'message_delta') {
        if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') stop = STOP[event.delta.stop_reason] ?? 'stop';
        if (isRecord(event.usage) && typeof event.usage.output_tokens === 'number') outputTokens = event.usage.output_tokens;
      }
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
  chunks.push({ id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] });
  if (text !== '') chunks.push({ id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] });
  if (toolCalls.length > 0) {
    chunks.push({
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: toolCalls.map((call, index) => ({
              index,
              id: isRecord(call) ? call.id : `call_${index}`,
              type: 'function',
              function: isRecord(call) && isRecord(call.function) ? call.function : { name: '', arguments: '' },
            })),
          },
          finish_reason: null,
        },
      ],
    });
  }
  chunks.push({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: stop ?? (toolCalls.length > 0 ? 'tool_calls' : 'stop') }],
  });
  chunks.push({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
  const out = chunks.map((chunk) => sseEvent(null, chunk));
  out.push('data: [DONE]\n\n');
  return out;
}
