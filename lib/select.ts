// Candidate ranking and the per-provider budget bookkeeping a failover needs.
//
// Nothing here reads the clock or a random source: the order of two candidates with the same rating is a
// function of the calls and the failures seen so far, so a replayed trial ranks them the same way.

export interface Candidate {
  provider: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  model: string;
  rating: number;
}

interface ProviderState {
  calls: number;
  failures: number;
  dead: boolean;
}

const state = new Map<string, ProviderState>();

function stateOf(provider: string): ProviderState {
  let s = state.get(provider);
  if (s === undefined) {
    s = { calls: 0, failures: 0, dead: false };
    state.set(provider, s);
  }
  return s;
}

/** One provider call was made: the tie-break counts it, so spreading calls keeps the count low. */
export function recordCall(provider: string): void {
  stateOf(provider).calls += 1;
}

/** A provider refused: a 402 ends it for the trial, a 429 or a 5xx only lowers its priority. */
export function recordFailure(provider: string): void {
  const s = stateOf(provider);
  s.failures += 1;
}

export function markDead(provider: string): void {
  stateOf(provider).dead = true;
}

/**
 * The candidates in the order a request should try them: the quality table's rating first (quality is 60%
 * of the score, so the best free model that meets the request always goes first), then the provider with
 * fewer failures, then the fewer calls, then the model id, so the order never depends on iteration order.
 */
export function rankCandidates(list: readonly Candidate[]): Candidate[] {
  const unique = new Map<string, Candidate>();
  for (const c of list) {
    const key = `${c.provider}\u0000${c.model}`;
    const prior = unique.get(key);
    if (prior === undefined || c.rating > prior.rating) unique.set(key, c);
  }
  const out = [...unique.values()];
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const sa = stateOf(a.provider);
    const sb = stateOf(b.provider);
    if (sa.dead !== sb.dead) return sa.dead ? 1 : -1;
    if (sa.failures !== sb.failures) return sa.failures - sb.failures;
    if (sa.calls !== sb.calls) return sa.calls - sb.calls;
    if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}
