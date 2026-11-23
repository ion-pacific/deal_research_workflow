import { authHeaders } from "./auth";

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "";

/** Thin fetch wrapper that injects the user's credentials (see lib/auth.ts). */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  for (const [k, v] of Object.entries(await authHeaders())) headers.set(k, v);
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${body || path}`);
  }
  return res.json() as Promise<T>;
}

// ----- types (mirror the FastAPI Pydantic shapes) -----
export type Phase =
  | "org_select"
  | "entity_select"
  | "data_room_setup"
  | "data_room_view";

export interface Session {
  id: string;
  originator_email: string;
  title: string | null;
  current_version_id: string;
  redo_version_id: string | null;
  forked_from_version_id: string | null;
  created_at: string;
  updated_at: string;
  is_starred: boolean;
  // TRUE means the title should not be auto-renamed (set on manual
  // edit AND after the first-org-selection auto-rename). The header
  // doesn't surface this directly but the SessionsListPage uses it
  // along with is_starred to render appropriate badges.
  title_is_locked: boolean;
}

export interface Version {
  id: string;
  session_id: string;
  parent_id: string | null;
  undo_unit_id: string;
  phase: Phase;
  state: Record<string, unknown>;
  source: string;
  ai_message_id: string | null;
  summary: string | null;
  created_at: string;
}

export interface SessionWithCurrent {
  session: Session;
  current_version: Version;
}

export interface OrgContact {
  email: string;
  name: string | null;
}

export interface OrgContactRecent {
  // Shape from dealcloud.org_ion_contacts.top_contacts /
  // dealcloud.org_their_contacts.top_contacts.
  ion_email?: string | null;
  ion_name?: string | null;
  email?: string | null;
  name?: string | null;
  active_touches?: number;
  passive_touches?: number;
  last_touch?: string | null;
}

export interface OrgDossier {
  org_id: number;
  name: string;
  dc_id?: string | null;
  org_type?: string | null;
  description?: string | null;
  parent?: { org_id: number; name: string } | null;
  fundraising_status?: string | null;
  investor_status?: string | null;
  counts: {
    documents: number;
    email_threads: number;
    calendar_events: number;
    slack_groups: number;
    communications: number;
  };
  latest_update_at?: string | null;
  main_contact?: OrgContact | null;
  main_ion_contact?: OrgContact | null;
  recent_documents: Array<Record<string, unknown>>;
  recent_email_threads: Array<Record<string, unknown>>;
  recent_calendar_events: Array<Record<string, unknown>>;
  recent_slack_groups: Array<Record<string, unknown>>;
  deal_stats: {
    assessed: boolean;
    deals_total: number;
    by_status: Record<string, number>;
    as_counterparty_count: number;
    as_underlying_count: number;
  };
  top_ion_contacts: OrgContactRecent[];
  top_their_contacts: OrgContactRecent[];
}

export type SearchMode = "trigram" | "semantic" | "hybrid";

export interface OrgSearchResult {
  org_id: number;
  name: string;
  // score / why_match are present on /orgs/search hits but not on
  // /orgs/by-ids batch fetches.
  score: number | null;
  why_match: string | null;
  sample_evidence: string[];
  // enriched fields from dealcloud.organization_summary; refreshed
  // nightly. Counts default to 0 when the org isn't in the summary
  // table; the contact fields are nullable.
  document_count: number;
  communication_count: number;
  latest_update_at: string | null;
  main_contact: OrgContact | null;
  main_ion_contact: OrgContact | null;
}

// ----- entities (phase 2) -----

export type EntityType =
  | "document"
  | "email_thread"
  | "calendar_event"
  | "slack_message_group";

export const ENTITY_TYPES: EntityType[] = [
  "document",
  "email_thread",
  "calendar_event",
  "slack_message_group",
];

export interface EntityFilter {
  // ISO timestamps. Empty string / null = no constraint.
  date_from?: string | null;
  date_to?: string | null;
  contains?: string | null;
}

export interface EntityCountResp {
  entity_type: EntityType;
  count: number;
}

export interface EntityListResp {
  entity_type: EntityType;
  count: number;
  rows: Array<Record<string, unknown>>;
  limit: number;
  offset: number;
}

export interface EntityOrgContextRow {
  org_id: number;
  org_name: string;
  alias_text: string | null;
  relationship_type: string | null;
  context: string | null;
  is_confirmed: boolean;
  match_method: string | null;
  confidence: number | null;
  model: string | null;
  notes: string | null;
}

export interface EntityOrgContextResp {
  entity_type: EntityType;
  entity_id: number;
  rows: EntityOrgContextRow[];
}

function entityFilterToQuery(
  filter: EntityFilter,
  extra: Record<string, string | number> = {},
): string {
  const params = new URLSearchParams();
  if (filter.date_from) params.set("date_from", filter.date_from);
  if (filter.date_to) params.set("date_to", filter.date_to);
  if (filter.contains && filter.contains.trim()) {
    params.set("contains", filter.contains.trim());
  }
  for (const [k, v] of Object.entries(extra)) {
    params.set(k, String(v));
  }
  const s = params.toString();
  return s ? `?${s}` : "";
}

// ----- data rooms (phase 3) -----

export interface PresetQuestion {
  id: number;
  label: string;
  question_text: string;
  sort_order: number | null;
  grouping: string | null;
  // Email of the user who authored this row. NULL on the seed
  // 'default'-grouping rows. Used to gate edit/delete in the UI --
  // you can only edit your own customs.
  originator?: string | null;
}

export interface BuildDataRoomResp {
  data_room_id: number;
  name: string;
  entity_count: number;
  question_count: number;
  new_version_id: string;
  created_at: string;
}

export interface PresetAnswer {
  answer_id: number | null;
  provider: "toltiq" | "claude";
  answer_status: string;
  answer_text: string | null;
  attachments: unknown;
  answer_error: string | null;
  answer_completed_at: string | null;
}

export interface PresetQA {
  preset_question_id: number;
  sort_order: number | null;
  label: string;
  question_text: string;
  answers: PresetAnswer[];
}

export interface FollowupQA {
  answer_id: number;
  question_text: string;
  status: string;
  answer_text: string | null;
  attachments: unknown;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
  // 'toltiq' (default) | 'claude'. Lets the UI render which provider
  // answered, important during the A/B period.
  provider: "toltiq" | "claude";
}

export interface DataRoomDetail {
  id: number;
  name: string;
  main_organization_id: number;
  status: string;
  toltiq_deal_id: string | null;
  // 'toltiq' (default + back-compat) | 'claude' | 'both'
  provider: "toltiq" | "claude" | "both";
  filters_applied: Record<string, unknown> | null;
  error_message: string | null;
  originator: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  entity_progress: Record<string, number>;
  preset_questions: PresetQA[];
  followup_questions: FollowupQA[];
}

// ----- data room coverage (see memory: data_room_coverage_analysis) -----

export interface CoverageHit {
  doc_name: string;
  present: "yes" | "partial";
  evidence: string;
  // Files in the room that are this same document under a different
  // version or in another folder (draft, for-execution, fully-executed,
  // closing-binder copy). 1 when there is only one.
  copies?: number;
}

export interface CoverageReview {
  status: "confirmed_gap" | "dismissed";
  note: string | null;
  reviewed_by: string;
  reviewed_at: string;
}

export interface CoverageCriterion {
  criterion_id: number;
  category: string;
  criterion: string;
  // deal.transaction_type values this criterion applies to, e.g.
  // ["LP Deal", "GP Deal"]. Was a single "LP" | "GP" | "both" tag
  // before the 2026-09-02 taxonomy migration.
  applies_to: string[];
  importance: "core" | "common" | "occasional" | null;
  // 'Found' | 'Found — high hit count, review before trusting' |
  // 'Found (keyword only — not LLM-confirmed, needs review)' |
  // 'Unconfirmed' | 'Candidate Gap' | 'Scanning'
  status: string;
  hits: CoverageHit[];
  // Matching files before same-document copies were collapsed.
  hit_total?: number;
  keyword_hits: string[];
  // Human-review-gate state (step 10a) -- null means never reviewed, NOT
  // "no gap". Only meaningful for status === 'Candidate Gap' today, but
  // present regardless of status.
  review: CoverageReview | null;
}

export interface RoomCoverage {
  room_id: number;
  indexing_state: {
    total: number;
    by_state: Record<string, number>;
    unconfirmed_docs: number;
  };
  criteria: CoverageCriterion[];
}

// ----- deal one-pagers -----

// Typed shape of the portfolio_relationship section's `content`. The
// web view special-cases this section to render as a banner above the
// rest (see DealOnePagerPage.tsx).

// One fund holding a position. A single deal can be held by multiple
// funds -- typically the deal-specific SPV plus the parent blind-pool
// fund it rolls up into. `fund_type` is the raw DC value ('Single Deal',
// 'Blind-Pool', 'SMA', ...); `fund_type_label` is the UI-friendly form
// ('SPV', 'Blind Pool', 'SMA', 'Other').
export interface PortfolioFund {
  fund_id: number;
  fund_name: string | null;
  fund_type: string | null;
  fund_type_label: string;
}

export interface PortfolioDirectPosition {
  deal_id: number;
  deal_name: string;
  deal_status: string;
  is_co_invest: boolean;
  funds: PortfolioFund[];
  invested_capital: number | null;
  deal_size: number | null;
  fair_value: number | null;
  realized_capital: number | null;
  total_value_to_invested: number | null;
}

export interface PortfolioIndirectPosition {
  deal_id: number;
  deal_name: string;
  deal_status: string;
  via_org_id: number;
  via_org_name: string | null;
  ion_fund_id: number | null;
  ion_fund_name: string | null;
}

export interface PortfolioRelationshipContent {
  in_portfolio: boolean;
  direct_positions: PortfolioDirectPosition[];
  indirect_positions: PortfolioIndirectPosition[];
  total_invested_direct: number;
  total_fair_value_direct: number;
  summary: string;
}

// ----- investors section + flagged-investor watchlist -----

export interface InvestorSource {
  title: string | null;
  url: string;
}

export interface InvestorRound {
  round: string | null;
  amount: string | null;
  date: string | null;
  lead: boolean | null;
  sources: InvestorSource[];
}

export interface InvestorEntry {
  name: string;
  rounds: InvestorRound[];
  // Present when a user manually marked this investor as top-tier (as
  // of the one-pager's last build) -- the live view reconciles this
  // against GET /api/v1/investors/marks so a mark/unmark click is
  // reflected without a rebuild.
  marked_by?: string | null;
}

export interface InvestorsContent {
  company: string | null;
  top_tier: InvestorEntry[];
  all_known_investors: string[];
  public_error: string | null;
}

// The GLOBAL investor-marks registry (not deal-scoped) -- see
// deal_cloud_enhancer's dealcloud.flagged_investor / one_pager_sections/
// investors.py. is_active=true means manually marked top-tier;
// is_active=false means manually unmarked/excluded, even if the LLM
// itself classifies the investor top-tier.
export interface InvestorMark {
  name: string;
  normalized_name: string;
  flagged_by: string;
  is_active: boolean;
  created_at: string | null;
}

export interface InvestorMarksResp {
  investors: InvestorMark[];
}

export interface DealListItem {
  deal_id: number;
  name: string;
  status: string;
  company: string | null;
  has_one_pager: boolean;
  one_pager_status: string | null;
  generated_at: string | null;
}

export interface OnePagerSection {
  section_key: string;
  title: string;
  status: string;
  content: unknown | null;
  content_markdown: string;
}

export interface OnePager {
  one_pager_id: number;
  status: string;
  generated_at: string | null;
  sections: OnePagerSection[];
}

export interface BuildState {
  // 'idle' | 'running' | 'stale' (a 'running' row older than the
  // ~10-min stale window, i.e. the dce build dyno likely died).
  state: "idle" | "running" | "stale";
  running_pager_id: number | null;
  started_at: string | null;
}

export interface DealInfo {
  deal_id: number;
  name: string;
  status: string;
  transaction_type: string | null;
  company: string | null;
}

export interface DealOnePagerResp {
  deal: DealInfo;
  one_pager: OnePager | null;
  build: BuildState;
}

export interface BuildOnePagerResp {
  deal_id: number;
  building: boolean;
  already_running: boolean;
}

// ----- manual DealCloud sync -----

export interface DealcloudSyncStatus {
  // 'idle' (no sync since the dce dyno last started) | 'running' |
  // 'complete' | 'failed' | 'stale' (a 'running' state older than dce's
  // stale window -- the sync likely died mid-run).
  status: "idle" | "running" | "complete" | "failed" | "stale";
  full: boolean;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export interface TriggerSyncResp {
  ok: boolean;
  already_running: boolean;
  status: DealcloudSyncStatus;
}

// ----- chat -----

export interface ChatMessage {
  id: string;
  session_id: string;
  phase: string;
  role: "user" | "assistant" | "tool";
  content: Record<string, unknown>;
  pre_version_id: string | null;
  post_version_id: string | null;
  parent_message_id: string | null;
  model_id: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  latency_ms: number | null;
  error: string | null;
  created_at: string;
}

// SSE events the backend emits. See backend/app/services/chat_research/
// orchestrator.py for the producer side.
export type ChatEvent =
  | { type: "turn_start"; session_id: string; phase: string;
      user_message_id: string; current_version_id: string; undo_unit_id: string }
  | { type: "assistant_start" }
  | { type: "text_delta"; text: string }
  | { type: "thinking_delta"; text: string }
  | { type: "assistant_message"; message_id: string; stop_reason: string;
      ai_message_id?: string;
      content: Array<Record<string, unknown>>;
      usage: { input_tokens: number; output_tokens: number;
        cache_creation_input_tokens: number; cache_read_input_tokens: number } }
  | { type: "tool_call"; tool_use_id: string; name: string;
      input: Record<string, unknown>; assistant_message_id: string }
  | { type: "tool_result"; tool_use_id: string; name: string;
      output: string; is_error: boolean; tool_message_id?: string;
      mutates_state?: boolean }
  | { type: "version_created"; version_id: string; phase: string;
      summary: string }
  | { type: "turn_complete"; stop_reason: string }
  | { type: "turn_done"; session_id: string; current_version_id: string }
  | { type: "turn_failed"; reason: string; last_stop_reason?: string }
  | { type: "error"; message: string };

// ----- endpoints -----
export const api = {
  whoami: () => request<{ email: string }>("/api/v1/me"),

  listSessions: () => request<Session[]>("/api/v1/sessions"),
  createSession: () =>
    request<SessionWithCurrent>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    }),
  getSession: (id: string) =>
    request<SessionWithCurrent>(`/api/v1/sessions/${id}`),
  updateSession: (
    id: string,
    body: { title?: string; is_starred?: boolean },
  ) =>
    request<Session>(`/api/v1/sessions/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  appendVersion: (
    sessionId: string,
    body: {
      parent_id: string;
      phase: Phase;
      state: Record<string, unknown>;
      summary?: string;
    }
  ) =>
    request<{ version: Version; session: Session }>(
      `/api/v1/sessions/${sessionId}/versions`,
      { method: "POST", body: JSON.stringify(body) }
    ),

  // mode: 'trigram' (default; backwards compat) | 'semantic' (cosine
  // over embeddings) | 'hybrid' (RRF over both legs). The FE's search
  // box exposes a small toggle so the user can compare; AI chat tools
  // hit the route with mode='hybrid' regardless.
  searchOrgs: (q: string, limit = 10, mode: SearchMode = "trigram") =>
    request<OrgSearchResult[]>(
      `/api/v1/orgs/search?q=${encodeURIComponent(q)}&limit=${limit}&mode=${mode}`
    ),

  getOrgsByIds: (ids: number[]) =>
    request<OrgSearchResult[]>(
      `/api/v1/orgs/by-ids?ids=${ids.join(",")}`
    ),

  // Rich dossier for the expand panel: counts per entity type, recent
  // items per channel, deal stats, top-5 Ion + counterpart contacts.
  // Lazy-fetched (only when a card is expanded).
  getOrgDossier: (orgId: number) =>
    request<OrgDossier>(`/api/v1/orgs/${orgId}/dossier`),

  countEntities: (
    sessionId: string,
    entityType: EntityType,
    filter: EntityFilter,
  ) => {
    const qs = entityFilterToQuery(filter);
    return request<EntityCountResp>(
      `/api/v1/sessions/${sessionId}/entities/${entityType}/count${qs}`,
    );
  },

  listEntities: (
    sessionId: string,
    entityType: EntityType,
    filter: EntityFilter,
    limit: number,
    offset: number,
  ) => {
    const qs = entityFilterToQuery(filter, { limit, offset });
    return request<EntityListResp>(
      `/api/v1/sessions/${sessionId}/entities/${entityType}/list${qs}`,
    );
  },

  // Why this entity is linked to the session's selected orgs. Lazy-
  // fetched only when an entity row is expanded.
  getEntityOrgContext: (
    sessionId: string,
    entityType: EntityType,
    entityId: number,
  ) =>
    request<EntityOrgContextResp>(
      `/api/v1/sessions/${sessionId}/entities/${entityType}/${entityId}/org-context`,
    ),

  // ---- phase 3: data-rooms ----
  getPresetQuestions: () =>
    request<PresetQuestion[]>("/api/v1/data-rooms/preset-questions"),

  // Hydrate question rows by id (used to render custom questions
  // already in the session's preset_question_ids -- the defaults
  // endpoint won't include them).
  getPresetQuestionsByIds: (ids: number[]) =>
    request<PresetQuestion[]>(
      `/api/v1/data-rooms/preset-questions/by-ids?ids=${ids.join(",")}`,
    ),

  // Create a custom question. Used by both add (label+text new) and
  // edit (label+text replacing an existing row) flows -- in the edit
  // case the caller is then expected to swap the new id into
  // preset_question_ids in a session_version append.
  createPresetQuestion: (label: string, question_text: string) =>
    request<PresetQuestion>("/api/v1/data-rooms/preset-questions", {
      method: "POST",
      body: JSON.stringify({ label, question_text }),
    }),

  buildDataRoom: (
    sessionId: string,
    body?: { provider?: "toltiq" | "claude" | "both" },
  ) =>
    request<BuildDataRoomResp>(
      `/api/v1/sessions/${sessionId}/data-rooms`,
      {
        method: "POST",
        body: JSON.stringify(body ?? { provider: "toltiq" }),
      },
    ),

  // Phase 4 view: status + entity progress + preset Q&A + followups.
  // Polled while the room is non-terminal so the FE shows the build
  // advancing in real time.
  getDataRoom: (roomId: number) =>
    request<DataRoomDetail>(`/api/v1/data-rooms/${roomId}`),

  // Direct ToltIQ passthrough. Returns 202 immediately with the new
  // answer_id; the actual ToltIQ workflow runs in a background task.
  // Caller should refetch getDataRoom shortly after to see the row
  // transition running -> complete.
  askDataRoom: (roomId: number, question: string) =>
    request<{ answer_id: number; status: string }>(
      `/api/v1/data-rooms/${roomId}/ask`,
      { method: "POST", body: JSON.stringify({ question }) },
    ),

  // Parallel Claude path. SYNCHRONOUS: blocks ~3-8s while Claude
  // answers from our pgvector retrieval over the room's docs.
  // Returns the full answer text + metadata. Used for the A/B
  // comparison against ToltIQ.
  askDataRoomClaude: (roomId: number, question: string) =>
    request<{
      answer_id: number;
      answer_text: string;
      retrieved_doc_ids: number[];
      status: string;
      model: string | null;
      latency_s: number | null;
      tokens: { input: number; output: number; cache_read: number } | null;
    }>(
      `/api/v1/data-rooms/${roomId}/ask-claude`,
      { method: "POST", body: JSON.stringify({ question }) },
    ),

  // Read-only checklist coverage. Safe to call anytime, including
  // mid-scan (unscanned criteria come back status='Scanning' rather than
  // a false verdict). Logic lives entirely in deal_cloud_enhancer.
  getDataRoomCoverage: (roomId: number, dealType?: string | string[]) =>
    request<RoomCoverage>(
      `/api/v1/data-rooms/${roomId}/coverage` +
        (dealType
          ? `?deal_type=${encodeURIComponent(
              Array.isArray(dealType) ? dealType.join(",") : dealType,
            )}`
          : ""),
    ),

  // Processes one bounded batch of the room's pending scan work and returns
  // immediately. Poll this (on a visible "Scanning…" UI state, not
  // silently) until remaining reaches 0, then refetch
  // getDataRoomCoverage.
  //
  // One phase per call, in order: "reading_files" opens documents nobody
  // has read yet (~10/call), then "classifying" runs the checklist
  // (~25/call). The remaining<=0 stop condition covers whichever phase is
  // active, so polling logic needs no change -- `phase` is for labelling
  // progress honestly, since a classify count sits at 0 while files are
  // still being read.
  scanDataRoomCoverageBatch: (roomId: number) =>
    request<{
      room_id: number;
      facets_written: number;
      docs_processed: number;
      remaining: number;
      phase: "reading_files" | "classifying";
      docs_read: number;
      docs_failed: number;
    }>(`/api/v1/data-rooms/${roomId}/coverage/scan-batch`, { method: "POST" }),

  // Human-review-gate action: confirm a Candidate Gap as real, or dismiss
  // it as not applicable to this deal. Append-only on the backend --
  // reviewing the same criterion again records a NEW decision, not an
  // overwrite. reviewed_by is always the server's authenticated user, never
  // sent from here.
  setDataRoomCoverageReview: (
    roomId: number,
    criterionId: number,
    status: "confirmed_gap" | "dismissed",
    note?: string,
  ) =>
    request<CoverageReview>(`/api/v1/data-rooms/${roomId}/coverage/review`, {
      method: "POST",
      body: JSON.stringify({ criterion_id: criterionId, status, note: note ?? null }),
    }),

  // Re-runs a failed ToltIQ answer in place (same row, status reset to
  // 'running'). Backed by POST /data-rooms/{id}/answers/{id}/retry on
  // the API; works for both preset and ad-hoc follow-up answers.
  retryDataRoomAnswer: (roomId: number, answerId: number) =>
    request<{ answer_id: number; status: string }>(
      `/api/v1/data-rooms/${roomId}/answers/${answerId}/retry`,
      { method: "POST" },
    ),

  // Re-claims a failed data-room build. Resets status to 'pending';
  // the data-room-builder cron picks it up on its next 2-min tick.
  // Idempotent: already-uploaded entities skip, existing toltiq_deal_id
  // is reused.
  retryDataRoomBuild: (roomId: number) =>
    request<{ data_room_id: number; status: string }>(
      `/api/v1/data-rooms/${roomId}/retry`,
      { method: "POST" },
    ),

  // Add the OTHER provider's answers to a room that was built single-
  // provider. Backend flips room.provider to 'both' and either spawns
  // the Claude BackgroundTask (when adding claude) or resets status to
  // 'pending' so the cron picks it up (when adding toltiq). Existing
  // answers from the original provider are preserved.
  addDataRoomProvider: (roomId: number, provider: "toltiq" | "claude") =>
    request<{ room_id: number; provider: string; status: string }>(
      `/api/v1/data-rooms/${roomId}/add-provider`,
      { method: "POST", body: JSON.stringify({ provider }) },
    ),

  listMessages: (sessionId: string, limit = 200) =>
    request<ChatMessage[]>(
      `/api/v1/sessions/${sessionId}/messages?limit=${limit}`
    ),

  // ---- deal one-pagers ----
  // No q -> the live-pipeline deals (the weekly-baked set). With q ->
  // search any deal by name or company.
  listDeals: (q?: string, limit = 25) => {
    const params = new URLSearchParams();
    if (q && q.trim()) params.set("q", q.trim());
    params.set("limit", String(limit));
    return request<DealListItem[]>(`/api/v1/deals?${params.toString()}`);
  },

  // Latest complete/partial one-pager + current build state. Polled
  // while build.state === 'running'.
  getDealOnePager: (dealId: number) =>
    request<DealOnePagerResp>(`/api/v1/deals/${dealId}/one-pager`),

  // Trigger a (re)build in dce. Returns 202; poll getDealOnePager to
  // see the new row land. Idempotent unless force (a build already
  // running for the deal is reused).
  buildDealOnePager: (dealId: number, force = false) =>
    request<BuildOnePagerResp>(`/api/v1/deals/${dealId}/one-pager/build`, {
      method: "POST",
      body: JSON.stringify({ force }),
    }),

  // ---- investor marks (global top-tier override registry) ----
  // ALL marks (both directions). Used to render Mark/Unmark dropdown
  // state and to reclassify a one-pager's Top-tier Investors list
  // client-side, live, without needing a rebuild.
  listInvestorMarks: () =>
    request<InvestorMarksResp>("/api/v1/investors/marks"),

  markInvestor: (name: string) =>
    request<{ investor: InvestorMark }>("/api/v1/investors/mark", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  unmarkInvestor: (name: string) =>
    request<{ investor: InvestorMark }>("/api/v1/investors/unmark", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  // ---- manual DealCloud sync ----
  // Current state of the manual sync trigger. Polled while status ===
  // 'running'.
  getDealcloudSyncStatus: () =>
    request<DealcloudSyncStatus>("/api/v1/dealcloud-sync/status"),

  // Trigger a sync in dce. Returns 202; poll getDealcloudSyncStatus to see
  // it land. Idempotent unless force (a sync already running is reused).
  triggerDealcloudSync: (full = false, force = false) =>
    request<TriggerSyncResp>("/api/v1/dealcloud-sync", {
      method: "POST",
      body: JSON.stringify({ full, force }),
    }),
};

// -- SSE chat stream --------------------------------------------------------

// Browser EventSource can only do GET and can't set request headers, so
// it's a non-starter for our X-User-Email-authenticated POST. Roll a
// thin SSE parser over fetch + ReadableStream instead.
//
// Frame format (matches the backend's _sse_format):
//   event: <type>\ndata: <json>\n\n
//
// Caller passes onEvent for typed dispatch. The returned promise resolves
// when the stream closes cleanly; reject on network/parse errors. Pass an
// AbortSignal to cancel mid-stream (the backend's StreamingResponse
// observes the disconnect and tears the loop down via the orchestrator's
// finally block).

export interface ChatStreamRequest {
  sessionId: string;
  phase: Phase;
  message: string;
  parentId?: string;
  signal?: AbortSignal;
  // Per-turn snapshot of what the user is currently looking at (e.g.
  // for org_select: search query + displayed candidates). Forwarded
  // to the orchestrator and rendered as a system block. Skip for
  // phases that don't have a useful snapshot.
  uiContext?: Record<string, unknown> | null;
  // Per-message toggle: when true, the orchestrator registers the
  // `web_search` tool (Gemini-grounded) for this turn. Default false
  // (data-room-only). Frontend keeps the last choice sticky in the
  // chat store so flipping it sticks until the user flips back.
  webSearchEnabled?: boolean;
  // Phase 4 A/B knob. 'both' (default) exposes ask_toltiq AND
  // ask_claude_room; 'claude' strips ask_toltiq; 'toltiq' strips
  // ask_claude_room. UI shows the toggle only on data_room_view
  // rooms whose room.provider is 'both'.
  chatProviderMode?: "both" | "claude" | "toltiq";
  onEvent: (ev: ChatEvent) => void;
}

export async function streamChat(req: ChatStreamRequest): Promise<void> {
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  headers.set("Accept", "text/event-stream");
  for (const [k, v] of Object.entries(await authHeaders())) headers.set(k, v);

  const res = await fetch(`${BASE}/api/v1/sessions/${req.sessionId}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      phase: req.phase,
      message: req.message,
      parent_id: req.parentId,
      ui_context: req.uiContext ?? null,
      web_search_enabled: req.webSearchEnabled ?? false,
      chat_provider_mode: req.chatProviderMode ?? "both",
    }),
    signal: req.signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${body}`);
  }
  if (!res.body) {
    throw new Error("Response has no body — SSE not supported by this fetch?");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // Each SSE frame is delimited by a blank line. Walk forward,
      // dispatching every complete frame and keeping the trailing
      // partial in the buffer for the next iteration.
      let sep: number;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        dispatchFrame(frame, req.onEvent);
      }
    }
    // Flush any trailing frame (shouldn't normally happen since the
    // backend always ends frames with \n\n, but be defensive).
    if (buf.trim()) dispatchFrame(buf, req.onEvent);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // Best-effort; aborted reads can throw on releaseLock.
    }
  }
}

function dispatchFrame(frame: string, onEvent: (ev: ChatEvent) => void): void {
  // Read both the `event:` line and the `data:` line. The SSE spec
  // makes `event:` the authoritative type; we use it as a fallback
  // when the JSON body lacks one. Earlier orchestrator code emitted
  // turn_start / turn_done with payloads that didn't include `type`,
  // and the frontend's switch silently no-op'd, leaving the input
  // disabled forever. This makes either source of truth work.
  let dataStr = "";
  let eventType = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("data:")) dataStr += line.slice(5).trim();
    else if (line.startsWith("event:")) eventType = line.slice(6).trim();
  }
  if (!dataStr) return;
  try {
    const parsed = JSON.parse(dataStr) as Partial<ChatEvent> & {
      type?: string;
    };
    if (!parsed.type && eventType) {
      (parsed as { type: string }).type = eventType;
    }
    onEvent(parsed as ChatEvent);
  } catch (err) {
    console.warn("SSE: malformed JSON in data:", dataStr, err);
  }
}
