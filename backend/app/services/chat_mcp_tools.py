"""MCP-only tools: log a Claude research session as a DealCloud Activity.

DealCloud's "Activity" is the Interaction entity (entry type 5341, synced
read-only today by deal_cloud_enhancer's sync.py::sync_communications).
Writing one requires DealCloud credentials that only deal_cloud_enhancer
holds, so these tools call its `/internal/activities*` endpoints (same
shared-secret pattern as document_body.py's dce call for read_document).

Two-tool confirm flow, both taking the same ResearchActivityInput:
  * draft_research_activity  -- resolves org_ids/deal_id/requester_emails
    and returns the exact preview of what would be submitted. Zero
    DealCloud writes; call this (and re-call it as the user requests
    changes) before ever calling the tool below.
  * create_research_activity -- same resolution, but with confirm=true
    actually performs the DealCloud write. confirm=false (the default) is
    a no-op that just returns the same preview -- a safety net in case the
    model calls this before the user has approved the draft.

Deliberately NOT added to `chat_slack.tools.slack_registry` directly: that
registry is shared verbatim by Todd's Slack bot (chat_slack/orchestrator.py)
and the MCP server (mcp/server.py). Registering write tools there would
silently hand DealCloud-write ability to anyone who can DM Todd. Instead
`mcp_registry` is a clone with these two tools added on top, and only the
MCP server imports it.

## Writing tool descriptions: what may and may not go in one

Tool descriptions and input schemas are CACHED PER USER by claude.ai, and
there is no way for us to invalidate that cache. Each person has to use
Connectors -> ... -> "Refresh tools list" themselves, and until they do they
are steering on whatever text was current when they connected. The MCP
`tools.listChanged` notification does not help: our HTTP transport runs
`stateless=True` (no session to push down), `build_server()` snapshots the
tool list at process start, and the list only ever changes on redeploy --
which restarts the process. There is never a live session whose tool list
changed.

Tool RESPONSES are never cached. So:

  * Description = what the tool IS and when to reach for it. Stable facts
    only.
  * Response (the `note` field every tool here returns) = anything that can
    change: current policy, counts, "don't quote X yet".

Concretely, keep OUT of a description:
  - access/permission claims ("requires the same email", "will refuse
    another person's job", or the inverse "shared across the firm"). Be
    SILENT on policy -- silence cannot go stale, whereas any assertion can,
    in either direction.
  - specific magnitudes that are really server constants (batch sizes, SQL
    LIMITs, row counts, model names). State the RELATIONSHIP instead ("a
    capped handful -- use X for the full list"), which keeps the steering
    without a number that rots.

This is not hypothetical. A description promising "Requires the SAME
requested_by_email ... will refuse another person's job" outlived the gate
it described; the server had stopped refusing, but every user who had not
refreshed still had a tool that told the model to demand an email and expect
a refusal.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone

import psycopg2.extras
from pydantic import BaseModel, Field

from ..config import settings
from ..db import get_conn
from .chat_lib import ToolResult
from .chat_slack.tools import (
    slack_registry,
    _find_docs_by_path,
    _format_doc_matches,
)

MAX_NOTES_CHARS = 20_000


# ---------------------------------------------------------------------------
# dce internal API call
# ---------------------------------------------------------------------------

def _dce_post(path: str, payload: dict) -> dict:
    """POST to a deal_cloud_enhancer /internal/* endpoint. Returns the
    parsed JSON response, or a dict with ok=False + error on any failure
    (not configured, unreachable, non-2xx)."""
    if not settings.dce_internal_url or not settings.dce_internal_secret:
        return {"ok": False, "error": "dce_internal_not_configured"}

    url = f"{settings.dce_internal_url.rstrip('/')}/{path.lstrip('/')}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "X-Internal-Secret": settings.dce_internal_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": f"http_{e.code}"}
        body.setdefault("ok", False)
        return body
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": f"dce_unreachable: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Context resolution (org/deal come straight from the shared dealcloud
# schema; attendee identity requires a live dce/DealCloud lookup since
# Employee records aren't synced locally)
# ---------------------------------------------------------------------------

def _resolve_orgs(org_ids: list[int]) -> tuple[list[dict], list[str]]:
    if not org_ids:
        return [], []
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, name, dc_id FROM dealcloud.organization WHERE id = ANY(%s)",
            (org_ids,),
        )
        rows_by_id = {r["id"]: r for r in cur.fetchall()}

    resolved, warnings = [], []
    for org_id in org_ids:
        row = rows_by_id.get(org_id)
        if row is None:
            warnings.append(f"org_id {org_id} not found")
        elif row["dc_id"] is None:
            warnings.append(
                f"'{row['name']}' (org_id {org_id}) has no linked DealCloud "
                f"company -- dropped from Related Organizations"
            )
        else:
            resolved.append({"org_id": org_id, "name": row["name"], "dc_id": row["dc_id"]})
    return resolved, warnings


def _resolve_deal(deal_id: int) -> tuple[dict | None, str | None]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id, name, dc_id FROM dealcloud.deal WHERE id = %s", (deal_id,))
        row = cur.fetchone()
    if row is None:
        return None, f"deal_id {deal_id} not found"
    if row["dc_id"] is None:
        return None, f"'{row['name']}' (deal_id {deal_id}) has no linked DealCloud deal -- dropped from Deal"
    return {"deal_id": deal_id, "name": row["name"], "dc_id": row["dc_id"]}, None


def _resolve_attendees(emails: list[str]) -> tuple[list[dict], list[str]]:
    resolved, warnings = [], []
    for email in emails:
        resp = _dce_post("/internal/activities/resolve-attendee", {"email": email})
        if resp.get("ok"):
            resolved.append({"email": email, "name": resp.get("name")})
        else:
            warnings.append(
                f"no DealCloud Employee record for {email!r} ({resp.get('error', 'unknown')})"
            )
    return resolved, warnings


# ---------------------------------------------------------------------------
# Input model + shared preview builder
# ---------------------------------------------------------------------------

class ResearchActivityInput(BaseModel):
    org_ids: list[int] = Field(
        default_factory=list,
        description=(
            "dealcloud.organization.id(s) this activity is filed under -- "
            "becomes DealCloud's Related Organizations. Resolve company "
            "names to ids via find_organizations first."
        ),
    )
    deal_id: int | None = Field(
        None,
        description=(
            "dealcloud.deal.id this session relates to, if any. Resolve "
            "via list_deals first if the user only gave a company/deal name."
        ),
    )
    subject: str = Field(
        ...,
        min_length=1, max_length=200,
        description="Short subject line, e.g. 'Research session: Acme Corp -- market sizing'.",
    )
    notes: str = Field(
        ...,
        min_length=1,
        description="The conversation's summary/conclusions to log as the Activity's Notes.",
    )
    date: str | None = Field(
        None,
        description="ISO date (YYYY-MM-DD) this session happened. Defaults to today.",
    )
    requester_emails: list[str] = Field(
        default_factory=lambda: ["rom@ionpacific.com"],
        description=(
            "Ion Pacific email(s) of whoever ran this session -- credited "
            "as Internal Attendees. Defaults to the primary user; override "
            "if someone else asked for this."
        ),
    )


def _build_preview(inp: ResearchActivityInput) -> dict:
    orgs, warnings = _resolve_orgs(inp.org_ids)

    deal = None
    if inp.deal_id is not None:
        deal, deal_warning = _resolve_deal(inp.deal_id)
        if deal_warning:
            warnings.append(deal_warning)

    attendees, attendee_warnings = _resolve_attendees(inp.requester_emails)
    warnings.extend(attendee_warnings)

    notes = inp.notes
    truncated = len(notes) > MAX_NOTES_CHARS
    if truncated:
        notes = notes[:MAX_NOTES_CHARS]

    if not attendees:
        warnings.append(
            "no attendee resolved -- Internal Attendees is required on "
            "DealCloud's Interaction entity, so this cannot be created yet"
        )
    if not orgs:
        warnings.append("no organization resolved -- Related Organizations will be empty")

    return {
        "subject": inp.subject,
        "type": "Other",
        "type_note": (
            "DealCloud's Interaction.Type picklist has no \"Research "
            "Session\" value yet -- using \"Other\" until one is added in "
            "DealCloud Admin (Platform Builder)."
        ),
        "date": inp.date or datetime.now(timezone.utc).date().isoformat(),
        "notes": notes,
        "notes_truncated": truncated,
        "internal_attendees": attendees,
        "related_organizations": orgs,
        "deal": deal,
        "warnings": warnings,
        "ready_to_create": bool(attendees),
    }


# ---------------------------------------------------------------------------
# Registry: the Slack surface, with declared divergences, plus MCP-only tools
# ---------------------------------------------------------------------------
#
# PARITY IS THE GOAL. A tool built for Slack should show up in Claude too,
# with the same implementation. The only reason to diverge is that a tool
# genuinely needs a DIFFERENT implementation over MCP -- and the one real
# case is a tool that calls a model server-side, because an MCP caller IS a
# model and can do that itself (see MCP_REIMPLEMENTED below).
#
# So the surface is declared explicitly rather than inherited blindly: every
# Slack tool must be classified in exactly one bucket below, and
# _assert_surface_declared() fails the build if one isn't. That is the whole
# point -- previously this module did a bare slack_registry.clone(), so
# ask_data_room was exposed over MCP by accident and nobody noticed it needed
# an ANTHROPIC_API_KEY on a service that never had one. A loud build failure
# is the cheap version of that bug.
#
# Adding a Slack tool? Put its name in MCP_INHERITED. That is the default and
# almost always the right answer.

# Same implementation on both surfaces.
MCP_INHERITED: frozenset[str] = frozenset({
    "find_organizations",
    "find_documents_by_name",
    "find_in_documents",
    "find_comparable_orgs",
    "bundle_via_supersede",
    "get_org_portfolio_status",
    "get_org_deal_history",
    "get_org_ion_contacts",
    "get_org_their_contacts",
    "get_org_communication_timeline",
    "get_org_dossier",
    "search_email_threads",
    "read_email_thread",
    "find_email_contacts",
    "read_document_summary",
    "search_documents",
    "read_document",
    "get_deal_one_pager",
    "list_deals",
    "find_new_deals_to_discuss",
    "list_all_deals",
    "deal_underlying_companies",
    "get_fundraising_summary",
    "list_funds",
    "get_fund_status",
    "build_data_room",
    "check_data_room_build",
    "start_data_room_build_sweep",
    "check_data_room_build_sweep",
})

# Exposed, but with an MCP-specific implementation replacing the Slack one.
# Value is the reason, which belongs in code rather than a commit message.
MCP_REIMPLEMENTED: dict[str, str] = {
    "ask_data_room": (
        "The Slack version calls Claude server-side on an "
        "ANTHROPIC_API_KEY, because Slack has no model of its own. An MCP "
        "caller IS a model, so it retrieves and lets the caller answer -- "
        "cheaper, better grounded (the caller can chain read_document), "
        "billed to the caller's own session, and no API key needed on the "
        "MCP service at all."
    ),
}

# Deliberately NOT exposed over MCP. Withholding a tool breaks parity, so
# each one needs a real reason here, not just an omission.
MCP_WITHHELD: dict[str, str] = {
    "web_search": (
        "Same logic as ask_data_room's reimplementation, taken one step "
        "further: this tool's whole body is a server-side model call "
        "(Gemini + Google Search grounding), and an MCP caller IS a model "
        "with first-party web search of its own. Proxying its searches "
        "through our Gemini key would be slower, worse grounded, billed to "
        "us, and would need a GEMINI_API_KEY on ion-claude-mcp that the "
        "service does not have -- exactly the failure ask_data_room hit. "
        "There is no retrieval-only version to reimplement, so it is "
        "withheld rather than replaced."
    ),
    "research_company_web": (
        "Withheld for the same reason as web_search: it is a grounded "
        "Gemini call, and a claude.ai caller can research a company with "
        "its own web search. The value Todd gets from it -- a fixed "
        "section layout so the model doesn't improvise one per lookup -- "
        "is prompt guidance the MCP caller can be given directly."
    ),
    "read_uploaded_file": (
        "Reads a file somebody dropped into a Slack conversation with "
        "Todd, scoped to that conversation by the (team_id, channel_id) "
        "in ctx. An MCP caller has no Slack conversation, so ctx is empty "
        "and a lookup by file_name would fall back to matching across "
        "EVERY channel -- one user's DM upload surfacing in another "
        "user's claude.ai session. There is nothing to reimplement "
        "either: a claude.ai caller attaches files to its own context "
        "directly and never needs this."
    ),
    "list_uploaded_files": (
        "Same reason as read_uploaded_file: it enumerates one Slack "
        "channel's uploads and needs the conversation key from ctx, "
        "which an MCP caller doesn't have. Exposing it would list other "
        "people's attachments or nothing at all."
    ),
    "create_spreadsheet": (
        "Its whole output is a file UPLOADED INTO A SLACK CHANNEL, taken "
        "from ctx. An MCP caller has no channel, so the tool would build "
        "a workbook and have nowhere to put it. A claude.ai caller can "
        "also just write the file itself. If an MCP-side version is ever "
        "wanted it needs a different contract -- return the bytes, don't "
        "post them -- which is a reimplementation, not parity."
    ),
    "export_to_spreadsheet": (
        "Same reason as create_spreadsheet: it uploads to the Slack "
        "channel in ctx. The underlying data is already on the MCP "
        "surface through list_all_deals and list_funds, which an MCP "
        "caller can page and render however it likes, so nothing is "
        "actually withheld except the upload."
    ),
}


def _assert_surface_declared() -> None:
    """Fail loudly if the Slack surface and this declaration disagree.

    Two directions, both real failure modes:
      * an undeclared Slack tool -- someone added a tool and nobody decided
        whether it works as-is over MCP;
      * a declared name that no longer exists in slack_registry -- a rename
        would otherwise silently un-apply a reimplementation, quietly
        restoring the very server-side-model version we replaced.
    """
    declared = set(MCP_INHERITED) | set(MCP_REIMPLEMENTED) | set(MCP_WITHHELD)
    actual = set(slack_registry.names())
    if undeclared := actual - declared:
        raise RuntimeError(
            "Slack tool(s) not classified for the MCP surface: "
            f"{sorted(undeclared)}. Add each to MCP_INHERITED (the default "
            "-- parity), or to MCP_REIMPLEMENTED / MCP_WITHHELD with a "
            "reason. See chat_mcp_tools.MCP_INHERITED."
        )
    if stale := declared - actual:
        raise RuntimeError(
            f"MCP surface declares tool(s) that no longer exist in "
            f"slack_registry: {sorted(stale)}. They were probably renamed "
            "or removed -- update the declaration."
        )


_assert_surface_declared()

mcp_registry = slack_registry.clone()
for _name in MCP_WITHHELD:
    mcp_registry.remove(_name)
# Reimplementations are registered below; drop the Slack version first
# because ToolRegistry.register() refuses duplicate names.
for _name in MCP_REIMPLEMENTED:
    mcp_registry.remove(_name)


class McpAskDataRoomInput(BaseModel):
    job_id: int = Field(..., description="The job_id returned by build_data_room.")
    requested_by_email: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional. Ion Pacific email of the person asking, if you "
            "already know it from this conversation. Never guess it, and "
            "never ask the user for it just to make this call."
        ),
    )
    question: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "The question to retrieve relevant documents for. Optional "
            "when `path_or_name` is given on its own."
        ),
    )
    path_or_name: str | None = Field(
        default=None,
        max_length=400,
        description=(
            "Optional LITERAL filename/folder-path filter, applied before "
            "retrieval -- use it to confine a question to one part of the "
            "room ('VDR/LOIs - Contracts'). To simply FIND a document by "
            "name, or to establish whether one exists at all, call "
            "find_documents_by_name instead."
        ),
    )


@mcp_registry.tool(
    "ask_data_room",
    (
        "Retrieve the most relevant documents from a completed data-room "
        "build job so YOU can answer from them. Returns the top matches "
        "with their summaries, a source list for citations, and the "
        "grounding rules you must follow -- it does NOT return a "
        "pre-written answer; you write it. Only call this once "
        "check_data_room_build shows status='complete'.\n\n"
        "Answer ONLY from what this returns -- no outside knowledge. The "
        "summaries are TRUNCATED previews, so when one looks relevant but "
        "thin, call read_document on its document_id to read the full "
        "body before concluding anything. These are retrieval hits, not "
        "the whole room: never report a fact or document as absent just "
        "because it isn't here. To establish whether a document EXISTS, "
        "or to find one by filename or folder, call "
        "find_documents_by_name -- it scans every indexed path, so its "
        "empty result is definitive where this tool's is not. For an "
        "exhaustive per-document content pass, use "
        "start_data_room_build_sweep.\n\n"
        "Every document comes with a Link. Whenever you name a specific "
        "document, render it as a markdown link -- [document name](link) -- "
        "never a raw doc_id and never a bare URL. The returned instructions "
        "restate this; follow them."
    ),
    McpAskDataRoomInput,
    mutates_state=False,
)
def mcp_ask_data_room(inp: McpAskDataRoomInput, ctx: dict) -> ToolResult:
    """MCP reimplementation -- see MCP_REIMPLEMENTED['ask_data_room'].

    Mirrors the Slack tool exactly; only the answering step differs
    (retrieval is returned instead of a server-side Claude answer). There is
    no per-job ownership check: a room is per-FOLDER and shared, and the
    caller is already authenticated by the MCP transport. SharePoint remains
    the real access boundary -- dce does not model per-user document
    permissions.
    """
    from .data_room_build import DceUnavailable as _DceUnavailable, get_build_job
    from .claude_data_room import (
        ClaudeRoomError as _ClaudeRoomError,
        retrieve_room_context_for_docs,
    )
    try:
        job = get_build_job(inp.job_id)
    except ValueError as e:
        return ToolResult(output=str(e))
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if not job.doc_ids:
        return ToolResult(output=(
            f"job {inp.job_id} has no readable documents to search yet "
            f"(status={job.status}, {job.docs_processed}/{job.docs_total} "
            f"docs processed) -- call check_data_room_build again shortly."
        ))
    if not inp.question and not inp.path_or_name:
        return ToolResult(output=(
            "give either `question` (retrieve documents for it) or "
            "`path_or_name` (find documents by filename/folder)."
        ))

    doc_ids = job.doc_ids
    parts: list[str] = []
    # Mirrors the Slack tool: an incomplete room is still worth asking, and
    # the check-for-new-documents rule flips a finished room back to
    # 'pending' over a handful of added files.
    if job.status != "complete":
        behind = max(0, (job.docs_total or 0) - (job.docs_processed or 0))
        parts.append(
            f"[Note: this room is still being built (status={job.status}; "
            + (f"{job.docs_processed}/{job.docs_total} docs read, {behind} "
               f"outstanding" if behind else
               "newly-added documents are still being picked up")
            + "). Retrieval below covers only the documents read so far.]"
        )
    if inp.path_or_name:
        rows = _find_docs_by_path(inp.path_or_name, doc_ids=doc_ids)
        if not rows:
            return ToolResult(output=(
                f"No document in this room has {inp.path_or_name!r} in its "
                f"filename or folder path (a literal lookup over every "
                f"indexed path, so this is definitive for the room). Call "
                f"find_documents_by_name without a job_id to search the "
                f"whole document index."
            ))
        parts.append(_format_doc_matches(rows, inp.path_or_name))
        doc_ids = [r["id"] for r in rows]

    if inp.question:
        try:
            parts.append(retrieve_room_context_for_docs(doc_ids, inp.question))
        except _ClaudeRoomError as e:
            return ToolResult(output=f"Data room retrieval error: {e}")
    return ToolResult(output=("\n\n".join(p for p in parts if p)))


@mcp_registry.tool(
    "draft_research_activity",
    (
        "STEP 1 of 2 for logging a Claude research session as a DealCloud "
        "Activity (Interaction). Resolves org_ids / deal_id / "
        "requester_emails against DealCloud and returns the EXACT preview "
        "of what would be submitted -- Subject, Type, Date, Notes, "
        "Internal Attendees, Related Organizations, Deal -- WITHOUT "
        "writing anything. MANDATORY: immediately after every call to this "
        "tool (including re-calls with adjusted arguments), print the full "
        "preview -- every field above, including the complete Notes body "
        "verbatim -- as a formatted message in the chat, unprompted. Do "
        "this every single time a draft is produced, not only when the "
        "user asks to see it. Then let the user request changes (re-call "
        "this tool with adjusted arguments) before ever calling "
        "create_research_activity. Resolve org_ids via find_organizations "
        "and deal_id via list_deals first if the user only gave names."
    ),
    ResearchActivityInput,
)
def draft_research_activity(inp: ResearchActivityInput, ctx: dict) -> ToolResult:
    return ToolResult(output=_build_preview(inp))


class CreateResearchActivityInput(ResearchActivityInput):
    confirm: bool = Field(
        False,
        description=(
            "Must be explicitly set true to actually write to DealCloud. "
            "Only set this after the user has seen the draft_research_activity "
            "preview (or this same tool's own preview from a prior "
            "confirm=false call) and explicitly approved the values. "
            "confirm=false (the default) is a no-op: returns the same "
            "preview, writes nothing."
        ),
    )


@mcp_registry.tool(
    "create_research_activity",
    (
        "STEP 2 of 2: creates the DealCloud Activity (Interaction) for a "
        "Claude research session -- a REAL, visible write to production "
        "DealCloud that the whole firm can see. Do NOT call with "
        "confirm=true until the user has seen the preview from "
        "draft_research_activity and explicitly confirmed the org(s), "
        "deal, subject, notes, and attendee(s). confirm=false (or "
        "omitted) just returns the same preview and writes nothing -- use "
        "that if you need to double check the resolved values first. "
        "MANDATORY: whenever this tool is called with confirm=false (or "
        "confirm omitted), print the full returned preview -- every field, "
        "including the complete Notes body verbatim -- as a formatted "
        "message in the chat, unprompted, same as draft_research_activity. "
        "Every "
        "successful create is logged in DealCloud's audit trail "
        "(dealcloud.activity_log) and can be undone by an admin running a "
        "script -- if the user asks to undo/delete one, tell them to ask "
        "an admin to run manual_helper_scripts/undo_research_activity.py "
        "with the returned entry_id or log_id; there is no chat-facing "
        "undo tool by design."
    ),
    CreateResearchActivityInput,
    mutates_state=True,
)
def create_research_activity(inp: CreateResearchActivityInput, ctx: dict) -> ToolResult:
    preview = _build_preview(inp)

    if not inp.confirm:
        return ToolResult(output={**preview, "created": False, "reason": "confirm=false -- nothing written"})

    if not preview["internal_attendees"]:
        return ToolResult(output={**preview, "created": False, "error": "no attendee resolved -- refusing to create"})

    resp = _dce_post("/internal/activities", {
        "subject": preview["subject"],
        "notes": preview["notes"],
        "date": preview["date"],
        "attendee_emails": [a["email"] for a in preview["internal_attendees"]],
        "related_org_dc_ids": [o["dc_id"] for o in preview["related_organizations"]],
        "deal_dc_id": preview["deal"]["dc_id"] if preview["deal"] else None,
    })

    if resp.get("ok"):
        return ToolResult(output={
            **preview, "created": True,
            "entry_id": resp.get("entry_id"),
            "log_id": resp.get("log_id"),
        })
    return ToolResult(output={**preview, "created": False, "error": resp.get("error", "create_failed")})
