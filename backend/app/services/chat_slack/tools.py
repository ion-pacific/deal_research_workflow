"""Tool registry for Todd's Slack conversational engine.

All tools are read-only -- the conversational flow doesn't mutate
session state (Slack doesn't have a research.session). The 5 SQL
dossier functions live in `dealcloud.*` and are wrapped one-tool-each
so the model can pick the cheapest answer for the question instead of
fetching all five every turn.

Reuses (don't reinvent):
  * `org_search.search_organizations`           -- trigram + prefix + exact
  * `org_dossier.get_org_dossier`               -- single-org rich snapshot
  * `dealcloud.bundle_via_supersede(int[])`     -- canonical heads
  * 5 SQL functions in `dealcloud.*`            -- the Q1-Q5 dossier
"""
from __future__ import annotations

import html
import re
from datetime import date
from typing import Any, Literal

import psycopg2.extras
from pydantic import BaseModel, Field

from ..chat_lib import (ToolRegistry, ToolResult,
                        register_company_web_tools, register_web_tools)
from ..data_room_build import (REASON_OVER_CAP, REASON_STALE_SKIP,
                               REASON_UNREADABLE, REASON_UNSUPPORTED,
                               describe_unread_doc, partition_unread_docs)
from ..org_dossier import get_org_dossier as _get_org_dossier
from ..org_search import find_comparable_organizations, search_organizations
from .deals_tracker import compute_new_deals_to_discuss, TrackerError
from ...db import get_conn


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class FindOrganizationsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Either an exact company name/alias OR a free-text description "
            "of what kind of company you're looking for (sector, business "
            "model, geography). Examples: 'Snyk', 'short-term rental "
            "marketplace', 'Singapore family office investing in AI'. "
            "Runs hybrid trigram + semantic-embedding search, so concept "
            "queries that don't name a specific company still work."
        ),
        min_length=1,
        max_length=200,
    )
    limit: int = Field(
        10, description="Max results to return.", ge=1, le=25
    )


class FindComparableOrgsInput(BaseModel):
    org_id: int | None = Field(
        None,
        description=(
            "dealcloud.organization.id of the company to find comps FOR. "
            "Reuses that company's stored business embedding (preferred -- "
            "use find_organizations first to resolve a name to an id)."
        ),
    )
    description: str | None = Field(
        None,
        description=(
            "Alternative to org_id: a free-text business description to find "
            "comps for (e.g. 'short-term rental marketplace for urban "
            "apartments'). Use when the company isn't in our database yet. "
            "Provide org_id OR description -- org_id wins if both are given."
        ),
        max_length=2000,
    )
    limit: int = Field(10, description="Max comps to return.", ge=1, le=25)
    require_internal_data: bool = Field(
        True,
        description=(
            "When true (default) only return comps we hold internal material "
            "on (>=1 document or communication). Set false to widen to any "
            "linked org."
        ),
    )


class OrgIdsInput(BaseModel):
    org_ids: list[int] = Field(
        ...,
        description=(
            "List of dealcloud.organization.id values. Pass canonical heads "
            "(use bundle_via_supersede first if you only have arbitrary ids)."
        ),
        min_length=1,
        max_length=20,
    )


class OrgIdInput(BaseModel):
    org_id: int = Field(..., description="dealcloud.organization.id")


class DocumentIdInput(BaseModel):
    document_id: int = Field(..., description="dealcloud.document.id")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

slack_registry = ToolRegistry()

# Appended to every tool description that hands back documents. Without it
# the model cites documents as "doc_id=43012" or a bare filename -- the
# doc_id convention exists only because the drw FRONTEND rewrites
# `[doc_id=N]` markers into clickable chips, which no chat surface does.
# Markdown is the single format asked for: Claude.ai renders it natively, and
# chat_slack/orchestrator.py now runs model prose through _md_to_slack, which
# turns `[label](url)` into Slack's `<url|label>`.
_LINK_RULE = (
    " CITING: whenever you name a specific document to the user, render it "
    "as a markdown link to its web_url -- [document name](web_url). Never "
    "show a raw document_id and never paste a bare URL; both are noise to "
    "the reader. If a document has no web_url, give its path instead and "
    "say it isn't linkable -- never guess a URL."
)



@slack_registry.tool(
    "find_organizations",
    (
        "Search the deal cloud organization database by company name OR by "
        "a description of the kind of company you want. Runs HYBRID search: "
        "an exact/trigram leg over names+aliases AND a semantic-embedding "
        "leg over each org's business description, fused together. So this "
        "does two jobs: (1) look up a company the user names, and "
        "(2) discover companies by meaning/sector -- e.g. "
        "'short-term rental marketplaces' or 'companies like Airbnb' returns "
        "businesses in that space even when the user names none of them. "
        "ALWAYS use this when the user mentions a company by name OR asks to "
        "find companies matching a theme/sector -- don't guess from prior "
        "knowledge. (To find companies similar to one we already have, "
        "prefer `find_comparable_orgs`.)"
    ),
    FindOrganizationsInput,
)
def find_organizations(inp: FindOrganizationsInput, ctx: dict) -> ToolResult:
    # Same rationale as chat_research/tools.py: hybrid for descriptive
    # query support, with automatic trigram fallback if the semantic
    # leg fails.
    rows = search_organizations(inp.query, inp.limit, mode="hybrid")
    return ToolResult(output={"query": inp.query, "results": rows})


@slack_registry.tool(
    "find_comparable_orgs",
    (
        "Find companies in the deal cloud whose BUSINESS is most similar to "
        "a given company (comparable companies / 'comps'). Seeds from an "
        "existing company by `org_id` (reusing its business embedding) or "
        "from a free-text `description`. Use this for 'find comps for X', "
        "'what companies like X do we have', or 'do we have internal data on "
        "anyone in this space'. By default returns only comps we actually "
        "hold material on (>=1 document or communication), each with its "
        "document/communication counts and main contacts -- so you can see "
        "at a glance what internal data exists. Read-only. Prefer this over "
        "find_organizations when the user already has a reference company "
        "and wants similar ones."
    ),
    FindComparableOrgsInput,
)
def find_comparable_orgs(inp: FindComparableOrgsInput, ctx: dict) -> ToolResult:
    if inp.org_id is None and not (inp.description and inp.description.strip()):
        return ToolResult(output={
            "error": "provide either org_id or description",
            "results": [],
        })
    rows = find_comparable_organizations(
        seed_org_id=inp.org_id,
        query_text=inp.description,
        limit=inp.limit,
        require_internal_data=inp.require_internal_data,
    )
    return ToolResult(output={
        "seed_org_id": inp.org_id,
        "seed_description": inp.description if inp.org_id is None else None,
        "require_internal_data": inp.require_internal_data,
        "results": rows,
    })


@slack_registry.tool(
    "bundle_via_supersede",
    (
        "Collapse a set of org_ids to canonical heads via the "
        "dealcloud.bundle_via_supersede SQL function. Use this whenever "
        "you need to pass org_ids to one of the get_org_* tools, to make "
        "sure superseded orgs are walked to their canonical replacements. "
        "Returns the deduplicated list of canonical org_ids."
    ),
    OrgIdsInput,
)
def bundle_via_supersede(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT dealcloud.bundle_via_supersede(%s::int[]) AS ids",
            (inp.org_ids,),
        )
        row = cur.fetchone()
    canonical = list((row or {}).get("ids") or [])
    return ToolResult(output={"input_ids": inp.org_ids, "canonical_ids": canonical})


@slack_registry.tool(
    "get_org_portfolio_status",
    (
        "Q1: Is this org currently in our portfolio? Returns "
        "`in_portfolio` (bool), `as_counterparty` (deals where the org is "
        "the deal's main counterparty), `as_underlying` (deals where the "
        "org is an underlying portfolio company, including LLM-derived "
        "links from IC memos), and `doc_only_underlying_hints` (looser "
        "signal: docs co-mentioning the org with a known DC counterparty). "
        "Restricted to deal.status IN ('Portfolio Company', 'Partnership')."
    ),
    OrgIdsInput,
)
def get_org_portfolio_status(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_portfolio_status", inp.org_ids)


@slack_registry.tool(
    "get_org_deal_history",
    (
        "Q2: Have we ever assessed this org -- portfolio, pipeline, "
        "passed/dead, or anything else in DealCloud? Like "
        "get_org_portfolio_status but without the status filter, so it "
        "includes failed/dropped/pipeline deals too. Returns `assessed`, "
        "`deals_total`, `by_status`, `as_counterparty`, `as_underlying`."
    ),
    OrgIdsInput,
)
def get_org_deal_history(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_deal_history", inp.org_ids)


@slack_registry.tool(
    "get_org_ion_contacts",
    (
        "Q3: Who at Ion Pacific has worked with this org? The most-active "
        "Ion people by activity (active vs passive touches across email / "
        "calendar / DC communications), plus last-touch-by-channel. "
        "Slack is rolled into Q5's totals but not split per Ion employee."
    ),
    OrgIdsInput,
)
def get_org_ion_contacts(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_ion_contacts", inp.org_ids)


@slack_registry.tool(
    "get_org_their_contacts",
    (
        "Q4: Who at the org have we engaged? The most-engaged external "
        "contacts "
        "ranked by domain-match boost (their email is on a domain we "
        "associate with the org) then by activity. Includes whether each "
        "is in DealCloud and the org's known email domains."
    ),
    OrgIdsInput,
)
def get_org_their_contacts(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_their_contacts", inp.org_ids)


@slack_registry.tool(
    "get_org_communication_timeline",
    (
        "Q5: When have we engaged this org? Returns first/last touch, "
        "duration, total touches, per-channel breakdown (email / slack / "
        "calendar / DC communications / documents), and activity by "
        "quarter. Counts and dates only -- for what was said on logged "
        "calls/meetings use search_dealcloud_activities."
    ),
    OrgIdsInput,
)
def get_org_communication_timeline(inp: OrgIdsInput, ctx: dict) -> ToolResult:
    return _call_dossier_fn("org_communication_timeline", inp.org_ids)


@slack_registry.tool(
    "get_org_dossier",
    (
        "Compact rich snapshot for ONE org: identity, total entity "
        "counts, main contacts, and a capped handful of the most recent "
        "documents, email threads, calendar events and slack groups, plus "
        "aggregate deal stats. The per-section caps are small by design -- "
        "use list_org_recent_documents for a company's full document list "
        "rather than assuming the dossier showed everything. Use when "
        "the user wants a quick \"what's this org\" answer or to compare "
        "similarly-named candidates by recent activity. ~2-3 KB."
    ),
    OrgIdInput,
)
def get_org_dossier(inp: OrgIdInput, ctx: dict) -> ToolResult:
    try:
        return ToolResult(output=_get_org_dossier(inp.org_id))
    except ValueError as e:
        return ToolResult(output=str(e))


@slack_registry.tool(
    "read_document_summary",
    (
        "Read the LLM-generated summary of one document -- a short paragraph, "
        "not the document body. Use when the user asks what a specific "
        "document is about or wants context on a doc you found via "
        "get_org_dossier (which lists recent doc ids and names). "
        "If the summary doesn't answer the question, say so -- do NOT "
        "speculate about the doc's full contents."
        + _LINK_RULE
    ),
    DocumentIdInput,
)
def read_document_summary(inp: DocumentIdInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, name, path, modified_at, web_url, summary
              FROM dealcloud.document
             WHERE id = %s
            """,
            (inp.document_id,),
        )
        row = cur.fetchone()
    if row is None:
        return ToolResult(output=f"Document {inp.document_id} not found.")
    return ToolResult(
        output={
            "id":          row["id"],
            "name":        row["name"],
            "path":        row["path"],
            "modified_at": row["modified_at"],
            "web_url":     row["web_url"],
            "summary":     row["summary"] or "(no summary)",
        }
    )


class SearchDocumentsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "What to find docs about. Topic phrase ('GP commitment'), "
            "doc type ('limited partnership agreement', 'IC memo'), "
            "or partial filename. Uses hybrid retrieval (filename "
            "trigram + embedding cosine on doc name+summary)."
        ),
        min_length=1, max_length=200,
    )
    org_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Restrict to docs attributed to these orgs (via "
            "dealcloud.organization_entity). Pass the canonical org_id "
            "list you got from bundle_via_supersede. Empty list means "
            "search the whole corpus (use sparingly -- 280k docs)."
        ),
        max_length=20,
    )
    limit: int = Field(
        10, ge=1, le=25,
        description="Max documents to return.",
    )


class ReadDocumentInput(BaseModel):
    document_id: int | None = Field(
        None,
        description=(
            "dealcloud.document.id. Preferred when known -- you usually "
            "get it from get_org_dossier, search_documents, or "
            "read_document_summary first."
        ),
    )
    document_name: str | None = Field(
        None,
        description=(
            "Partial filename (case-insensitive). Used only if "
            "document_id is not provided. Returns the most-recently-"
            "modified match."
        ),
    )
    web_url: str | None = Field(
        None,
        description=(
            "Document's SharePoint web_url. Used only if document_id "
            "is not provided."
        ),
    )
    max_chars: int = Field(
        20_000, ge=500, le=200_000,
        description=(
            "Truncate the returned body beyond this many chars. "
            "When the doc is long and you need a specific section, "
            "prefer passing `query` over bumping this -- much more "
            "token-efficient."
        ),
    )
    query: str | None = Field(
        None,
        description=(
            "Optional in-doc search: when given, the returned body "
            "is filtered to paragraphs containing this query "
            "(case-insensitive) plus ~500 chars of surrounding "
            "context. Use this when the doc is long (e.g. a PPM or "
            "LPA) and you only need the section about a specific "
            "topic like 'GP commitment' or 'management fee'."
        ),
    )


@slack_registry.tool(
    "search_documents",
    (
        "Find documents BY TOPIC, scoped to org_ids. "
        "Use this BEFORE read_document whenever the user asks about "
        "content (financials, fund terms, deal status, etc.) -- "
        "spelunking the dossier's chronological doc list is slow "
        "and often misses the right doc. Returns up to `limit` rows "
        "ranked by hybrid retrieval (filename trigram + embedding "
        "cosine over doc name+summary). Each row has document_id, "
        "name, path, web_url, summary_preview, score, and `criteria` "
        "-- the IC checklist items that document was mapped to. Lean on "
        "`criteria` when summary_preview is truncated mid-sentence: it "
        "was derived from the full summary, so it says what the document "
        "is about when the preview does not. An EMPTY criteria list is "
        "not a signal -- plenty of documents match no checklist item, "
        "and most of the corpus has not been through that pass at all. "
        "This is top-k ranking, so it can MISS a document that exists "
        "and can never show that one is absent: for 'is there a file "
        "called X', 'what is in folder Y', or any question about a "
        "document's existence or location, call find_documents_by_name, "
        "which scans every indexed path. "
        "Pass the canonical org_ids from bundle_via_supersede; empty "
        "list searches the whole corpus (avoid unless org search has "
        "failed). Read-only."
        + _LINK_RULE
    ),
    SearchDocumentsInput,
)
def search_documents(inp: SearchDocumentsInput, ctx: dict) -> ToolResult:
    from ..document_search import search_documents_for_orgs
    rows = search_documents_for_orgs(
        org_ids=inp.org_ids,
        query=inp.query,
        limit=inp.limit,
        mode="hybrid",
    )
    return ToolResult(output={
        "query": inp.query,
        "org_ids": inp.org_ids,
        "count": len(rows),
        "results": rows,
    })


@slack_registry.tool(
    "read_document",
    (
        "Read the FULL TEXT BODY of a document (PDF / DOCX / PPTX / "
        "XLSX / TXT). More expensive than read_document_summary -- "
        "prefer the summary first, only escalate to this when the "
        "summary doesn't answer the user's question. Cached after "
        "first read. Returns body (possibly truncated to max_chars), "
        "total_chars, truncated flag, plus name / path / web_url for "
        "citing back to the user in Slack. Identify the doc by "
        "document_id (preferred), document_name, or web_url. "
        "For long docs (PPM, LPA, IC memo) pass `query` to filter "
        "the returned body to paragraphs about a specific topic."
        + _LINK_RULE
    ),
    ReadDocumentInput,
)
def read_document(inp: ReadDocumentInput, ctx: dict) -> ToolResult:
    from ..document_body import get_document_body, to_tool_output
    result = get_document_body(
        document_id=inp.document_id,
        document_name=inp.document_name,
        web_url=inp.web_url,
        max_chars=inp.max_chars,
        query=inp.query,
    )
    return ToolResult(output=to_tool_output(result))


# ---------------------------------------------------------------------------
# Deal one-pager (pre-built weekly by deal_cloud_enhancer; we only READ)
# ---------------------------------------------------------------------------

class DealNameInput(BaseModel):
    deal_name: str = Field(
        ...,
        description=(
            "The DEAL name / codename (e.g. 'Project Auto II', "
            "'Project Ostrich V') -- NOT the company name. One company "
            "can have several deals, so the one-pager is keyed by deal. "
            "If the user gives you a company, call list_deals first to "
            "find the deal name."
        ),
        min_length=1, max_length=120,
    )


class CompanyNameInput(BaseModel):
    company: str = Field(
        ...,
        description="Company name to list deals for (e.g. 'Moove').",
        min_length=1, max_length=120,
    )


def _match_deals(cur, deal_name: str) -> list[dict]:
    cur.execute(
        """
        SELECT d.id, d.name, d.status, d.transaction_type,
               o.name AS org_name
          FROM dealcloud.deal d
          LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
         WHERE d.name ILIKE %s
            OR dealcloud.similarity(d.name, %s) > 0.35
         ORDER BY (lower(d.name) = lower(%s)) DESC,
                  dealcloud.similarity(d.name, %s) DESC
         LIMIT 10
        """,
        (f"%{deal_name}%", deal_name, deal_name, deal_name),
    )
    return [dict(r) for r in cur.fetchall()]


def _deals_for_company(cur, company: str) -> list[dict]:
    cur.execute(
        """
        SELECT d.id, d.name, d.status, d.transaction_type, o.name AS org_name
          FROM dealcloud.deal d
          JOIN dealcloud.organization o ON o.id = d.organization_id
         WHERE o.name ILIKE %s
            OR dealcloud.similarity(o.name, %s) > 0.4
         ORDER BY o.name, d.status, d.name
         LIMIT 25
        """,
        (f"%{company}%", company),
    )
    return [dict(r) for r in cur.fetchall()]


def _match_funds(cur, fund_name: str) -> list[dict]:
    cur.execute(
        """
        SELECT id, name, short_name, fund_type, fund_status
          FROM dealcloud.fund
         WHERE name ILIKE %s
            OR short_name ILIKE %s
            OR dealcloud.similarity(name, %s) > 0.35
            OR dealcloud.similarity(short_name, %s) > 0.35
         ORDER BY (lower(name) = lower(%s)) DESC,
                  (lower(short_name) = lower(%s)) DESC,
                  dealcloud.similarity(name, %s) DESC
         LIMIT 10
        """,
        (f"%{fund_name}%", f"%{fund_name}%", fund_name, fund_name,
         fund_name, fund_name, fund_name),
    )
    return [dict(r) for r in cur.fetchall()]


_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")


def _md_to_slack(text: str) -> str:
    """Convert the one-pager's standard markdown into Slack mrkdwn so it
    renders natively in Slack: `[label](url)` -> `<url|label>`,
    `**bold**` -> `*bold*`, `## Heading` -> `*Heading*`, and `- ` / four-
    space-`- ` bullets -> `•` / `◦`. (The stored content_markdown is
    standard markdown -- right for a web view; Slack needs this.)"""
    if not text:
        return ""
    out_lines = []
    for line in text.split("\n"):
        m = re.match(r"^(\s*)-\s+(.*)$", line)
        if line.startswith("## "):
            line = "*" + line[3:].strip() + "*"
        elif m:
            indent = m.group(1)
            bullet = "◦" if len(indent) >= 4 else "•"
            line = f"{indent}{bullet} {m.group(2)}"
        out_lines.append(line)
    text = "\n".join(out_lines)
    text = _MD_BOLD.sub(r"*\1*", text)
    text = _MD_LINK.sub(r"<\2|\1>", text)
    return text


def _contacts_slack_table(content: dict) -> str:
    """Render the contacts section as a caption line for the deal's main
    Ion Pacific contact, plus a monospace Slack table with a row per
    company contact (Name / Email / Role)."""
    their = content.get("their_contacts") or []
    poc = content.get("main_ion_contact") or {}

    caption = "*Ion Pacific contact:* —"
    if poc.get("name"):
        caption = (f"*Ion Pacific contact:* {poc['name']} "
                   f"({poc.get('active_touches', 0)} active touches)")

    rows = []
    for c in their:
        name = (c.get("name") or "").split("(")[0].strip() or "—"
        email = c.get("email") or "—"
        role = c.get("job_title") or c.get("relationship") or "—"
        rows.append([name[:26], email[:32], role[:24]])
    if not rows:
        return caption + "\n_No company contacts on record._"

    headers = ["Name", "Email", "Role"]
    widths = [max(len(str(v)) for v in col) for col in zip(headers, *rows)]
    def fmt(r):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))
    table = "\n".join([fmt(headers), fmt(["-" * w for w in widths])]
                      + [fmt(r) for r in rows])
    return caption + "\n```\n" + table + "\n```"


def _section_slack(section_key: str, content: dict, content_markdown: str,
                   status: str) -> str:
    """Slack-native body for one section, rendered from the typed content
    where it matters (contacts table), else by converting the stored
    markdown to Slack mrkdwn."""
    if section_key == "contacts" and content:
        return _contacts_slack_table(content)
    return _md_to_slack(content_markdown or f"_({status})_")


def _assemble_one_pager(cur, deal_id: int) -> dict | None:
    """Read the latest complete/partial one-pager for a deal and return
    its sections (structured content + rendered markdown) in order, plus
    an assembled markdown blob. None if no one-pager has been built."""
    cur.execute(
        """
        SELECT id, status, generated_at
          FROM dealcloud.deal_one_pager
         WHERE deal_id = %s AND status IN ('complete', 'partial')
         ORDER BY generated_at DESC NULLS LAST
         LIMIT 1
        """,
        (deal_id,),
    )
    pager = cur.fetchone()
    if not pager:
        return None
    cur.execute(
        """
        SELECT s.title, r.section_key, r.status, r.content, r.content_markdown
          FROM dealcloud.deal_one_pager_section_result r
          JOIN dealcloud.deal_one_pager_section s ON s.id = r.section_id
         WHERE r.one_pager_id = %s
         ORDER BY s.sort_order, s.id
        """,
        (pager["id"],),
    )
    sections = [dict(r) for r in cur.fetchall()]
    md = "\n\n".join(
        f"## {s['title']}\n\n{s['content_markdown'] or '_(' + s['status'] + ')_'}"
        for s in sections
    )
    slack_md = "\n\n".join(
        f"*{s['title']}*\n"
        + _section_slack(s["section_key"], s["content"],
                         s["content_markdown"], s["status"])
        for s in sections
    )
    return {
        "one_pager_status": pager["status"],
        "generated_at": pager["generated_at"],
        "sections": sections,
        "markdown": md,
        "slack_markdown": slack_md,
    }


@slack_registry.tool(
    "get_deal_one_pager",
    (
        "Fetch the pre-built one-pager for a DEAL (keyed by deal name / "
        "codename, e.g. 'Project Auto II' -- NOT the company name). "
        "One-pagers are rebuilt weekly (Sunday) for every live-pipeline "
        "deal; this only READS the stored result, it does not generate. "
        "Returns the assembled one-pager (Company Overview, Deal "
        "Overview, Deal History, Contacts, News & Flags, Investors) with "
        "source links. If the deal name matches several deals, you get a "
        "disambiguation list -- ask the user which one. If the name is "
        "actually a company, you get that company's deals to choose "
        "from. If no one-pager exists yet for the matched deal, that's "
        "reported too (it may be a non-pipeline deal that isn't built "
        "weekly)."
    ),
    DealNameInput,
)
def get_deal_one_pager(inp: DealNameInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        matches = _match_deals(cur, inp.deal_name)

        if not matches:
            # Maybe the user gave a company name rather than a deal name.
            company_deals = _deals_for_company(cur, inp.deal_name)
            if company_deals:
                return ToolResult(output={
                    "matched": "company_not_deal",
                    "note": (f"No deal is named '{inp.deal_name}', but it "
                             f"looks like a company. Ask the user which of "
                             f"these deals they mean, then call "
                             f"get_deal_one_pager with that deal name."),
                    "deals": company_deals,
                })
            return ToolResult(output={
                "matched": "none",
                "note": f"No deal found matching '{inp.deal_name}'.",
            })

        if len(matches) > 1:
            # Exact-name hit wins outright; otherwise disambiguate.
            exact = [m for m in matches if m["name"].lower() == inp.deal_name.lower()]
            if len(exact) != 1:
                return ToolResult(output={
                    "matched": "ambiguous",
                    "note": ("Several deals match -- ask the user which one, "
                             "then call again with the exact deal name."),
                    "candidates": matches,
                })
            matches = exact

        deal = matches[0]
        pager = _assemble_one_pager(cur, deal["id"])

    if pager is None:
        return ToolResult(output={
            "matched": "deal_no_one_pager",
            "deal": deal,
            "note": (f"Deal '{deal['name']}' ({deal['status']}) has no "
                     f"one-pager built yet. One-pagers are pre-built "
                     f"weekly for live-pipeline deals; this deal may be "
                     f"outside that set ({deal['status']})."),
        })

    # The one-pager's slack_markdown is large (often >3k tokens) and
    # already Slack-formatted, so DON'T hand it to the model to echo --
    # that's slow to stream and gets truncated at MAX_TOKENS. Instead
    # post it directly via a side_event and tell the model it's done.
    header = f"*One-pager — {deal['name']}*  _({deal['status']})_"
    return ToolResult(
        output={
            "matched": "deal",
            "deal_id": deal["id"],
            "deal_name": deal["name"],
            "deal_status": deal["status"],
            "company": deal.get("org_name"),
            "one_pager_status": pager["one_pager_status"],
            "generated_at": pager["generated_at"],
            "posted": True,
            "note": (
                f"The full one-pager for '{deal['name']}' has ALREADY been "
                "posted to the user directly (it's pre-formatted for Slack). "
                "Do NOT repost or re-summarise it. If you're fetching several "
                "one-pagers, add at most a brief one-line wrap-up after the "
                "last one; for a single one-pager, add nothing further."
            ),
        },
        side_events=[{
            "type": "post_markdown",
            "header": header,
            "markdown": pager["slack_markdown"],
        }],
    )


@slack_registry.tool(
    "list_deals",
    (
        "List the deals we have for a COMPANY (so you can find the deal "
        "name to pass to get_deal_one_pager). Use when the user names a "
        "company rather than a specific deal. Returns each deal's name, "
        "status, and type."
    ),
    CompanyNameInput,
)
def list_deals(inp: CompanyNameInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        deals = _deals_for_company(cur, inp.company)
    return ToolResult(output={
        "company_query": inp.company,
        "count": len(deals),
        "deals": deals,
    })


# ---------------------------------------------------------------------------
# New deals to discuss (Deals Tracker diff)
# ---------------------------------------------------------------------------

class NewDealsToDiscussInput(BaseModel):
    as_of_date: str | None = Field(
        None,
        description=(
            "Optional meeting week as an ISO date 'YYYY-MM-DD'. The tool "
            "compares the latest 'Deals Tracker' posted on or before this "
            "date against the one before it. Omit to use the two most "
            "recent trackers -- the usual 'what's new this week' case."
        ),
    )


@slack_registry.tool(
    "find_new_deals_to_discuss",
    (
        "List the deals NEWLY up for discussion at a pipeline meeting. "
        "Reads the weekly 'Deals Tracker <date>.xlsx' files posted in "
        "#existing_pipeline, diffs the latest one (or the one for a given "
        "week via as_of_date) against the previous week's, and returns "
        "deals that are 'to be discussed' (any status EXCEPT 'Warming "
        "Station') now but weren't last week -- i.e. absent last week, or "
        "only 'Warming Station' before. Each returned deal_name is a deal "
        "codename matching get_deal_one_pager, so call that for each one if "
        "the user asks to see the one-pagers."
    ),
    NewDealsToDiscussInput,
)
def find_new_deals_to_discuss(inp: NewDealsToDiscussInput, ctx: dict) -> ToolResult:
    from datetime import date

    as_of = None
    if inp.as_of_date:
        try:
            as_of = date.fromisoformat(inp.as_of_date.strip())
        except ValueError:
            return ToolResult(output={
                "error": (f"Couldn't parse as_of_date '{inp.as_of_date}'. "
                          "Use an ISO date like 2026-06-15."),
            })
    try:
        return ToolResult(output=compute_new_deals_to_discuss(as_of))
    except TrackerError as e:
        return ToolResult(output={"error": str(e)})


# ---------------------------------------------------------------------------
# List all deals (book overview: deal -> counterpart company -> status)
# ---------------------------------------------------------------------------

# Most-active statuses first so a capped page still shows the deals people
# usually care about; everything else (incl. the ~1,300 'Passed/Dead')
# sorts last. Unknown/new statuses fall after this list.
_STATUS_PRIORITY = (
    "Active Pipeline", "Under Observation", "Early Discussions",
    "Pre-Pipeline", "Warming Station", "Partnership", "Portfolio Company",
    "Passed/Dead",
)


class ListAllDealsInput(BaseModel):
    status: list[str] | None = Field(
        None,
        description=(
            "Optional: only deals whose current status is one of these "
            "(exact match, case-insensitive). Known statuses: 'Active "
            "Pipeline', 'Under Observation', 'Early Discussions', "
            "'Pre-Pipeline', 'Warming Station', 'Partnership', 'Portfolio "
            "Company', 'Passed/Dead'. Omit for all statuses. Use this to "
            "answer 'all active deals' etc. -- most of the book is "
            "'Passed/Dead'."
        ),
    )
    company: str | None = Field(
        None,
        description=("Optional substring filter on the counterpart company "
                     "name OR the deal name/codename (case-insensitive)."),
        max_length=120,
    )
    limit: int = Field(
        500,
        description=("Max deals to return, most-active statuses first. "
                     "Default 500 covers every live/portfolio deal; raise "
                     "(up to 2000) to include 'Passed/Dead' history."),
        ge=1, le=2000,
    )
    offset: int = Field(0, description="Skip this many rows (paging).", ge=0)


@slack_registry.tool(
    "list_all_deals",
    (
        "List deals across the whole book with each deal's counterpart "
        "company and current status -- the all-deals companion to "
        "list_deals (which is scoped to one company). Returns a "
        "status_counts summary over the entire book plus a page of "
        "{deal_name, company, status, transaction_type}, ordered "
        "most-active status first. Filter with `status` (e.g. ['Active "
        "Pipeline']) and/or `company`; page with limit/offset. Most of the "
        "book is 'Passed/Dead', so prefer a status filter unless the user "
        "really wants the full history -- the status_counts in the "
        "response gives current totals per status."
    ),
    ListAllDealsInput,
)
def list_all_deals(inp: ListAllDealsInput, ctx: dict) -> ToolResult:
    where: list[str] = []
    params: list[Any] = []
    if inp.status:
        where.append("lower(d.status) = ANY(%s)")
        params.append([s.strip().lower() for s in inp.status])
    if inp.company:
        where.append("(o.name ILIKE %s OR d.name ILIKE %s)")
        params += [f"%{inp.company}%", f"%{inp.company}%"]
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    prio_case = ("CASE d.status "
                 + " ".join(f"WHEN %s THEN {i}"
                            for i in range(len(_STATUS_PRIORITY)))
                 + f" ELSE {len(_STATUS_PRIORITY)} END")

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Status breakdown over the WHOLE book (cheap, always useful).
        cur.execute(
            "SELECT status, COUNT(*) AS n FROM dealcloud.deal "
            "GROUP BY status ORDER BY n DESC"
        )
        status_counts = {(r["status"] or "(unknown)"): r["n"]
                         for r in cur.fetchall()}
        # Total matching the filters (so the caller knows if it's truncated).
        cur.execute(
            f"SELECT COUNT(*) AS n FROM dealcloud.deal d "
            f"LEFT JOIN dealcloud.organization o ON o.id = d.organization_id "
            f"{where_sql}",
            params,
        )
        total = cur.fetchone()["n"]
        # The page. Param order follows the SQL text: WHERE, then ORDER BY
        # CASE, then LIMIT/OFFSET.
        cur.execute(
            f"""
            SELECT d.name AS deal_name, o.name AS company, d.status,
                   d.transaction_type
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
              {where_sql}
             ORDER BY {prio_case}, lower(o.name) NULLS LAST, lower(d.name)
             LIMIT %s OFFSET %s
            """,
            params + list(_STATUS_PRIORITY) + [inp.limit, inp.offset],
        )
        deals = [dict(r) for r in cur.fetchall()]

    return ToolResult(output={
        "total_matching": total,
        "returned": len(deals),
        "offset": inp.offset,
        "truncated": inp.offset + len(deals) < total,
        "status_counts": status_counts,
        "filters": {"status": inp.status, "company": inp.company},
        "deals": deals,
    })


# ---------------------------------------------------------------------------
# Deal underlying companies (confidence-tiered)
# ---------------------------------------------------------------------------

class DealUnderlyingCompaniesInput(BaseModel):
    deal_name: str = Field(
        ...,
        description=(
            "The DEAL name / codename (e.g. 'Project Lego'). If you pass a "
            "company name instead, the tool returns that company's deals to "
            "choose from."
        ),
        min_length=1, max_length=120,
    )
    include_derived: bool = Field(
        True,
        description=(
            "Include the lower-confidence DERIVED companies (pulled from "
            "document mentions, NOT confirmed in DealCloud). They're returned "
            "clearly tiered + flagged. Set False to get only the high-"
            "confidence DealCloud-backed holdings."
        ),
    )
    derived_limit: int = Field(
        25, ge=1, le=100,
        description="Max derived companies to return (there can be dozens of "
                    "low-confidence document mentions).",
    )
    include_nav_allocation: bool = Field(
        True,
        description=(
            "Read the deal's IC memo at query time to surface any stated "
            "per-company NAV / value-driver allocation passage. Adds one "
            "document read; set False to skip for speed."
        ),
    )


# Single query: every deal_underlying_company row for the deal, with the
# count of *focused* deal documents (<=10 distinct orgs -- i.e. deal-specific,
# not a firm-level market-map/overview deck) that map each org, plus the most
# relevant evidence document for the connection. Fully schema-qualified and
# no bare trigram operator -- safe on the pooled Neon endpoint the MCP server
# uses (see neon_pooler_search_path_drift).
_DUC_SQL = """
WITH deal_docs AS (
    SELECT dd.document_id AS doc_id
      FROM dealcloud.document_deal dd
     WHERE dd.deal_id = %s
),
doc_orgcount AS (
    SELECT d.doc_id,
           (SELECT count(DISTINCT oe.organization_id)
              FROM dealcloud.organization_entity oe
             WHERE oe.entity_type = 'document' AND oe.entity_id = d.doc_id) AS n_orgs
      FROM deal_docs d
),
duc AS (
    SELECT u.organization_id, o.name,
           u.connection_source, u.is_value_driver, u.is_underlying,
           u.derived_n_docs,
           u.nav_pct_estimate, u.nav_estimate_confidence,
           u.nav_estimate_basis, u.nav_estimate_source_document_id,
           u.nav_estimated_at
      FROM dealcloud.deal_underlying_company u
      JOIN dealcloud.organization o ON o.id = u.organization_id
     WHERE u.deal_id = %s
)
SELECT duc.organization_id, duc.name, duc.connection_source,
       duc.is_value_driver, duc.is_underlying, duc.derived_n_docs,
       duc.nav_pct_estimate, duc.nav_estimate_confidence,
       duc.nav_estimate_basis, duc.nav_estimated_at,
       navdoc.id AS nav_doc_id, navdoc.name AS nav_doc_name,
       navdoc.web_url AS nav_doc_url,
       (SELECT count(*) FROM doc_orgcount dc
          JOIN dealcloud.organization_entity oe
            ON oe.entity_type = 'document' AND oe.entity_id = dc.doc_id
         WHERE oe.organization_id = duc.organization_id
           AND dc.n_orgs <= 10) AS focused_docs,
       ev.document_id AS ev_doc_id, ev.ev_name AS ev_doc_name,
       ev.web_url AS ev_doc_url
  FROM duc
  LEFT JOIN dealcloud.document navdoc
         ON navdoc.id = duc.nav_estimate_source_document_id
  LEFT JOIN LATERAL (
      SELECT doc.id AS document_id, doc.name AS ev_name, doc.web_url
        FROM doc_orgcount dc
        JOIN dealcloud.organization_entity oe
          ON oe.entity_type = 'document' AND oe.entity_id = dc.doc_id
        JOIN dealcloud.document doc ON doc.id = dc.doc_id
       WHERE oe.organization_id = duc.organization_id
       ORDER BY (dc.n_orgs <= 10) DESC, doc.modified_at DESC NULLS LAST
       LIMIT 1
  ) ev ON TRUE
 ORDER BY duc.is_value_driver DESC, duc.name
"""


def _derived_tier(focused_docs: int) -> str:
    """Confidence that an llm_derived org is genuinely part of the deal,
    from how many *deal-specific* (<=10-org) documents corroborate it.
    Mentions that only show up in broad market-map/overview decks score 0
    -> low (the dominant noise case)."""
    if focused_docs >= 3:
        return "high"
    if focused_docs >= 1:
        return "medium"
    return "low"


def _ic_memo_allocation(cur, deal_id: int) -> dict:
    """Find the deal's IC memo and pull the passage most likely to state a
    per-company NAV / value-driver allocation. Returns the doc ref + an
    excerpt (or a not-found note). Per-company % is NOT in structured data,
    so this best-effort reads the memo text at query time."""
    cur.execute(
        """
        SELECT doc.id, doc.name, doc.web_url
          FROM dealcloud.document_deal dd
          JOIN dealcloud.document doc ON doc.id = dd.document_id
         WHERE dd.deal_id = %s
           AND (doc.name ILIKE %s OR doc.name ILIKE %s OR doc.name ILIKE %s)
         ORDER BY (doc.name ILIKE %s) DESC, doc.modified_at DESC NULLS LAST
         LIMIT 1
        """,
        (deal_id, "%IC %", "%IC_%", "%Investment Committee%", "%.pdf"),
    )
    memo = cur.fetchone()
    if not memo:
        return {"ic_memo": None, "allocation_excerpt": None,
                "note": "No IC-memo-shaped document is linked to this deal."}

    from ..document_body import get_document_body
    # Try the terms Ion memos use for the holdings split, in priority order;
    # take the first that actually matches a passage. get_document_body
    # returns the document head (with a query_not_found error) when a term is
    # absent, so a miss is distinguishable from a hit.
    res = None
    for term in ("value driver", "allocation", "% of NAV", "fair value"):
        r = get_document_body(document_id=memo["id"], query=term, max_chars=6000)
        if not r.ok:
            res = r  # body genuinely unavailable -- stop, report it
            break
        res = r
        if not (r.error and str(r.error).startswith("query_not_found")):
            break  # real hit on this term

    excerpt = res.body if res.ok else None
    note = None
    if res.ok and res.error and str(res.error).startswith("query_not_found"):
        note = ("No explicit per-company allocation passage matched in the "
                "memo text; returned the document head. The per-company split "
                "may be in a table or image we can't extract -- do NOT infer "
                "percentages that aren't present in this text.")
    elif not res.ok:
        note = f"IC memo body unavailable ({res.error})."
    return {
        "ic_memo": {"document_id": memo["id"], "name": memo["name"],
                    "web_url": memo["web_url"]},
        "allocation_excerpt": excerpt,
        "note": note,
    }


@slack_registry.tool(
    "deal_underlying_companies",
    (
        "Authoritative, confidence-tiered list of a DEAL's UNDERLYING "
        "PORTFOLIO COMPANIES. Use this whenever the user asks what companies "
        "are under / inside / part of / held by a deal (especially fund or GP "
        "stakes). It deterministically separates the real holdings from "
        "document noise, which free-text document reading does NOT: "
        "(1) `confirmed_companies` -- DealCloud-backed, HIGH confidence, with "
        "`is_value_driver` flags -- THESE are the actual underlying companies; "
        "(2) `derived_companies` -- pulled from document mentions and NOT "
        "confirmed in DealCloud, each tiered high/medium/LOW with an evidence "
        "document link. MOST are LOW-confidence market-map / comparable / "
        "customer mentions, NOT holdings -- never present a low/medium derived "
        "company as a portfolio company without saying so and citing its "
        "evidence doc; (3) `deal_financials` (deal-level NAV/invested) and "
        "`nav_allocation` (the IC-memo passage to read any per-company split "
        "from -- there is NO structured per-company allocation, so only report "
        "percentages actually stated in that excerpt). Read-only. Prefer this "
        "over guessing underlying companies from raw documents."
    ),
    DealUnderlyingCompaniesInput,
)
def deal_underlying_companies(inp: DealUnderlyingCompaniesInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        matches = _match_deals(cur, inp.deal_name)

        if not matches:
            company_deals = _deals_for_company(cur, inp.deal_name)
            if company_deals:
                return ToolResult(output={
                    "matched": "company_not_deal",
                    "note": (f"No deal is named '{inp.deal_name}', but it looks "
                             f"like a company. Ask the user which deal, then "
                             f"call again with that deal name."),
                    "deals": company_deals,
                })
            return ToolResult(output={
                "matched": "none",
                "note": f"No deal found matching '{inp.deal_name}'.",
            })
        if len(matches) > 1:
            exact = [m for m in matches if m["name"].lower() == inp.deal_name.lower()]
            if len(exact) != 1:
                return ToolResult(output={
                    "matched": "ambiguous",
                    "note": ("Several deals match -- ask the user which one, "
                             "then call again with the exact deal name."),
                    "candidates": matches,
                })
            matches = exact
        deal = matches[0]

        # Deal-level financials (NO per-company allocation exists in source).
        cur.execute(
            """
            SELECT d.id, d.name, d.status, d.transaction_type,
                   d.organization_id AS counterparty_org_id,
                   o.name AS counterparty_name,
                   d.invested_capital, d.deal_size, d.fair_value,
                   d.realized_capital, d.total_value_to_invested, d.co_invest
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
             WHERE d.id = %s
            """,
            (deal["id"],),
        )
        d = cur.fetchone()

        cur.execute(_DUC_SQL, (deal["id"], deal["id"]))
        rows = [dict(r) for r in cur.fetchall()]

        # NAV split is precomputed per company by build_underlying_nav_estimates
        # (best-effort IC-memo read, refreshed periodically). Serve that
        # structured + fast. Only fall back to a live memo read if NOTHING is
        # precomputed yet for this deal (cron hasn't reached it).
        _conf_rows = [r for r in rows if r["connection_source"] != "llm_derived"]
        _has_precomputed = any(r["nav_estimated_at"] for r in _conf_rows)
        if _has_precomputed:
            _src = next(({"document_id": r["nav_doc_id"], "name": r["nav_doc_name"],
                          "web_url": r["nav_doc_url"]}
                         for r in _conf_rows if r["nav_doc_id"]), None)
            _est_at = max((r["nav_estimated_at"] for r in _conf_rows
                           if r["nav_estimated_at"]), default=None)
            nav = {
                "source": "precomputed",
                "ic_memo": _src,
                "estimated_at": _est_at.isoformat() if _est_at else None,
                "method": ("Per-company NAV share, best-effort extracted from the "
                           "deal's IC memo and refreshed periodically. Per-company "
                           "percentages are on each confirmed company's `nav` "
                           "field; a null nav_pct means the memo didn't state that "
                           "company's size. Estimates, not exact figures."),
            }
        elif inp.include_nav_allocation:
            nav = _ic_memo_allocation(cur, deal["id"])
            nav["source"] = "live_read_fallback"
        else:
            nav = None

    confirmed, derived = [], []
    for r in rows:
        if r["connection_source"] == "llm_derived":
            derived.append(r)
        else:
            confirmed.append(r)

    def _ev(r):
        return ({"document_id": r["ev_doc_id"], "name": r["ev_doc_name"],
                 "web_url": r["ev_doc_url"]} if r["ev_doc_id"] else None)

    def _nav(r):
        if r["nav_estimated_at"] is None:
            return None  # not yet estimated by the periodic job
        return {
            "nav_pct_estimate": r["nav_pct_estimate"],
            "confidence": r["nav_estimate_confidence"],
            "basis": r["nav_estimate_basis"],
            "source_document": (
                {"document_id": r["nav_doc_id"], "name": r["nav_doc_name"],
                 "web_url": r["nav_doc_url"]} if r["nav_doc_id"] else None),
        }

    confirmed_out = [{
        "org_id": r["organization_id"], "name": r["name"], "confidence": "high",
        "source": r["connection_source"],
        "is_value_driver": bool(r["is_value_driver"]),
        "nav": _nav(r),
    } for r in confirmed]
    # value drivers first, then alphabetical (SQL already ordered so)

    derived_sorted = sorted(
        derived, key=lambda r: (-(r["focused_docs"] or 0),
                                -(r["derived_n_docs"] or 0), r["name"])
    )
    derived_out = []
    for r in derived_sorted[:inp.derived_limit]:
        tier = _derived_tier(r["focused_docs"] or 0)
        derived_out.append({
            "org_id": r["organization_id"], "name": r["name"],
            "confidence": tier,
            "deal_specific_doc_count": r["focused_docs"] or 0,
            "total_mention_doc_count": r["derived_n_docs"],
            "assessment": (
                "Corroborated by deal-specific documents."
                if tier == "high" else
                "Appears in a deal-specific document; verify before treating "
                "as a holding." if tier == "medium" else
                "Only appears in broad market-map / overview documents -- "
                "most likely a comparable / customer / market mention, NOT a "
                "portfolio company."
            ),
            "evidence_document": _ev(r),
        })

    inv = d["invested_capital"]
    out = {
        "matched": "deal",
        "deal": {
            "deal_id": d["id"], "deal_name": d["name"], "status": d["status"],
            "transaction_type": d["transaction_type"],
            "counterparty_org_id": d["counterparty_org_id"],
            "counterparty_name": d["counterparty_name"],
        },
        "deal_financials": {
            "invested_capital": abs(inv) if inv is not None else None,
            "deal_size": d["deal_size"],
            "fair_value_nav": d["fair_value"],
            "realized_capital": d["realized_capital"],
            "total_value_to_invested": d["total_value_to_invested"],
            "co_invest": d["co_invest"],
            "note": ("Deal/fund-level only. There is NO per-underlying-company "
                     "allocation in the source data; any per-company % must "
                     "come from the IC-memo excerpt in `nav_allocation`."),
        },
        "confirmed_companies": confirmed_out,
        "value_drivers": [c["name"] for c in confirmed_out if c["is_value_driver"]],
        "derived_summary": {
            "total": len(derived),
            "shown": len(derived_out),
            "omitted": max(0, len(derived) - len(derived_out)),
            "by_confidence": {
                t: sum(1 for r in derived if _derived_tier(r["focused_docs"] or 0) == t)
                for t in ("high", "medium", "low")
            },
        },
        "derived_companies": derived_out if inp.include_derived else [],
        "nav_allocation": nav,
        "present_instructions": (
            "Lead with `confirmed_companies` -- these ARE the deal's underlying "
            "portfolio companies (DealCloud-backed); call out `value_drivers`. "
            "Present `derived_companies` separately and clearly as LOWER-"
            "confidence document mentions: low-confidence ones are almost "
            "certainly market-map / comparable / customer noise, not holdings "
            "-- if you mention them, say so and link the evidence_document. "
            "For per-company NAV %, use each confirmed company's `nav` field "
            "(precomputed from the deal's IC memo, with a confidence + basis + "
            "source_document): report nav_pct_estimate where present and cite "
            "its basis; where `nav` is null or nav_pct_estimate is null, say the "
            "memo doesn't state that company's size rather than guessing. These "
            "are estimates -- never invent percentages."
        ),
    }
    return ToolResult(output=out)


# ---------------------------------------------------------------------------
# Fundraising / LP commitments
# ---------------------------------------------------------------------------

class FundraisingSummaryInput(BaseModel):
    fund_name: str | None = Field(
        None,
        description=(
            "Fund or SPV name to focus on (e.g. 'ION Pacific Fund IV', "
            "'Project Auto SPV'). Partial/case-insensitive match. "
            "Omit to get a summary of ALL funds in DealCloud."
        ),
        max_length=200,
    )
    year: int | None = Field(
        None,
        description=(
            "Filter LP commitments to those whose `created_date` falls in "
            "this calendar year (e.g. 2026). Omit for all-time totals. "
            "Use this for 'how much did we raise this year' queries."
        ),
        ge=2000,
        le=2100,
    )
    include_lp_detail: bool = Field(
        True,
        description=(
            "When True (default) include a per-LP breakdown of who "
            "committed how much, with stage, status, and transfer info. "
            "Set False for a fund-level-only summary."
        ),
    )


@slack_registry.tool(
    "get_fundraising_summary",
    (
        "LP commitment / fundraising summary for Ion Pacific funds. "
        "Use this for ANY question about how much capital was raised, who "
        "committed to which fund/SPV, fundraising progress vs target, or "
        "LP-by-LP commitment breakdown. Returns: fund-level stats "
        "(fund_size, fundraise_target, sum_of_lp_capital, gp_commit, "
        "count_of_lps, close dates) PLUS -- when include_lp_detail=True "
        "-- every LP's actual_commitment_amount, stage, status, and "
        "transfer status. Filter by `fund_name` (fuzzy) to focus on one "
        "fund; use `year` to scope to commitments entered in a given "
        "calendar year ('how much did we raise in 2026'). All amounts are "
        "in the currency stored in DealCloud (typically USD millions). "
        "Read-only."
    ),
    FundraisingSummaryInput,
)
def get_fundraising_summary(inp: FundraisingSummaryInput, ctx: dict) -> ToolResult:
    # DealCloud usage note (discovered from live data):
    # - actual_commitment_amount is almost never filled (<1% of rows).
    # - probability_adjusted WHERE stage='5. Committed' is the real committed
    #   capital and exactly matches fund.sum_of_lp_capital for closed funds.
    # - 'transferred' tracks LP-to-LP ownership transfers, not money wired.
    # - created_date = when the record was entered in DC (used as year proxy).
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # 1. Fund-level summary
        fund_params: list[Any] = []
        if inp.fund_name:
            fund_where = (
                "WHERE f.name ILIKE %s "
                "OR dealcloud.similarity(f.name, %s) > 0.3"
            )
            fund_params = [f"%{inp.fund_name}%", inp.fund_name]
        else:
            fund_where = ""

        # Year filter on created_date (proxy for when the commitment was entered).
        # Note: DC has no explicit "commitment close date" field, so this is
        # approximate. fund.sum_of_lp_capital is authoritative for all-time totals.
        year_filter = (
            "AND EXTRACT(YEAR FROM c.created_date) = %s"
            if inp.year else ""
        )
        agg_year_params: list[Any] = [inp.year] if inp.year else []

        cur.execute(
            f"""
            SELECT
                f.id            AS fund_id,
                f.dc_id         AS fund_dc_id,
                f.name          AS fund_name,
                f.short_name,
                f.fund_type,
                f.fund_status,
                f.fund_size,
                f.fundraise_target,
                f.sum_of_lp_capital,
                f.gp_commit_amount,
                f.gp_commit_pct,
                f.count_of_lps,
                f.vintage_year,
                f.first_close_date,
                f.final_close_date,
                f.next_close_date,
                -- Committed (stage 5): the real LP capital figure
                COUNT(c.id)
                    FILTER (WHERE c.stage = '5. Committed' {year_filter})
                    AS committed_lp_count,
                SUM(c.probability_adjusted)
                    FILTER (WHERE c.stage = '5. Committed' {year_filter})
                    AS committed_lp_total,
                -- Active pipeline (stages 2-4)
                SUM(c.probability_adjusted)
                    FILTER (WHERE c.stage IN (
                        '2. Low probability','3. Medium probability','4. High probability')
                        {year_filter})
                    AS pipeline_total,
                COUNT(c.id)
                    FILTER (WHERE c.stage IN (
                        '2. Low probability','3. Medium probability','4. High probability')
                        {year_filter})
                    AS pipeline_count
            FROM dealcloud.fund f
            LEFT JOIN dealcloud.commitment c ON c.fund_dc_id = f.dc_id
            {fund_where}
            GROUP BY f.id, f.dc_id, f.name, f.short_name, f.fund_type,
                     f.fund_status, f.fund_size, f.fundraise_target,
                     f.sum_of_lp_capital, f.gp_commit_amount,
                     f.gp_commit_pct, f.count_of_lps, f.vintage_year,
                     f.first_close_date, f.final_close_date, f.next_close_date
            ORDER BY f.fund_size DESC NULLS LAST, f.name
            """,
            agg_year_params * 4 + fund_params,
        )
        funds_raw = [dict(r) for r in cur.fetchall()]

        if not funds_raw:
            return ToolResult(output={
                "matched": "none",
                "note": (
                    f"No fund found matching '{inp.fund_name}'."
                    if inp.fund_name else "No funds found in DealCloud."
                ),
            })

        # 2. Per-LP detail — committed only (stage 5) unless year filter narrows scope
        lp_by_fund: dict[int, list[dict]] = {}
        if inp.include_lp_detail:
            fund_dc_ids = [f["fund_dc_id"] for f in funds_raw]
            lp_params: list[Any] = [fund_dc_ids]
            lp_year_clause = ""
            if inp.year:
                lp_year_clause = "AND EXTRACT(YEAR FROM c.created_date) = %s"
                lp_params.append(inp.year)

            cur.execute(
                f"""
                SELECT
                    c.fund_dc_id,
                    COALESCE(o.name, c.investor_name) AS investor,
                    o.id                    AS investor_org_id,
                    c.investor_type,
                    c.probability_adjusted  AS commitment_amount,
                    c.commitment_potential,
                    c.stage,
                    c.status,
                    c.fundraising_status,
                    c.created_date
                FROM dealcloud.commitment c
                LEFT JOIN dealcloud.organization o ON o.id = c.investor_org_id
                WHERE c.fund_dc_id = ANY(%s::int[])
                  AND c.stage NOT IN ('6. Declined or lost', '1. Target')
                  {lp_year_clause}
                ORDER BY c.fund_dc_id,
                         (c.stage = '5. Committed') DESC,
                         c.probability_adjusted DESC NULLS LAST
                """,
                lp_params,
            )
            for row in cur.fetchall():
                fdc = row["fund_dc_id"]
                lp_by_fund.setdefault(fdc, []).append({
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in row.items()
                    if k != "fund_dc_id"
                })

    def _fmt_fund(f: dict) -> dict:
        lps = lp_by_fund.get(f["fund_dc_id"], []) if inp.include_lp_detail else None
        return {
            "fund_id":           f["fund_id"],
            "fund_name":         f["fund_name"],
            "short_name":        f["short_name"],
            "fund_type":         f["fund_type"],
            "fund_status":       f["fund_status"],
            "vintage_year":      f["vintage_year"],
            # Authoritative fund-level capital (DealCloud's own aggregates)
            "fund_size":         f["fund_size"],
            "fundraise_target":  f["fundraise_target"],
            "sum_of_lp_capital": f["sum_of_lp_capital"],
            "gp_commit_amount":  f["gp_commit_amount"],
            "gp_commit_pct":     f["gp_commit_pct"],
            "count_of_lps":      f["count_of_lps"],
            "first_close_date":  (f["first_close_date"].date().isoformat()
                                  if f["first_close_date"] else None),
            "final_close_date":  (f["final_close_date"].date().isoformat()
                                  if f["final_close_date"] else None),
            "next_close_date":   (f["next_close_date"].date().isoformat()
                                  if f["next_close_date"] else None),
            # From commitment rows (filtered by year if set)
            "commitment_stats": {
                "year_filter": inp.year,
                "year_filter_note": (
                    "Filtered by created_date (when the record was entered in "
                    "DealCloud -- approximate proxy for commitment date since "
                    "no explicit commitment-close-date field exists). "
                    "sum_of_lp_capital above is the authoritative all-time total."
                ) if inp.year else None,
                "committed_lp_count": f["committed_lp_count"] or 0,
                "committed_lp_total": f["committed_lp_total"],
                "pipeline_count":     f["pipeline_count"] or 0,
                "pipeline_total":     f["pipeline_total"],
            },
            **({"lp_commitments": lps} if inp.include_lp_detail else {}),
        }

    return ToolResult(output={
        "fund_name_query": inp.fund_name,
        "year_filter":     inp.year,
        "fund_count":      len(funds_raw),
        "funds":           [_fmt_fund(f) for f in funds_raw],
        "data_note": (
            "commitment_amount uses probability_adjusted (the field DealCloud "
            "actually populates). stage='5. Committed' rows are the real LP "
            "capital; stages 2-4 are active pipeline. sum_of_lp_capital on "
            "each fund is DealCloud's own pre-aggregated LP total (authoritative "
            "for all-time figures)."
        ),
    })


# ---------------------------------------------------------------------------
# Fund status -- NAV / capital called / capital returned, and the per-deal
# deployment breakdown. Complements get_fundraising_summary (which covers
# the LP-commitment/fundraising side) with the deployment/performance side:
# how much of the fund's capital actually went into deals, how much has
# come back, and what the fund/each deal is worth now.
#
# DealCloud data-quality note (see Fund Performance query below): the
# "Funds Performance" entity (quarterly fund-level NAV/contributions/
# distributions) only has real data for 3 funds in this tenant (the
# Stonecutter I/II/III fund family) -- everywhere else it's unfilled in
# DealCloud itself. For those other funds we fall back to a rollup derived
# from the fund's linked deals (deal.invested_capital/fair_value/
# realized_capital), clearly labeled as an approximation, with a
# deal_data_coverage block so callers can see how many of the fund's deals
# actually have each figure recorded (missing != zero).
# ---------------------------------------------------------------------------

_FUND_AGG_JOIN_SQL = """
              LEFT JOIN LATERAL (
                  SELECT date_q, nav, cumulative_contributions, cumulative_distributions
                    FROM dealcloud.fund_performance
                   WHERE fund_id = f.id AND date_q IS NOT NULL
                   ORDER BY date_q DESC LIMIT 1
              ) fp ON true
              LEFT JOIN LATERAL (
                  SELECT count(*) AS n_deals,
                         count(invested_capital) AS n_deals_with_invested,
                         sum(abs(invested_capital)) AS sum_invested,
                         count(fair_value) AS n_deals_with_fv,
                         sum(fair_value) AS sum_fv,
                         count(realized_capital) AS n_deals_with_realized,
                         sum(realized_capital) AS sum_realized
                    FROM dealcloud.deal
                   WHERE fund_id = f.id
              ) agg ON true
"""


def _fund_performance_block(row: dict) -> dict:
    """Build the {source, as_of_date, capital_called_to_date,
    capital_returned_to_date, current_value_nav, ...} block from a row that
    carries both the latest fund_performance columns (date_q/nav/
    cumulative_contributions/cumulative_distributions) and the deal-rollup
    agg columns (n_deals/n_deals_with_invested/sum_invested/n_deals_with_fv/
    sum_fv/n_deals_with_realized/sum_realized) -- i.e. a row produced by a
    query that joins _FUND_AGG_JOIN_SQL."""
    n_deals = row.get("n_deals") or 0
    if row.get("date_q") is not None:
        contributed = row.get("cumulative_contributions")
        return {
            "source": "fund_performance",
            "as_of_date": row["date_q"].date().isoformat(),
            # DealCloud stores capital calls negative (LP-outflow / J-curve
            # convention, same sign convention as deal.invested_capital
            # elsewhere in this codebase) -- ABS() for an intuitive
            # "money invested" figure, consistent with distributions/NAV
            # which are already stored positive.
            "capital_called_to_date": abs(contributed) if contributed is not None else None,
            "capital_returned_to_date": row.get("cumulative_distributions"),
            "current_value_nav": row.get("nav"),
            "note": (
                "DealCloud's own quarterly Fund Performance record for this "
                "fund -- authoritative."
            ),
        }
    if n_deals:
        return {
            "source": "derived_from_deals",
            "as_of_date": None,
            "capital_called_to_date": row.get("sum_invested"),
            "capital_returned_to_date": row.get("sum_realized"),
            "current_value_nav": row.get("sum_fv"),
            "deal_data_coverage": {
                "deals_in_fund": n_deals,
                "deals_with_invested_capital": row.get("n_deals_with_invested") or 0,
                "deals_with_fair_value": row.get("n_deals_with_fv") or 0,
                "deals_with_realized_capital": row.get("n_deals_with_realized") or 0,
            },
            "note": (
                "DealCloud has no quarterly Fund Performance record for this "
                "fund, so these figures are DERIVED by summing this fund's "
                "linked deals' invested_capital / fair_value / "
                "realized_capital. Treat as an approximation, NOT an "
                "official NAV -- see deal_data_coverage for how many of the "
                "fund's deals actually have each field populated in "
                "DealCloud (a deal missing a field is EXCLUDED from the "
                "sum, not counted as zero)."
            ),
        }
    return {
        "source": "no_data",
        "as_of_date": None,
        "capital_called_to_date": None,
        "capital_returned_to_date": None,
        "current_value_nav": None,
        "note": (
            "No Fund Performance record and no linked deals with financial "
            "data found in DealCloud for this fund."
        ),
    }


class ListFundsInput(BaseModel):
    name: str | None = Field(
        None,
        description=(
            "Optional fuzzy filter on fund name or short name (e.g. "
            "'Stonecutter', 'Pathway'). Omit to list every fund."
        ),
        max_length=200,
    )
    reportable_only: bool = Field(
        False,
        description=(
            "When True, only include funds DealCloud flags as reportable "
            "under Ion Pacific's GP/affiliated-entities AUM "
            "(fund.reporting_status = 'Yes') -- excludes internal/test/"
            "friends-and-family vehicles. Default False returns every fund."
        ),
    )
    limit: int = Field(100, ge=1, le=200, description="Max funds to return.")
    offset: int = Field(0, ge=0, description="Skip this many rows (paging).")


@slack_registry.tool(
    "list_funds",
    (
        "High-level status for every fund/SPV, one row each: capital "
        "committed by LPs (fund_size, sum_of_lp_capital), capital called "
        "to date, capital returned to investors to date, and the most "
        "recent valuation of the fund (NAV) -- plus how many deals the "
        "fund holds. Each fund's `performance` block reports whether the "
        "called/returned/NAV figures come from DealCloud's own quarterly "
        "Fund Performance record (`source: fund_performance`, "
        "authoritative -- currently only a few funds have this) or are "
        "derived by summing the fund's deals (`source: derived_from_deals`, "
        "an approximation -- check `deal_data_coverage`), or that there's "
        "no data at all (`source: no_data`). Filter with `name` (fuzzy) or "
        "`reportable_only`; page with limit/offset. Use get_fund_status for "
        "one fund's full per-deal breakdown, and get_fundraising_summary "
        "for a per-LP commitment breakdown."
    ),
    ListFundsInput,
)
def list_funds(inp: ListFundsInput, ctx: dict) -> ToolResult:
    where: list[str] = []
    params: list[Any] = []
    if inp.name:
        where.append(
            "(f.name ILIKE %s OR f.short_name ILIKE %s "
            "OR dealcloud.similarity(f.name, %s) > 0.3)"
        )
        params += [f"%{inp.name}%", f"%{inp.name}%", inp.name]
    if inp.reportable_only:
        where.append("f.reporting_status = 'Yes'")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT COUNT(*) AS n FROM dealcloud.fund f {where_sql}", params)
        total = cur.fetchone()["n"]

        cur.execute(
            f"""
            SELECT f.id AS fund_id, f.name AS fund_name, f.short_name, f.fund_type,
                   f.fund_status, f.reporting_status, f.vintage_year,
                   f.fund_size, f.sum_of_lp_capital,
                   fp.date_q, fp.nav, fp.cumulative_contributions, fp.cumulative_distributions,
                   agg.n_deals, agg.n_deals_with_invested, agg.sum_invested,
                   agg.n_deals_with_fv, agg.sum_fv,
                   agg.n_deals_with_realized, agg.sum_realized
              FROM dealcloud.fund f
              {_FUND_AGG_JOIN_SQL}
              {where_sql}
             ORDER BY f.fund_size DESC NULLS LAST, f.name
             LIMIT %s OFFSET %s
            """,
            params + [inp.limit, inp.offset],
        )
        rows = [dict(r) for r in cur.fetchall()]

    funds = [{
        "fund_id":          r["fund_id"],
        "fund_name":        r["fund_name"],
        "short_name":       r["short_name"],
        "fund_type":        r["fund_type"],
        "fund_status":      r["fund_status"],
        "reporting_status": r["reporting_status"],
        "vintage_year":     r["vintage_year"],
        "capital_committed": {
            "fund_size":         r["fund_size"],
            "sum_of_lp_capital": r["sum_of_lp_capital"],
        },
        "performance": _fund_performance_block(r),
        "n_deals": r["n_deals"] or 0,
    } for r in rows]

    return ToolResult(output={
        "name_query": inp.name,
        "reportable_only": inp.reportable_only,
        "total_matching": total,
        "returned": len(funds),
        "offset": inp.offset,
        "truncated": inp.offset + len(funds) < total,
        "funds": funds,
        "data_note": (
            "capital_committed is what LPs have committed to the fund "
            "(DealCloud's own pre-aggregated totals). performance is what's "
            "actually been called/returned/valued -- see each fund's "
            "performance.source and .note before treating figures as exact."
        ),
    })


class FundStatusInput(BaseModel):
    fund_name: str = Field(
        ...,
        description=(
            "Fund or SPV name (e.g. 'Stonecutter II', 'Pathway I', 'Project "
            "Auto'). Partial/case-insensitive, matches on name or "
            "short_name. Use list_funds first if unsure of the exact name."
        ),
        min_length=1, max_length=200,
    )


@slack_registry.tool(
    "get_fund_status",
    (
        "Full status for ONE fund/SPV: how much capital has been called "
        "from investors, how much has been returned to them, and the most "
        "recent valuation of the fund (NAV) -- PLUS a per-deal breakdown "
        "of every deal the fund holds, each with how much of the fund's "
        "capital went into that deal, how much capital has come back from "
        "it, and the current valuation of the fund's stake in it. Use this "
        "for 'how is fund X doing' / 'what does fund X hold' / 'break down "
        "fund X's deals' questions. The fund-level `performance` block (and "
        "each deal's figures) is sourced from DealCloud's own records where "
        "available, clearly flagged as an approximation ('derived_from_deals') "
        "where it had to be derived by summing deal-level data instead -- "
        "read each block's `note`/`source` before quoting a number as exact. "
        "For a per-LP commitment breakdown instead, use "
        "get_fundraising_summary; for a bulk view across all funds, use "
        "list_funds."
    ),
    FundStatusInput,
)
def get_fund_status(inp: FundStatusInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        matches = _match_funds(cur, inp.fund_name)
        if not matches:
            return ToolResult(output={
                "matched": "none",
                "note": (f"No fund found matching '{inp.fund_name}'. Try "
                         "list_funds to browse all funds."),
            })
        if len(matches) > 1:
            exact = [m for m in matches if inp.fund_name.lower() in (
                m["name"].lower(), (m["short_name"] or "").lower()
            )]
            if len(exact) != 1:
                return ToolResult(output={
                    "matched": "ambiguous",
                    "note": ("Several funds match -- ask the user which one, "
                             "then call again with the exact name."),
                    "candidates": matches,
                })
            matches = exact
        fund = matches[0]

        cur.execute(
            f"""
            SELECT f.id AS fund_id, f.name AS fund_name, f.short_name, f.fund_type,
                   f.fund_status, f.reporting_status, f.vintage_year,
                   f.fund_size, f.sum_of_lp_capital, f.fundraise_target,
                   f.gp_commit_amount, f.count_of_lps,
                   f.first_close_date, f.final_close_date,
                   fp.date_q, fp.nav, fp.cumulative_contributions, fp.cumulative_distributions,
                   agg.n_deals, agg.n_deals_with_invested, agg.sum_invested,
                   agg.n_deals_with_fv, agg.sum_fv,
                   agg.n_deals_with_realized, agg.sum_realized
              FROM dealcloud.fund f
              {_FUND_AGG_JOIN_SQL}
             WHERE f.id = %s
            """,
            (fund["id"],),
        )
        f_row = dict(cur.fetchone())

        cur.execute(
            """
            SELECT d.id AS deal_id, d.name AS deal_name, d.status, d.transaction_type,
                   o.name AS counterparty_name,
                   d.invested_capital, d.fair_value, d.realized_capital,
                   d.total_value_to_invested,
                   iv.valuation_date, iv.fair_value AS iv_fair_value,
                   iv.paid_in_capital, iv.realized_capital_gross,
                   iv.realized_capital_net, iv.gross_irr, iv.net_irr
              FROM dealcloud.deal d
              LEFT JOIN dealcloud.organization o ON o.id = d.organization_id
              LEFT JOIN LATERAL (
                  SELECT valuation_date, fair_value, paid_in_capital,
                         realized_capital_gross, realized_capital_net,
                         gross_irr, net_irr
                    FROM dealcloud.investment_valuation
                   WHERE deal_id = d.id
                   ORDER BY valuation_date DESC NULLS LAST
                   LIMIT 1
              ) iv ON true
             WHERE d.fund_id = %s
             ORDER BY lower(d.name)
            """,
            (fund["id"],),
        )
        deal_rows = [dict(r) for r in cur.fetchall()]

    deals = []
    for d in deal_rows:
        inv = d["invested_capital"]
        capital_invested = abs(inv) if inv is not None else d["paid_in_capital"]
        capital_returned = (d["realized_capital"] if d["realized_capital"] is not None
                            else d["realized_capital_gross"])
        current_stake_value = (d["fair_value"] if d["fair_value"] is not None
                               else d["iv_fair_value"])
        has_data = any(v is not None for v in
                       (capital_invested, capital_returned, current_stake_value))
        deals.append({
            "deal_id": d["deal_id"],
            "deal_name": d["deal_name"],
            "status": d["status"],
            "transaction_type": d["transaction_type"],
            "counterparty_name": d["counterparty_name"],
            "capital_invested": capital_invested,
            "capital_returned": capital_returned,
            "current_stake_value": current_stake_value,
            "valuation_as_of": (d["valuation_date"].date().isoformat()
                               if d["valuation_date"] else None),
            "total_value_to_invested_multiple": d["total_value_to_invested"],
            "gross_irr": d["gross_irr"],
            "net_irr": d["net_irr"],
            "has_financial_data": has_data,
        })

    n_deals = len(deals)
    n_with_data = sum(1 for d in deals if d["has_financial_data"])

    return ToolResult(output={
        "matched": "fund",
        "fund": {
            "fund_id": f_row["fund_id"],
            "fund_name": f_row["fund_name"],
            "short_name": f_row["short_name"],
            "fund_type": f_row["fund_type"],
            "fund_status": f_row["fund_status"],
            "reporting_status": f_row["reporting_status"],
            "vintage_year": f_row["vintage_year"],
            "first_close_date": (f_row["first_close_date"].date().isoformat()
                                 if f_row["first_close_date"] else None),
            "final_close_date": (f_row["final_close_date"].date().isoformat()
                                 if f_row["final_close_date"] else None),
        },
        "capital_committed": {
            "fund_size": f_row["fund_size"],
            "sum_of_lp_capital": f_row["sum_of_lp_capital"],
            "fundraise_target": f_row["fundraise_target"],
            "gp_commit_amount": f_row["gp_commit_amount"],
            "count_of_lps": f_row["count_of_lps"],
            "note": ("What LPs have committed to the fund -- see "
                     "get_fundraising_summary for a per-LP breakdown."),
        },
        "performance": _fund_performance_block(f_row),
        "deals": deals,
        "deal_data_completeness": (
            f"{n_with_data}/{n_deals} of this fund's linked deals have at "
            "least one financial figure (capital invested/returned/current "
            "value) recorded in DealCloud."
            if n_deals else
            "This fund has no deals linked via deal.fund_id in DealCloud."
        ),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Email correspondence (dce's email pipeline, see memory:
# deal_cloud_email_pipeline). dealcloud.email_thread holds ONLY promoted,
# deal-related threads, each with an LLM summary -- raw bodies never leave
# dce's transient staging table, so there is nothing more to read than what
# these tools return. The get_org_* tools above already roll email into
# per-org aggregates; these three are the thread- and person-level view:
# search threads, open one, and look a correspondent up by name/address.
# ---------------------------------------------------------------------------

EmailCategory = Literal[
    "inbound_deal_inquiry", "outbound_outreach", "ongoing_deal_discussion",
    "portfolio_update", "deal_admin",
]

_EMAIL_PREVIEW_CHARS = 600
_EMAIL_SEARCH_INTERNAL_CAP = 5
_EMAIL_SEARCH_EXTERNAL_CAP = 8
_EMAIL_SEARCH_ORG_CAP = 6
_EMAIL_READ_PARTICIPANT_CAP = 60
_EMAIL_READ_MESSAGE_CAP = 40
_EMAIL_CONTACT_ION_CAP = 5
_EMAIL_CONTACT_ORG_CAP = 3

_TERM_RE = re.compile(r'"([^"]+)"|(\S+)')


def _like(term: str) -> str:
    """ILIKE pattern for a literal substring (escapes %, _ and backslash)."""
    esc = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


def _split_terms(text: str | None) -> list[str]:
    """Whitespace-split, keeping "quoted phrases" together."""
    if not text:
        return []
    return [a or b for a, b in _TERM_RE.findall(text) if (a or b).strip()]


def _email_direction(row: dict) -> str:
    """Same badges dce's Emails tab shows."""
    if row.get("was_initiated_internally"):
        return "we -> them" if row.get("has_external_response") else "we wrote, no reply"
    if row.get("was_initiated_internally") is False:
        return "they -> us" if row.get("has_internal_response") else "they wrote, no reply"
    return "unknown"


def _email_coverage(cur) -> dict:
    cur.execute(
        "SELECT MIN(last_message_at)::date AS earliest, "
        "       MAX(last_message_at)::date AS latest, COUNT(*) AS threads "
        "  FROM dealcloud.email_thread"
    )
    return dict(cur.fetchone())


def _email_note(coverage: dict) -> str:
    # Lives in the RESPONSE, not the description: the window moves every
    # weekly sync (see chat_mcp_tools' "Writing tool descriptions").
    return (
        f"Email index covers {coverage['threads']} deal-related threads "
        f"from {coverage['earliest']} to {coverage['latest']}. Only threads "
        "an LLM triage judged deal-related were kept (personal, newsletter, "
        "vendor, recruiter and internal-ops mail is dropped), and only a "
        "summary is stored -- never the email body. So 'no thread found' "
        "means no deal-related thread was indexed in that window, not proof "
        "that nobody ever emailed them; older contact may still show up in "
        "get_org_communication_timeline (DealCloud activities, calendar), "
        "and logged call/meeting notes in search_dealcloud_activities."
    )


class SearchEmailThreadsInput(BaseModel):
    query: str | None = Field(
        None,
        max_length=200,
        description=(
            "Keywords matched literally (case-insensitive) against each "
            "thread's subject and summary. Every word must appear; wrap a "
            "phrase in double quotes to match it as-is, e.g. "
            "'\"term sheet\" Series B'. Not semantic -- use the words a "
            "summary would actually contain."
        ),
    )
    org_ids: list[int] = Field(
        default_factory=list, max_length=20,
        description=(
            "Restrict to threads attributed to these orgs. Pass canonical "
            "heads from find_organizations / bundle_via_supersede. Not every "
            "thread is attributed to an org, so if this comes back thin, "
            "retry with `participant` set to the org's email domain."
        ),
    )
    participant: str | None = Field(
        None, max_length=200,
        description=(
            "An EXTERNAL correspondent on the thread: a name, an email "
            "address, or a domain ('sequoiacap.com'). Substring match."
        ),
    )
    ion_person: str | None = Field(
        None, max_length=200,
        description=(
            "An Ion Pacific person on the thread: a name or "
            "@ionpacific.com address. Substring match."
        ),
    )
    categories: list[EmailCategory] = Field(
        default_factory=list,
        description=(
            "Restrict to these triage categories. Empty = all. "
            "inbound_deal_inquiry (they pitched/asked us), outbound_outreach "
            "(we reached out), ongoing_deal_discussion, portfolio_update, "
            "deal_admin (legal/fund-admin/closing logistics)."
        ),
    )
    since: date | None = Field(
        None, description="Only threads active on or after this date (YYYY-MM-DD)."
    )
    until: date | None = Field(
        None, description="Only threads that started on or before this date (YYYY-MM-DD)."
    )
    limit: int = Field(15, ge=1, le=50, description="Max threads to return.")
    offset: int = Field(0, ge=0, description="For paging through `total_matches`.")


class EmailThreadIdInput(BaseModel):
    thread_id: int = Field(
        ..., description="dealcloud.email_thread.id, from search_email_threads or get_org_dossier."
    )


class FindEmailContactsInput(BaseModel):
    query: str = Field(
        ..., min_length=2, max_length=200,
        description=(
            "A person's name, an email address, or a domain "
            "('@a16z.com', 'a16z.com'). Every word must match the "
            "correspondent's name or address (substring, case-insensitive)."
        ),
    )
    include_internal: bool = Field(
        False,
        description="Also return Ion Pacific people. Default: external correspondents only.",
    )
    limit: int = Field(15, ge=1, le=50, description="Max people to return.")


@slack_registry.tool(
    "search_email_threads",
    (
        "Search Ion Pacific's indexed EMAIL correspondence -- deal-related "
        "threads from Ion staff mailboxes, each stored as an LLM summary "
        "with its participants and the orgs it concerns. Use for 'what did "
        "we discuss with X', 'when did we last email them about Y', 'who "
        "pitched us on Z', 'which threads mention a term sheet', or 'what "
        "has <Ion person> been corresponding with <company> about'. "
        "Combine any of: keywords, org_ids, an external participant "
        "(name/address/domain), an Ion person, categories and a date "
        "range; at least one filter is required. Newest first, with "
        "`total_matches` for paging. Each row has thread_id, subject, "
        "dates, direction, category, a summary preview, the Ion and "
        "external participants, and the attributed orgs. Call "
        "read_email_thread for the full summary and message timeline. "
        "For an org-level overview prefer get_org_communication_timeline "
        "/ get_org_their_contacts; this tool is for the threads themselves. "
        "Read-only."
    ),
    SearchEmailThreadsInput,
)
def search_email_threads(inp: SearchEmailThreadsInput, ctx: dict) -> ToolResult:
    terms = _split_terms(inp.query)
    if not (terms or inp.org_ids or inp.participant or inp.ion_person
            or inp.categories or inp.since or inp.until):
        return ToolResult(output=(
            "Give at least one filter: query, org_ids, participant, "
            "ion_person, categories, since or until."
        ))

    where, params = [], []
    for t in terms:
        where.append(
            "(COALESCE(et.subject, '') || ' ' || COALESCE(et.summary, '')) ILIKE %s"
        )
        params.append(_like(t))
    if inp.org_ids:
        where.append(
            "EXISTS (SELECT 1 FROM dealcloud.organization_entity oe "
            "         WHERE oe.entity_type = 'email_thread' "
            "           AND oe.entity_id = et.id "
            "           AND oe.organization_id = ANY(%s))"
        )
        params.append(inp.org_ids)
    for who, internal in ((inp.participant, False), (inp.ion_person, True)):
        if who and who.strip():
            where.append(
                "EXISTS (SELECT 1 FROM dealcloud.email_thread_participant p "
                "         WHERE p.thread_id = et.id AND p.is_internal = %s "
                "           AND (p.email ILIKE %s OR p.name ILIKE %s))"
            )
            pat = _like(who.strip())
            params += [internal, pat, pat]
    if inp.categories:
        where.append("et.category = ANY(%s)")
        params.append(list(inp.categories))
    if inp.since:
        where.append("et.last_message_at >= %s")
        params.append(inp.since)
    if inp.until:
        where.append("et.first_message_at < %s::date + 1")
        params.append(inp.until)

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT et.id AS thread_id, et.subject,
                   et.first_message_at::date AS first_message_at,
                   et.last_message_at::date  AS last_message_at,
                   et.message_count, et.category,
                   et.was_initiated_internally, et.has_internal_response,
                   et.has_external_response,
                   LEFT(COALESCE(et.summary, ''), %s) AS summary_preview,
                   LENGTH(COALESCE(et.summary, '')) > %s AS summary_truncated,
                   COUNT(*) OVER () AS total_matches
              FROM dealcloud.email_thread et
             WHERE {' AND '.join(where)}
             ORDER BY et.last_message_at DESC, et.id DESC
             LIMIT %s OFFSET %s
            """,
            [_EMAIL_PREVIEW_CHARS, _EMAIL_PREVIEW_CHARS, *params,
             inp.limit, inp.offset],
        )
        rows = [dict(r) for r in cur.fetchall()]
        total = rows[0]["total_matches"] if rows else 0
        ids = [r["thread_id"] for r in rows]

        people: dict[int, dict[str, list]] = {}
        orgs: dict[int, list] = {}
        if ids:
            cur.execute(
                """
                SELECT thread_id, email, name, is_internal
                  FROM dealcloud.email_thread_participant
                 WHERE thread_id = ANY(%s)
                 ORDER BY thread_id, is_initiator DESC, message_count DESC, email
                """,
                (ids,),
            )
            for p in cur.fetchall():
                side = "ion" if p["is_internal"] else "external"
                bucket = people.setdefault(p["thread_id"], {"ion": [], "external": []})[side]
                cap = _EMAIL_SEARCH_INTERNAL_CAP if p["is_internal"] else _EMAIL_SEARCH_EXTERNAL_CAP
                if len(bucket) < cap:
                    bucket.append(
                        f"{p['name']} <{p['email']}>" if p["name"] else p["email"]
                    )
            cur.execute(
                """
                SELECT DISTINCT ON (oe.entity_id, o.id)
                       oe.entity_id AS thread_id, o.id AS org_id, o.name,
                       oe.relationship_type
                  FROM dealcloud.organization_entity oe
                  JOIN dealcloud.organization o ON o.id = oe.organization_id
                 WHERE oe.entity_type = 'email_thread' AND oe.entity_id = ANY(%s)
                 ORDER BY oe.entity_id, o.id
                """,
                (ids,),
            )
            for o in cur.fetchall():
                lst = orgs.setdefault(o["thread_id"], [])
                if len(lst) < _EMAIL_SEARCH_ORG_CAP:
                    lst.append({"org_id": o["org_id"], "name": o["name"],
                                "relationship": o["relationship_type"]})
        coverage = _email_coverage(cur)

    results = []
    for r in rows:
        p = people.get(r["thread_id"], {"ion": [], "external": []})
        results.append({
            "thread_id":        r["thread_id"],
            "subject":          r["subject"],
            "first_message_at": r["first_message_at"],
            "last_message_at":  r["last_message_at"],
            "message_count":    r["message_count"],
            "direction":        _email_direction(r),
            "category":         r["category"],
            "summary_preview":  r["summary_preview"]
                                + ("..." if r["summary_truncated"] else ""),
            "ion_participants":      p["ion"],
            "external_participants": p["external"],
            "orgs":             orgs.get(r["thread_id"], []),
        })
    return ToolResult(output={
        "total_matches": total,
        "returned": len(results),
        "offset": inp.offset,
        "results": results,
        "note": _email_note(coverage),
    })


@slack_registry.tool(
    "read_email_thread",
    (
        "Open ONE email thread: its full LLM summary, every participant "
        "(Ion vs external, who started it, how many messages each sent), "
        "the orgs it concerns with the role each plays and why, and the "
        "message-by-message timeline (who sent what when, inbound or "
        "outbound). This is everything stored about a thread -- email "
        "bodies are not kept, so if the summary doesn't answer the "
        "question, say so rather than guessing at what was written. "
        "Read-only."
    ),
    EmailThreadIdInput,
)
def read_email_thread(inp: EmailThreadIdInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id AS thread_id, subject, first_message_at, last_message_at,
                   message_count, category, was_initiated_internally,
                   has_internal_response, has_external_response, summary
              FROM dealcloud.email_thread WHERE id = %s
            """,
            (inp.thread_id,),
        )
        t = cur.fetchone()
        if t is None:
            return ToolResult(output=f"Email thread {inp.thread_id} not found.")
        t = dict(t)

        cur.execute(
            """
            SELECT email, NULLIF(name, '') AS name, is_internal AS is_ion,
                   is_initiator,
                   message_count, contact_id IS NOT NULL AS in_dealcloud
              FROM dealcloud.email_thread_participant
             WHERE thread_id = %s
             ORDER BY is_initiator DESC, is_internal, message_count DESC, email
            """,
            (inp.thread_id,),
        )
        participants = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT DISTINCT ON (o.id)
                   o.id AS org_id, o.name, oe.relationship_type, oe.context
              FROM dealcloud.organization_entity oe
              JOIN dealcloud.organization o ON o.id = oe.organization_id
             WHERE oe.entity_type = 'email_thread' AND oe.entity_id = %s
             ORDER BY o.id, oe.context IS NULL
            """,
            (inp.thread_id,),
        )
        thread_orgs = [dict(r) for r in cur.fetchall()]

        # The same message is stored once per mailbox it sits in (sender's
        # Sent + each Ion recipient's Inbox), so collapse on internet_msg_id.
        cur.execute(
            """
            SELECT sent_at, sender_email, sender_is_internal, direction
              FROM (
                SELECT DISTINCT ON (COALESCE(internet_msg_id, graph_message_id))
                       sent_at, sender_email, sender_is_internal, direction
                  FROM dealcloud.email_message
                 WHERE thread_id = %s
                 ORDER BY COALESCE(internet_msg_id, graph_message_id),
                          (folder ILIKE 'sent%%') DESC
              ) m
             ORDER BY sent_at
            """,
            (inp.thread_id,),
        )
        messages = [dict(r) for r in cur.fetchall()]
        coverage = _email_coverage(cur)

    names = {p["email"].lower(): p["name"] for p in participants if p["name"]}
    shown = messages[-_EMAIL_READ_MESSAGE_CAP:]
    timeline = [{
        "sent_at":    m["sent_at"],
        "from":       names.get((m["sender_email"] or "").lower(), m["sender_email"]),
        "from_email": m["sender_email"],
        "from_ion":   m["sender_is_internal"],
        "direction":  m["direction"],
    } for m in shown]

    return ToolResult(output={
        "thread_id":        t["thread_id"],
        "subject":          t["subject"],
        "first_message_at": t["first_message_at"],
        "last_message_at":  t["last_message_at"],
        "message_count":    t["message_count"],
        "direction":        _email_direction(t),
        "category":         t["category"],
        "summary":          t["summary"] or "(no summary)",
        "participants":     participants[:_EMAIL_READ_PARTICIPANT_CAP],
        "participants_omitted": max(0, len(participants) - _EMAIL_READ_PARTICIPANT_CAP),
        "orgs":             thread_orgs,
        "timeline":         timeline,
        "timeline_omitted_earlier": len(messages) - len(shown),
        "note":             _email_note(coverage),
    })


@slack_registry.tool(
    "find_email_contacts",
    (
        "Look up PEOPLE Ion Pacific has corresponded with by email, by "
        "name, address or domain -- 'have we been in touch with Jane Doe', "
        "'who have we emailed at a16z.com', 'what's John's email at Acme'. "
        "For each person returns their address and name, how many indexed "
        "threads they appear on and when (first/last), which Ion people "
        "they corresponded with most, and the orgs they're associated "
        "with (job title, relationship strength, whether they're a "
        "DealCloud contact). An org with domain_match=true is one their "
        "email domain belongs to -- their employer; the rest are orgs "
        "their threads were about, so don't present those as where the "
        "person works. Ranked by number of threads. Pass a "
        "person's address to search_email_threads(participant=...) to see "
        "the threads themselves. When you already have an org_id, "
        "get_org_their_contacts is the org-centric version. Read-only."
    ),
    FindEmailContactsInput,
)
def find_email_contacts(inp: FindEmailContactsInput, ctx: dict) -> ToolResult:
    terms = [t.lstrip("@") for t in _split_terms(inp.query)]
    terms = [t for t in terms if t]
    if not terms:
        return ToolResult(output="Give a name, email address or domain to look up.")
    where = ["(p.email ILIKE %s OR p.name ILIKE %s)"] * len(terms)
    params: list[Any] = [x for t in terms for x in (_like(t), _like(t))]
    if not inp.include_internal:
        where.append("NOT p.is_internal")

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            WITH m AS (
                SELECT LOWER(p.email) AS email, p.name, p.is_internal,
                       p.thread_id, p.message_count
                  FROM dealcloud.email_thread_participant p
                 WHERE {' AND '.join(where)}
            )
            SELECT m.email,
                   MODE() WITHIN GROUP (ORDER BY NULLIF(m.name, '')) AS name,
                   BOOL_OR(m.is_internal)         AS is_ion,
                   COUNT(DISTINCT m.thread_id)    AS thread_count,
                   SUM(m.message_count)           AS messages_sent,
                   MIN(et.first_message_at)::date AS first_thread_at,
                   MAX(et.last_message_at)::date  AS last_thread_at,
                   COUNT(*) OVER ()               AS total_matches
              FROM m JOIN dealcloud.email_thread et ON et.id = m.thread_id
             GROUP BY m.email
             ORDER BY thread_count DESC, last_thread_at DESC
             LIMIT %s
            """,
            [*params, inp.limit],
        )
        people = [dict(r) for r in cur.fetchall()]
        emails = [p["email"] for p in people]

        ion_by: dict[str, list] = {}
        orgs_by: dict[str, list] = {}
        if emails:
            cur.execute(
                """
                SELECT LOWER(c.email) AS contact,
                       LOWER(i.email) AS ion_email,
                       MAX(NULLIF(i.name, '')) AS ion_name,
                       COUNT(DISTINCT i.thread_id) AS shared_threads,
                       MAX(et.last_message_at)::date AS last_shared_at
                  FROM dealcloud.email_thread_participant c
                  JOIN dealcloud.email_thread_participant i
                    ON i.thread_id = c.thread_id AND i.is_internal
                   AND LOWER(i.email) <> LOWER(c.email)
                  JOIN dealcloud.email_thread et ON et.id = c.thread_id
                 WHERE LOWER(c.email) = ANY(%s)
                 GROUP BY 1, 2
                 ORDER BY 1, shared_threads DESC, last_shared_at DESC
                """,
                (emails,),
            )
            for r in cur.fetchall():
                lst = ion_by.setdefault(r["contact"], [])
                if len(lst) < _EMAIL_CONTACT_ION_CAP:
                    lst.append({
                        "name": r["ion_name"], "email": r["ion_email"],
                        "shared_threads": r["shared_threads"],
                        "last_shared_at": r["last_shared_at"],
                    })
            cur.execute(
                """
                SELECT LOWER(a.email) AS contact, o.id AS org_id, o.name,
                       a.job_title, a.derived_connection_strength,
                       a.is_deal_cloud_contact,
                       EXISTS (SELECT 1 FROM dealcloud.domain_organization d
                                WHERE d.organization_id = a.organization_id
                                  AND d.domain = SPLIT_PART(LOWER(a.email), '@', 2)
                              ) AS domain_match
                  FROM dealcloud.email_organization_agg a
                  JOIN dealcloud.organization o ON o.id = a.organization_id
                 WHERE LOWER(a.email) = ANY(%s)
                 ORDER BY 1, domain_match DESC, a.email_thread_count DESC,
                          a.communication_count DESC
                """,
                (emails,),
            )
            for r in cur.fetchall():
                lst = orgs_by.setdefault(r["contact"], [])
                if len(lst) < _EMAIL_CONTACT_ORG_CAP:
                    lst.append({
                        "org_id": r["org_id"], "name": r["name"],
                        "job_title": r["job_title"],
                        "connection_strength": r["derived_connection_strength"],
                        "in_dealcloud": r["is_deal_cloud_contact"],
                        "domain_match": r["domain_match"],
                    })
        coverage = _email_coverage(cur)

    total = people[0]["total_matches"] if people else 0
    for p in people:
        p.pop("total_matches")
        p["ion_counterparts"] = [] if p["is_ion"] else ion_by.get(p["email"], [])
        p["orgs"] = orgs_by.get(p["email"], [])
    return ToolResult(output={
        "query": inp.query,
        "total_matches": total,
        "results": people,
        "note": _email_note(coverage),
    })


# ---------------------------------------------------------------------------
# DealCloud Activities (the Interaction entity, 5341): calls, meetings and
# other touchpoints Ion staff logged by hand, with their free-text Notes.
# dce's sync.py::sync_communications already lands them in
# dealcloud.communication (dc_entity_type='interaction'); these two tools are
# the first to expose the Notes text rather than just counts. Emails saved
# to DealCloud (entity 5340, dc_entity_type='email') share that table with
# their full Outlook-HTML body in `notes`, and are included: anything filed
# in DealCloud is already visible firm-wide, unlike the mailbox index the
# email tools above read (summaries only).
# ---------------------------------------------------------------------------

_ACTIVITY_PREVIEW_CHARS = 600
_ACTIVITY_SEARCH_PEOPLE_CAP = 6
_ACTIVITY_SEARCH_ORG_CAP = 6
_ACTIVITY_READ_NOTES_CAP = 60_000
# Raw chars fetched per search row before HTML->text; an email body's first
# 600 chars of HTML can be all markup, so the preview needs more than that.
_ACTIVITY_SEARCH_FETCH_CHARS = 20_000
_ACTIVITY_KINDS = {"activity": "interaction", "email": "email"}

# A tag name must end at whitespace, '/' or '>': DealCloud email bodies carry
# raw "Jane Doe <jane@x.com>" headers that a looser pattern would delete.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:-]*(?=[\s/>])[^<>]*>|<![^<>]*>")
_HTML_DROP_RE = re.compile(r"<(style|script|head)\b.*?</\1\s*>|<!--.*?-->",
                           re.I | re.S)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>|</(p|div|tr|li|h[1-6]|table|blockquote)\s*>",
                            re.I)


def _notes_text(raw: str | None) -> str:
    """Notes as plain text. Interaction notes are already text/markdown and
    pass through; DealCloud email bodies are Outlook HTML."""
    s = (raw or "").strip()
    if not _HTML_TAG_RE.search(s):
        return s
    s = _HTML_DROP_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)  # source line wraps are not breaks in HTML
    s = _HTML_BREAK_RE.sub("\n", s)
    s = re.sub(r"<li\b[^>]*>", "- ", s, flags=re.I)
    s = _HTML_TAG_RE.sub("", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


class SearchActivitiesInput(BaseModel):
    kinds: list[Literal["activity", "email"]] = Field(
        default_factory=list,
        description=(
            "'activity' = calls/meetings/other logged touchpoints with their "
            "Notes; 'email' = emails filed to DealCloud, with the full body. "
            "Empty = both."
        ),
    )
    query: str | None = Field(
        None, max_length=300,
        description=(
            "Keywords matched against the subject AND notes / email body. Every "
            "word must match (case-insensitive substring); wrap a phrase in "
            "double quotes to keep it together, e.g. 'continuation \"term sheet\"'."
        ),
    )
    org_ids: list[int] = Field(
        default_factory=list, max_length=20,
        description=(
            "Restrict to activities linked to these orgs. Pass canonical "
            "heads from find_organizations / bundle_via_supersede."
        ),
    )
    deal_ids: list[int] = Field(
        default_factory=list, max_length=20,
        description=(
            "Restrict to activities linked to these dealcloud.deal ids. Few "
            "activities carry a deal link in DealCloud, so prefer org_ids "
            "(the deal's counterparty / underlying company) unless this "
            "returns results."
        ),
    )
    ion_person: str | None = Field(
        None, max_length=200,
        description="An Ion Pacific attendee: a name or @ionpacific.com address. Substring match.",
    )
    contact: str | None = Field(
        None, max_length=200,
        description="An EXTERNAL attendee (DealCloud contact): a name or email address. Substring match.",
    )
    types: list[str] = Field(
        default_factory=list, max_length=10,
        description="Restrict to these activity types, e.g. ['Call', 'Meeting']. Empty = all.",
    )
    with_notes_only: bool = Field(
        True,
        description=(
            "Default true: only activities that have Notes text. Set false to "
            "also list bare log entries (subject/date/attendees, no notes)."
        ),
    )
    since: date | None = Field(None, description="Only activities on or after this date (YYYY-MM-DD).")
    until: date | None = Field(None, description="Only activities on or before this date (YYYY-MM-DD).")
    limit: int = Field(15, ge=1, le=50, description="Max activities to return.")
    offset: int = Field(0, ge=0, description="For paging through `total_matches`.")


class ActivityIdInput(BaseModel):
    activity_id: int = Field(
        ..., description="activity_id from search_dealcloud_activities (dealcloud.communication.id).",
    )


def _activity_people(cur, ids: list[int]) -> tuple[dict, dict]:
    """(ion attendees, external contacts) per activity id, as display strings."""
    ion: dict[int, list[str]] = {}
    ext: dict[int, list[str]] = {}
    if not ids:
        return ion, ext
    cur.execute(
        """
        SELECT communication_id, employee_email, employee_name
          FROM dealcloud.communication_employee
         WHERE communication_id = ANY(%s)
         ORDER BY communication_id, employee_name, employee_email
        """,
        (ids,),
    )
    for r in cur.fetchall():
        ion.setdefault(r["communication_id"], []).append(
            r["employee_name"] or r["employee_email"])
    cur.execute(
        """
        SELECT cc.communication_id,
               NULLIF(TRIM(CONCAT_WS(' ', ct.first_name, ct.last_name)), '') AS name,
               ct.email, o.name AS org_name
          FROM dealcloud.communication_contact cc
          JOIN dealcloud.contact ct ON ct.id = cc.contact_id
          LEFT JOIN dealcloud.organization o ON o.id = ct.organization_id
         WHERE cc.communication_id = ANY(%s)
         ORDER BY cc.communication_id, ct.last_name, ct.first_name
        """,
        (ids,),
    )
    for r in cur.fetchall():
        label = r["name"] or r["email"] or "(unnamed contact)"
        if r["org_name"]:
            label = f"{label} ({r['org_name']})"
        ext.setdefault(r["communication_id"], []).append(label)
    return ion, ext


def _activity_coverage(cur) -> dict:
    cur.execute(
        """
        SELECT dc_entity_type AS kind,
               COUNT(*) FILTER (WHERE OCTET_LENGTH(notes) > 0) AS with_notes,
               COUNT(*) AS total,
               (MIN(date) FILTER (WHERE OCTET_LENGTH(notes) > 0))::date AS earliest,
               (MAX(date) FILTER (WHERE OCTET_LENGTH(notes) > 0))::date AS latest,
               MAX(last_synced_at)::date AS synced
          FROM dealcloud.communication
         WHERE dc_entity_type IN ('interaction', 'email')
         GROUP BY dc_entity_type
        """
    )
    return {r["kind"]: dict(r) for r in cur.fetchall()}


def _activity_note(cov: dict) -> str:
    # In the RESPONSE, not the description: the numbers move every sync.
    a = cov.get("interaction") or {}
    e = cov.get("email") or {}
    return (
        f"DealCloud holds {a.get('total', 0)} logged activities, "
        f"{a.get('with_notes', 0)} with Notes ({a.get('earliest')} to "
        f"{a.get('latest')}), and {e.get('total', 0)} filed emails "
        f"({e.get('earliest')} to {e.get('latest')}); last synced "
        f"{a.get('synced') or e.get('synced')}. Activity notes are whatever "
        "the Ion person typed or pasted -- one attendee's account, not a "
        "transcript. Only emails someone chose to file are here; "
        "search_email_threads covers the wider mailbox index. Not every "
        "call, meeting or email gets logged, so 'nothing found' is not "
        "proof there was no contact."
    )


@slack_registry.tool(
    "search_dealcloud_activities",
    (
        "Search what Ion staff logged in DealCloud: Activities (calls, "
        "meetings, other touchpoints) INCLUDING their free-text Notes (call "
        "notes, meeting write-ups, diligence takeaways), and emails filed to "
        "DealCloud with their FULL body text. Use for 'what "
        "did we hear on the last call with X', 'what notes do we have on "
        "Y', 'who at Ion met Z and what was said', 'which calls discussed a "
        "continuation vehicle', 'what did they write when they sent the "
        "deck'. Combine any of: kinds (activity / email), keywords (matched "
        "in subject + notes/body), org_ids, deal_ids, an Ion attendee or "
        "recipient, an external contact, types and a date range; at least "
        "one filter besides kinds is required. "
        "Newest first, with `total_matches` for paging. Each row has "
        "activity_id, kind, subject, type, date, attendees, linked "
        "orgs/deals and a notes/body preview; call read_dealcloud_activity "
        "for the full text. For the wider (summary-only) mailbox index use "
        "search_email_threads; for SharePoint files, search_documents. "
        "Read-only."
    ),
    SearchActivitiesInput,
)
def search_dealcloud_activities(inp: SearchActivitiesInput, ctx: dict) -> ToolResult:
    terms = _split_terms(inp.query)
    if not (terms or inp.org_ids or inp.deal_ids or inp.ion_person
            or inp.contact or inp.types or inp.since or inp.until):
        return ToolResult(output=(
            "Give at least one filter: query, org_ids, deal_ids, ion_person, "
            "contact, types, since or until."
        ))

    where = ["c.dc_entity_type = ANY(%s)"]
    params: list[Any] = [[_ACTIVITY_KINDS[k] for k in (inp.kinds or _ACTIVITY_KINDS)]]
    if inp.with_notes_only:
        where.append("OCTET_LENGTH(c.notes) > 0")
    # Not subject || notes: concatenating copies every (TOASTed) email body.
    for t in terms:
        where.append("(c.subject ILIKE %s OR c.notes ILIKE %s)")
        params += [_like(t), _like(t)]
    if inp.org_ids:
        where.append(
            "EXISTS (SELECT 1 FROM dealcloud.organization_entity oe "
            "         WHERE oe.entity_type = 'communication' "
            "           AND oe.entity_id = c.id "
            "           AND oe.organization_id = ANY(%s))"
        )
        params.append(inp.org_ids)
    if inp.deal_ids:
        where.append(
            "EXISTS (SELECT 1 FROM dealcloud.communication_deal cd "
            "         WHERE cd.communication_id = c.id AND cd.deal_id = ANY(%s))"
        )
        params.append(inp.deal_ids)
    if inp.ion_person and inp.ion_person.strip():
        pat = _like(inp.ion_person.strip())
        where.append(
            "EXISTS (SELECT 1 FROM dealcloud.communication_employee ce "
            "         WHERE ce.communication_id = c.id "
            "           AND (ce.employee_email ILIKE %s OR ce.employee_name ILIKE %s))"
        )
        params += [pat, pat]
    if inp.contact and inp.contact.strip():
        pat = _like(inp.contact.strip())
        where.append(
            "EXISTS (SELECT 1 FROM dealcloud.communication_contact cc "
            "          JOIN dealcloud.contact ct ON ct.id = cc.contact_id "
            "         WHERE cc.communication_id = c.id "
            "           AND (ct.email ILIKE %s "
            "                OR CONCAT_WS(' ', ct.first_name, ct.last_name) ILIKE %s))"
        )
        params += [pat, pat]
    if inp.types:
        where.append("LOWER(c.communication_type) = ANY(%s)")
        params.append([t.lower() for t in inp.types])
    if inp.since:
        where.append("c.date >= %s")
        params.append(inp.since)
    if inp.until:
        where.append("c.date < %s::date + 1")
        params.append(inp.until)

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT c.id AS activity_id, c.subject,
                   COALESCE(c.communication_type, 'Unspecified') AS type,
                   c.dc_entity_type, c.date::date AS date,
                   LEFT(COALESCE(c.notes, ''), %s) AS notes_head,
                   LENGTH(COALESCE(c.notes, '')) > %s AS notes_cut,
                   COUNT(*) OVER () AS total_matches
              FROM dealcloud.communication c
             WHERE {' AND '.join(where)}
             ORDER BY c.date DESC NULLS LAST, c.id DESC
             LIMIT %s OFFSET %s
            """,
            [_ACTIVITY_SEARCH_FETCH_CHARS, _ACTIVITY_SEARCH_FETCH_CHARS,
             *params, inp.limit, inp.offset],
        )
        rows = [dict(r) for r in cur.fetchall()]
        total = rows[0]["total_matches"] if rows else 0
        ids = [r["activity_id"] for r in rows]

        ion, ext = _activity_people(cur, ids)
        orgs: dict[int, list] = {}
        deals: dict[int, list] = {}
        if ids:
            cur.execute(
                """
                SELECT DISTINCT oe.entity_id AS activity_id, o.id AS org_id, o.name
                  FROM dealcloud.organization_entity oe
                  JOIN dealcloud.organization o ON o.id = oe.organization_id
                 WHERE oe.entity_type = 'communication' AND oe.entity_id = ANY(%s)
                 ORDER BY 1, 2
                """,
                (ids,),
            )
            for o in cur.fetchall():
                lst = orgs.setdefault(o["activity_id"], [])
                if len(lst) < _ACTIVITY_SEARCH_ORG_CAP:
                    lst.append({"org_id": o["org_id"], "name": o["name"]})
            cur.execute(
                """
                SELECT cd.communication_id AS activity_id, d.id AS deal_id, d.name
                  FROM dealcloud.communication_deal cd
                  JOIN dealcloud.deal d ON d.id = cd.deal_id
                 WHERE cd.communication_id = ANY(%s)
                """,
                (ids,),
            )
            for d in cur.fetchall():
                deals.setdefault(d["activity_id"], []).append(
                    {"deal_id": d["deal_id"], "name": d["name"]})
        coverage = _activity_coverage(cur)

    results = []
    for r in rows:
        aid = r["activity_id"]
        text = _notes_text(r["notes_head"])
        truncated = r["notes_cut"] or len(text) > _ACTIVITY_PREVIEW_CHARS
        results.append({
            "activity_id":   aid,
            "kind":          "email" if r["dc_entity_type"] == "email" else "activity",
            "subject":       r["subject"],
            "type":          r["type"],
            "date":          r["date"],
            "ion_attendees": ion.get(aid, [])[:_ACTIVITY_SEARCH_PEOPLE_CAP],
            "external_attendees": ext.get(aid, [])[:_ACTIVITY_SEARCH_PEOPLE_CAP],
            "orgs":          orgs.get(aid, []),
            "deals":         deals.get(aid, []),
            "notes_preview": (text[:_ACTIVITY_PREVIEW_CHARS]
                              + ("..." if truncated else "")) or "(no notes)",
        })
    return ToolResult(output={
        "total_matches": total,
        "returned": len(results),
        "offset": inp.offset,
        "results": results,
        "note": _activity_note(coverage),
    })


@slack_registry.tool(
    "read_dealcloud_activity",
    (
        "Open ONE DealCloud Activity (call / meeting / other logged "
        "touchpoint) or filed email: its complete Notes / email body as "
        "text, type, date, every Ion and "
        "external attendee, and the orgs and deals it is linked to. Use "
        "after search_dealcloud_activities surfaces a relevant activity_id. "
        "The notes are one attendee's own write-up -- attribute claims to "
        "them rather than stating them as fact. Read-only."
    ),
    ActivityIdInput,
)
def read_dealcloud_activity(inp: ActivityIdInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id AS activity_id, dc_id, dc_entity_type, subject,
                   COALESCE(communication_type, 'Unspecified') AS type,
                   date, notes, last_synced_at
              FROM dealcloud.communication
             WHERE id = %s AND dc_entity_type IN ('interaction', 'email')
            """,
            (inp.activity_id,),
        )
        a = cur.fetchone()
        if a is None:
            return ToolResult(output=f"DealCloud activity {inp.activity_id} not found.")
        a = dict(a)
        ion, ext = _activity_people(cur, [inp.activity_id])
        cur.execute(
            """
            SELECT DISTINCT o.id AS org_id, o.name
              FROM dealcloud.organization_entity oe
              JOIN dealcloud.organization o ON o.id = oe.organization_id
             WHERE oe.entity_type = 'communication' AND oe.entity_id = %s
             ORDER BY o.id
            """,
            (inp.activity_id,),
        )
        linked_orgs = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT d.id AS deal_id, d.name, d.status
              FROM dealcloud.communication_deal cd
              JOIN dealcloud.deal d ON d.id = cd.deal_id
             WHERE cd.communication_id = %s
            """,
            (inp.activity_id,),
        )
        linked_deals = [dict(r) for r in cur.fetchall()]

    notes = _notes_text(a["notes"])
    truncated = len(notes) > _ACTIVITY_READ_NOTES_CAP
    return ToolResult(output={
        "activity_id":   a["activity_id"],
        "kind":          "email" if a["dc_entity_type"] == "email" else "activity",
        "dealcloud_entry_id": a["dc_id"],
        "subject":       a["subject"],
        "type":          a["type"],
        "date":          a["date"],
        "ion_attendees": ion.get(inp.activity_id, []),
        "external_attendees": ext.get(inp.activity_id, []),
        "orgs":          linked_orgs,
        "deals":         linked_deals,
        "notes":         (notes[:_ACTIVITY_READ_NOTES_CAP] or "(no notes logged)"),
        "notes_truncated": truncated,
        "last_synced_at": a["last_synced_at"],
    })


# ---------------------------------------------------------------------------
# Slack conversations: Ion's own public channels as indexed by dce's
# slack_indexer (daily `slack-daily-sync`). A "group" is one thread, or a
# 30-minute window of top-level messages, stored with its verbatim text
# (`raw_text`, "[YYYY-MM-DD HH:MM] Name: text" lines), an LLM summary, and
# LLM-attributed orgs/deals. Until these two tools Slack reached the model
# only as get_org_dossier's 3 most recent groups (200-char summaries) -- and
# not at all for the ~half of groups with no org link. The text is verbatim
# by design: every Ion member can already read these channels in Slack.
# ---------------------------------------------------------------------------

_SLACK_PREVIEW_CHARS = 400
_SLACK_SNIPPET_BEFORE = 150
_SLACK_SNIPPET_LEN = 400
_SLACK_SEARCH_PEOPLE_CAP = 8
_SLACK_SEARCH_ORG_CAP = 6
_SLACK_READ_TEXT_CAP = 30_000
_SLACK_READ_FILE_CAP = 20
_SLACK_MENTION_LOOKUP_CAP = 15
# A few legacy rows carry non-numeric ts; every ts->time cast is guarded.
_SLACK_TS_OK = "smg.last_ts ~ '^[0-9]+(\\.[0-9]+)?$' AND smg.first_ts ~ '^[0-9]+(\\.[0-9]+)?$'"

_SLACK_USER_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|([^>]*))?>")
_SLACK_SPECIAL_RE = re.compile(r"<!(channel|here|everyone)(?:\|[^>]*)?>")
_SLACK_SUBTEAM_RE = re.compile(r"<!subteam\^[A-Z0-9]+(?:\|([^>]*))?>")
_SLACK_CHANNEL_RE = re.compile(r"<#[A-Z0-9]+\|([^>]*)>")
_SLACK_LINK_RE = re.compile(r"<((?:https?|mailto):[^|>]+)(?:\|([^>]*))?>")


def _slack_text(raw: str | None, resolve_names: bool = False) -> str:
    """Slack's wire markup -> plain text: `<url|label>` becomes
    `label (url)`, channel/group mentions their names, and HTML entities
    unescaped. With resolve_names, `<@U...>` mentions become `@Real Name`
    via the bot token (a handful of cached users.info calls; unresolved
    ids are kept as `@U...`)."""
    if not raw:
        return ""
    from ..slack.users import slack_user_display_name  # slack_sdk import is lazy

    names: dict[str, str | None] = {}

    def _user(m: re.Match) -> str:
        uid, label = m.group(1), m.group(2)
        if label:
            return f"@{label}"
        if resolve_names and uid not in names and len(names) < _SLACK_MENTION_LOOKUP_CAP:
            names[uid] = slack_user_display_name(uid)
        return f"@{names.get(uid) or uid}"

    def _link(m: re.Match) -> str:
        url, label = m.group(1), m.group(2)
        url = url.removeprefix("mailto:") if label else url
        return f"{label} ({url})" if label and label != url else url

    text = _SLACK_USER_RE.sub(_user, raw)
    text = _SLACK_SPECIAL_RE.sub(lambda m: f"@{m.group(1)}", text)
    text = _SLACK_SUBTEAM_RE.sub(lambda m: m.group(1) or "@group", text)
    text = _SLACK_CHANNEL_RE.sub(lambda m: f"#{m.group(1)}", text)
    text = _SLACK_LINK_RE.sub(_link, text)
    return html.unescape(text)


def _slack_summary(summary: str | None) -> str:
    """dce's scanner stores a failed LLM call as a marker, not NULL."""
    if not summary or summary in ("[llm_failed]", "[scan_failed]"):
        return "(no summary -- read the messages)"
    return summary


def _slack_snippet(text: str, terms: list[str]) -> str | None:
    """~400 chars of the verbatim text around the first term that occurs."""
    low = text.lower()
    hits = [i for i in (low.find(t.lower()) for t in terms) if i >= 0]
    if not hits:
        return None
    start = max(0, min(hits) - _SLACK_SNIPPET_BEFORE)
    end = start + _SLACK_SNIPPET_LEN
    return ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")


def _slack_coverage(cur) -> dict:
    cur.execute(
        f"""
        SELECT COUNT(*) AS groups,
               COUNT(DISTINCT smg.channel_id) AS channels,
               to_timestamp(MIN(smg.first_ts::numeric))::date AS earliest,
               to_timestamp(MAX(smg.last_ts::numeric))::date  AS latest,
               MAX(smg.created_at)::date AS last_indexed
          FROM dealcloud.slack_message_group smg
         WHERE {_SLACK_TS_OK}
        """
    )
    return dict(cur.fetchone())


def _slack_note(cov: dict) -> str:
    # In the RESPONSE, not the description: the numbers move every daily sync.
    return (
        f"Slack index covers {cov['groups']} conversations across "
        f"{cov['channels']} of Ion's PUBLIC channels, {cov['earliest']} to "
        f"{cov['latest']} (last indexed {cov['last_indexed']}). Private "
        "channels and DMs are never indexed, and a few chatter channels "
        "(all_staff, off-topic, newsletter, office) are skipped. A "
        "conversation is one thread or a 30-minute burst of channel "
        "messages, so one discussion can span several. Org and deal links "
        "are LLM-attributed and only about half of conversations have one: "
        "if org_ids/deal_ids come back thin, retry with the company's name "
        "as `query`. 'Nothing found' means nothing in those channels, not "
        "that it was never discussed. Times are UTC."
    )


class SearchSlackInput(BaseModel):
    query: str | None = Field(
        None, max_length=200,
        description=(
            "Keywords matched literally (case-insensitive) against the "
            "conversation's VERBATIM message text and its summary. Every "
            "word must appear; wrap a phrase in double quotes to match it "
            "as-is, e.g. '\"capital call\" Prelude'. Not semantic. A company "
            "name is usually the best query -- it also finds the "
            "conversations no org was attributed to."
        ),
    )
    org_ids: list[int] = Field(
        default_factory=list, max_length=20,
        description=(
            "Restrict to conversations attributed to these orgs (canonical "
            "heads from find_organizations / bundle_via_supersede)."
        ),
    )
    deal_ids: list[int] = Field(
        default_factory=list, max_length=20,
        description="Restrict to conversations linked to these dealcloud.deal ids.",
    )
    channel: str | None = Field(
        None, max_length=100,
        description="Channel name, or part of it ('deals', '#fund-ops'). Substring match.",
    )
    participant: str | None = Field(
        None, max_length=100,
        description=(
            "Someone who posted in the conversation, by name (Slack real "
            "names, e.g. 'Lailo'). Substring match."
        ),
    )
    since: date | None = Field(
        None, description="Only conversations active on or after this date (YYYY-MM-DD)."
    )
    until: date | None = Field(
        None, description="Only conversations that started on or before this date (YYYY-MM-DD)."
    )
    limit: int = Field(15, ge=1, le=50, description="Max conversations to return.")
    offset: int = Field(0, ge=0, description="For paging through `total_matches`.")


class SlackConversationIdInput(BaseModel):
    conversation_id: int = Field(
        ...,
        description=(
            "conversation_id from search_slack_conversations (the `id` of a "
            "slack group in get_org_dossier; dealcloud.slack_message_group.id)."
        ),
    )


@slack_registry.tool(
    "search_slack_conversations",
    (
        "Search Ion Pacific's own internal SLACK conversations (public "
        "channels), matching keywords in the VERBATIM messages as well as "
        "an LLM summary. Use for 'what has the team said about X on Slack', "
        "'who flagged Y', 'what's been discussed in #<channel> this month', "
        "'when did <person> last mention Z'. Combine any of: keywords, "
        "org_ids, deal_ids, a channel, a participant and a date range; at "
        "least one filter is required. Newest first, with `total_matches` "
        "for paging. Each row has conversation_id, channel, first/last "
        "message time, who posted, a summary preview, a `match_snippet` of "
        "the actual text around your keyword, the attributed orgs/deals and "
        "a Slack permalink. Call read_slack_conversation for the full "
        "message text. CITING: link a conversation as [#channel, date]"
        "(permalink). Read-only."
    ),
    SearchSlackInput,
)
def search_slack_conversations(inp: SearchSlackInput, ctx: dict) -> ToolResult:
    terms = _split_terms(inp.query)
    channel = (inp.channel or "").strip().lstrip("#")
    participant = (inp.participant or "").strip()
    if not (terms or inp.org_ids or inp.deal_ids or channel or participant
            or inp.since or inp.until):
        return ToolResult(output=(
            "Give at least one filter: query, org_ids, deal_ids, channel, "
            "participant, since or until."
        ))

    where, params = [_SLACK_TS_OK], []
    # Not summary || raw_text: no need to build a copy of every row's text.
    for t in terms:
        where.append("(smg.raw_text ILIKE %s OR smg.summary ILIKE %s)")
        params += [_like(t), _like(t)]
    if inp.org_ids:
        where.append(
            "EXISTS (SELECT 1 FROM dealcloud.organization_entity oe "
            "         WHERE oe.entity_type = 'slack_message_group' "
            "           AND oe.entity_id = smg.id "
            "           AND oe.organization_id = ANY(%s))"
        )
        params.append(inp.org_ids)
    if inp.deal_ids:
        where.append(
            "EXISTS (SELECT 1 FROM dealcloud.slack_message_group_deal sd "
            "         WHERE sd.message_group_id = smg.id AND sd.deal_id = ANY(%s))"
        )
        params.append(inp.deal_ids)
    if channel:
        where.append("sc.name ILIKE %s")
        params.append(_like(channel))
    if participant:
        where.append(
            "EXISTS (SELECT 1 FROM unnest(smg.participants) p WHERE p ILIKE %s)"
        )
        params.append(_like(participant))
    if inp.since:
        where.append("to_timestamp(smg.last_ts::numeric) >= %s")
        params.append(inp.since)
    if inp.until:
        where.append("to_timestamp(smg.first_ts::numeric) < %s::date + 1")
        params.append(inp.until)

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT smg.id AS conversation_id, sc.name AS channel,
                   sc.is_archived AS channel_archived,
                   smg.thread_ts IS NOT NULL AS is_thread,
                   to_timestamp(smg.first_ts::numeric) AS first_message_at,
                   to_timestamp(smg.last_ts::numeric)  AS last_message_at,
                   smg.message_count, smg.participants, smg.permalink,
                   smg.summary, smg.raw_text,
                   (SELECT COUNT(*) FROM dealcloud.slack_message_group_file f
                     WHERE f.message_group_id = smg.id) AS file_count,
                   COUNT(*) OVER () AS total_matches
              FROM dealcloud.slack_message_group smg
              LEFT JOIN dealcloud.slack_channel sc ON sc.id = smg.channel_id
             WHERE {' AND '.join(where)}
             ORDER BY smg.last_ts::numeric DESC, smg.id DESC
             LIMIT %s OFFSET %s
            """,
            [*params, inp.limit, inp.offset],
        )
        rows = [dict(r) for r in cur.fetchall()]
        total = rows[0]["total_matches"] if rows else 0
        ids = [r["conversation_id"] for r in rows]

        orgs: dict[int, list] = {}
        deals: dict[int, list] = {}
        if ids:
            cur.execute(
                """
                SELECT DISTINCT ON (oe.entity_id, o.id)
                       oe.entity_id AS conversation_id, o.id AS org_id, o.name,
                       oe.relationship_type
                  FROM dealcloud.organization_entity oe
                  JOIN dealcloud.organization o ON o.id = oe.organization_id
                 WHERE oe.entity_type = 'slack_message_group' AND oe.entity_id = ANY(%s)
                 ORDER BY oe.entity_id, o.id
                """,
                (ids,),
            )
            for o in cur.fetchall():
                lst = orgs.setdefault(o["conversation_id"], [])
                if len(lst) < _SLACK_SEARCH_ORG_CAP:
                    lst.append({"org_id": o["org_id"], "name": o["name"],
                                "relationship": o["relationship_type"]})
            cur.execute(
                """
                SELECT sd.message_group_id AS conversation_id, d.id AS deal_id, d.name
                  FROM dealcloud.slack_message_group_deal sd
                  JOIN dealcloud.deal d ON d.id = sd.deal_id
                 WHERE sd.message_group_id = ANY(%s)
                 ORDER BY 1, 2
                """,
                (ids,),
            )
            for d in cur.fetchall():
                deals.setdefault(d["conversation_id"], []).append(
                    {"deal_id": d["deal_id"], "name": d["name"]})
        coverage = _slack_coverage(cur)

    results = []
    for r in rows:
        cid = r["conversation_id"]
        summary = _slack_summary(r["summary"])
        results.append({
            "conversation_id":  cid,
            "channel":          r["channel"],
            "channel_archived": r["channel_archived"],
            "is_thread":        r["is_thread"],
            "first_message_at": r["first_message_at"],
            "last_message_at":  r["last_message_at"],
            "message_count":    r["message_count"],
            "participants":     (r["participants"] or [])[:_SLACK_SEARCH_PEOPLE_CAP],
            "summary_preview":  summary[:_SLACK_PREVIEW_CHARS]
                                + ("..." if len(summary) > _SLACK_PREVIEW_CHARS else ""),
            "match_snippet":    _slack_snippet(_slack_text(r["raw_text"]), terms)
                                if terms else None,
            "file_count":       r["file_count"],
            "orgs":             orgs.get(cid, []),
            "deals":            deals.get(cid, []),
            "permalink":        r["permalink"],
        })
    return ToolResult(output={
        "total_matches": total,
        "returned": len(results),
        "offset": inp.offset,
        "results": results,
        "note": _slack_note(coverage),
    })


@slack_registry.tool(
    "read_slack_conversation",
    (
        "Open ONE internal Slack conversation: its VERBATIM messages "
        "(timestamped, with who wrote each, @mentions resolved to names), "
        "the LLM summary, channel, participants, the orgs and deals it "
        "concerns with why, any files posted (with a summary and download "
        "link), and the ids of the conversations just before and after it "
        "in the same channel -- read those when this one starts or ends "
        "mid-discussion. Use after search_slack_conversations or "
        "get_org_dossier surfaces a conversation_id. Quote the messages "
        "rather than paraphrasing when the wording matters, and attribute "
        "each claim to whoever wrote it. CITING: link it as [#channel, "
        "date](permalink). Read-only."
    ),
    SlackConversationIdInput,
)
def read_slack_conversation(inp: SlackConversationIdInput, ctx: dict) -> ToolResult:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT smg.id AS conversation_id, smg.channel_id, smg.first_ts,
                   sc.name AS channel, sc.purpose AS channel_purpose,
                   sc.is_archived AS channel_archived,
                   smg.thread_ts IS NOT NULL AS is_thread,
                   to_timestamp(smg.first_ts::numeric) AS first_message_at,
                   to_timestamp(smg.last_ts::numeric)  AS last_message_at,
                   smg.message_count, smg.participants, smg.permalink,
                   smg.summary, smg.raw_text
              FROM dealcloud.slack_message_group smg
              LEFT JOIN dealcloud.slack_channel sc ON sc.id = smg.channel_id
             WHERE smg.id = %s AND {_SLACK_TS_OK}
            """,
            (inp.conversation_id,),
        )
        g = cur.fetchone()
        if g is None:
            return ToolResult(output=f"Slack conversation {inp.conversation_id} not found.")
        g = dict(g)

        cur.execute(
            """
            SELECT DISTINCT ON (o.id)
                   o.id AS org_id, o.name, oe.relationship_type, oe.context
              FROM dealcloud.organization_entity oe
              JOIN dealcloud.organization o ON o.id = oe.organization_id
             WHERE oe.entity_type = 'slack_message_group' AND oe.entity_id = %s
             ORDER BY o.id, oe.context IS NULL
            """,
            (inp.conversation_id,),
        )
        linked_orgs = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT d.id AS deal_id, d.name, d.status
              FROM dealcloud.slack_message_group_deal sd
              JOIN dealcloud.deal d ON d.id = sd.deal_id
             WHERE sd.message_group_id = %s
             ORDER BY d.id
            """,
            (inp.conversation_id,),
        )
        linked_deals = [dict(r) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT file_name, mime_type, file_size, slack_url, download_url,
                   LEFT(summary, %s) AS summary,
                   COUNT(*) OVER () AS total
              FROM dealcloud.slack_message_group_file
             WHERE message_group_id = %s
             ORDER BY id
             LIMIT %s
            """,
            (_SLACK_PREVIEW_CHARS, inp.conversation_id, _SLACK_READ_FILE_CAP),
        )
        files = [dict(r) for r in cur.fetchall()]
        # Neighbours by first_ts within the channel (the same key the
        # indexer's UNIQUE (channel_id, first_ts) is on).
        cur.execute(
            f"""
            SELECT (SELECT smg.id FROM dealcloud.slack_message_group smg
                     WHERE smg.channel_id = %(ch)s AND {_SLACK_TS_OK}
                       AND smg.first_ts::numeric < %(ts)s::numeric
                     ORDER BY smg.first_ts::numeric DESC LIMIT 1) AS previous_id,
                   (SELECT smg.id FROM dealcloud.slack_message_group smg
                     WHERE smg.channel_id = %(ch)s AND {_SLACK_TS_OK}
                       AND smg.first_ts::numeric > %(ts)s::numeric
                     ORDER BY smg.first_ts::numeric LIMIT 1) AS next_id
            """,
            {"ch": g["channel_id"], "ts": g["first_ts"]},
        )
        nb = cur.fetchone()
        coverage = _slack_coverage(cur)

    text = _slack_text(g["raw_text"], resolve_names=True)
    files_total = files[0]["total"] if files else 0
    for f in files:
        f.pop("total")
    return ToolResult(output={
        "conversation_id":  g["conversation_id"],
        "channel":          g["channel"],
        "channel_purpose":  g["channel_purpose"] or None,
        "channel_archived": g["channel_archived"],
        "is_thread":        g["is_thread"],
        "first_message_at": g["first_message_at"],
        "last_message_at":  g["last_message_at"],
        "message_count":    g["message_count"],
        "participants":     g["participants"] or [],
        "summary":          _slack_summary(g["summary"]),
        "messages":         text[:_SLACK_READ_TEXT_CAP] or "(no message text stored)",
        "messages_truncated": len(text) > _SLACK_READ_TEXT_CAP,
        "orgs":             linked_orgs,
        "deals":            linked_deals,
        "files":            files,
        "files_omitted":    max(0, files_total - len(files)),
        "previous_conversation_id": nb["previous_id"],
        "next_conversation_id":     nb["next_id"],
        "permalink":        g["permalink"],
        "note":             _slack_note(coverage),
    })


def _call_dossier_fn(fn_name: str, org_ids: list[int]) -> ToolResult:
    """Invoke one of the dealcloud.org_* functions, returning the JSONB
    output as a Python dict. Centralised so tool handlers stay tiny."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"SELECT dealcloud.{fn_name}(%s::int[]) AS j",
            (org_ids,),
        )
        row = cur.fetchone()
    return ToolResult(output=(row or {}).get("j") or {})


# ---------------------------------------------------------------------------
# Chat-triggered background data-room build (data_room_coverage phase 2, see
# memory: data_room_coverage_analysis). Registered directly on slack_registry
# (not a phase4-only registry, not a clone) so this auto-exposes to BOTH
# Todd/Slack and the Ion Deal Research MCP connector at once, same as
# list_all_deals / find_new_deals_to_discuss above. Reuses the EXISTING
# checklist/gap-detection engine as-is via deal_cloud_enhancer's
# data_room_build_job internal API (services/data_room_build.py) -- the
# build itself runs in the background on dce's data-room-build-runner cron,
# independent of this chat turn or any session staying open; the user gets
# a Slack DM when it finishes.
# ---------------------------------------------------------------------------

class BuildDataRoomInput(BaseModel):
    folder_path: str = Field(
        ...,
        min_length=1, max_length=1000,
        description=(
            "SharePoint folder path prefix, e.g. 'Common/Deal files/"
            "_SINGLE ASSET DEALS/2026/Positron/'. Every indexed document "
            "whose path starts with this prefix becomes part of the room."
        ),
    )
    requested_by_email: str = Field(
        ...,
        min_length=3, max_length=200,
        description=(
            "Ion Pacific email of whoever asked for this build -- this is "
            "who gets DM'd on Slack when the build finishes (docs scanned, "
            "Found/Unconfirmed/Candidate-Gap counts, and the actual "
            "Candidate Gap criteria names). ASK the user for their email "
            "if you don't already know it from this conversation -- do not "
            "guess or default to yourself/someone else, since a wrong "
            "value means the wrong person gets DM'd."
        ),
    )


def _dm_list(result, requester: str) -> str:
    """Who will actually be DM'd when this room finishes.

    A room is per-FOLDER and shared, so joining an existing one can mean
    colleagues who asked earlier get the DM too. Naming only the current
    requester would be wrong in that case, and silently CC'ing people
    without saying so is worse -- the user should know who else hears about
    it. Falls back to the requester alone for an older dce that doesn't
    report subscribers.
    """
    subs = [e for e in (getattr(result, "subscriber_emails", None) or []) if e]
    if not subs:
        return requester
    others = [e for e in subs if e.strip().lower() != requester.strip().lower()]
    if not others:
        return requester
    return f"{requester} (and {', '.join(others)}, who also asked for this folder)"


@slack_registry.tool(
    "build_data_room",
    (
        "Get the data room for a specific SharePoint folder path, building "
        "it if one doesn't exist yet. SAFE TO CALL REPEATEDLY: a data room "
        "IS its folder, so if one already exists this reuses it -- scanning "
        "only documents added since the last build, or doing nothing at all "
        "if the folder is unchanged. Check the returned `action` "
        "('created' / 'refreshed' / 'resummarized' / 'reused') and tell the "
        "user which happened; never imply a fresh build when the room "
        "already existed, and never say 'nothing changed' on "
        "'resummarized' -- that means the stored counts were stale and have "
        "just been corrected, so quote the new ones. "
        "Runs the same "
        "Found/Unconfirmed/Candidate-Gap engine the Coverage tab uses, but "
        "as a persistent background job drained by a cron over several "
        "minutes, NOT something that blocks this chat turn or needs a "
        "browser tab left open. Returns immediately with a job_id + "
        "docs_total. Tell the user you'll DM them on Slack when it's done "
        "(with the counts AND the actual list of missing/Candidate-Gap "
        "criteria names), then let the conversation move on -- do not poll "
        "in a loop yourself. Use check_data_room_build if the user wants "
        "a status update before the DM arrives, and ask_data_room once "
        "it's complete for ad-hoc questions about the room's documents."
    ),
    BuildDataRoomInput,
)
def build_data_room(inp: BuildDataRoomInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, create_build_job

    try:
        result = create_build_job(inp.folder_path, inp.requested_by_email)
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if result.action == "reused":
        note = (
            f"A data room already exists for '{inp.folder_path}' "
            f"(job_id={result.job_id}) and nothing has been added to the "
            f"folder since it was built, so there was nothing to rebuild. "
            f"Use check_data_room_build for its coverage summary, or "
            f"ask_data_room to ask questions about it -- do NOT tell the "
            f"user a new build was started."
        )
    elif result.action == "refreshed":
        # `new_docs` covers two different populations: documents genuinely
        # added to the folder, and documents that were always there but had
        # never been opened (dce's scanner queue ranks spreadsheets last, so
        # a room can complete with files still unread). Calling both "added"
        # was wrong for the second kind, and that kind is the more important
        # one to say out loud -- it means the previous coverage answer was
        # computed without those files.
        if result.docs_to_read:
            detail = (
                f"{result.docs_to_read} document(s) in it had never been "
                f"opened before (typically spreadsheets), so they're being "
                f"read now and the coverage numbers will change"
            )
            if result.new_docs > result.docs_to_read:
                detail += (
                    f", plus {result.new_docs - result.docs_to_read} "
                    f"awaiting classification"
                )
        else:
            detail = (
                f"started scanning the {result.new_docs} document(s) added "
                f"since it was last built -- the rest is already done, so "
                f"this should finish quickly"
            )
        note = (
            f"Reused the existing data room for '{inp.folder_path}' "
            f"(job_id={result.job_id}) and {detail}. "
            f"I'll DM {_dm_list(result, inp.requested_by_email)} on Slack "
            f"when it's updated. Do NOT quote coverage or document counts "
            f"until it completes -- they are about to change."
        )
    elif result.action == "resummarized":
        note = (
            f"The data room for '{inp.folder_path}' (job_id="
            f"{result.job_id}) already existed and is complete, but its "
            f"stored coverage numbers were out of date -- documents in the "
            f"folder had become readable since it was last summarised, so "
            f"they were missing from its counts. The summary has been "
            f"recomputed and now covers "
            f"{result.docs_total} document(s). Nothing needed re-scanning. "
            f"Report the UPDATED numbers from check_data_room_build, and do "
            f"NOT tell the user nothing had changed -- the counts did."
        )
    else:
        note = (
            f"Build started for {result.docs_total} document(s) under "
            f"'{inp.folder_path}'. I'll DM "
            f"{_dm_list(result, inp.requested_by_email)} on Slack when it's "
            f"done (job_id={result.job_id}) -- no need to wait here."
        )

    return ToolResult(output={
        "job_id": result.job_id,
        "docs_total": result.docs_total,
        "status": result.status,
        "action": result.action,
        "new_docs": result.new_docs,
        "subscriber_emails": result.subscriber_emails,
        "note": note,
    })


class CheckDataRoomBuildInput(BaseModel):
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


def _unreadable_note(summary: dict) -> str:
    """Same warning the completion Slack DM carries (see
    routes/internal.py::_format_unreadable_warning) -- documents the
    checklist scanner could not read at all, overwhelmingly
    spreadsheets. Without this the model reports a Candidate Gap as
    settled fact when the deal's financial model was never opened."""
    n = summary.get("docs_unreadable") or 0
    if not n:
        return ""

    def _render(docs: list[dict]) -> str:
        return "; ".join(
            f"{describe_unread_doc(d)} [link: {d.get('web_url')}]"
            if d.get("web_url")
            else f"{describe_unread_doc(d)} "
                 f"[no link; path: {d.get('path') or 'unknown'}]"
            for d in docs
        )

    # Prefer the rich form (name + path + web_url + reason) so each unread
    # file can be rendered as a clickable link AND attributed to the right
    # cause; fall back to names-only for a coverage_summary written before
    # dce started emitting unreadable_docs.
    docs = summary.get("unreadable_docs")
    if docs:
        parts = partition_unread_docs(summary)
        # Each cause gets its own clause, because they tell the user to do
        # three different things. Reporting them as one undifferentiated
        # "could not be read" is what described a 146 MB AGM PDF as a
        # scanner failure with a spreadsheet caveat attached.
        clauses = []
        if parts[REASON_OVER_CAP]:
            clauses.append(
                f"TOO LARGE to read, so never opened -- say so plainly and "
                f"that compressing or splitting the file would fix it: "
                f"{_render(parts[REASON_OVER_CAP])}"
            )
        if parts[REASON_STALE_SKIP]:
            clauses.append(
                f"skipped under an older, lower size limit and now WITHIN "
                f"the limit -- these need a re-scan, they are not too large "
                f"today: {_render(parts[REASON_STALE_SKIP])}"
            )
        if parts[REASON_UNSUPPORTED]:
            clauses.append(
                f"of a file type with NO reader here at all, so neither "
                f"compressing nor re-scanning would help -- do not suggest "
                f"either: {_render(parts[REASON_UNSUPPORTED])}"
            )
        if parts[REASON_UNREADABLE]:
            caveat = (" (spreadsheets are rarely machine-readable here)"
                      if parts["has_spreadsheet"] else "")
            clauses.append(
                f"opened but yielded no usable text{caveat}: "
                f"{_render(parts[REASON_UNREADABLE])}"
            )
        listed = ". Also ".join(clauses)
        count_listed = len(docs)
        link_rule = (
            " When you name any of these files to the user, make it a "
            "clickable markdown link to its link value -- [filename](url) -- "
            "never a bare filename the reader has to go hunting for, and "
            "never a raw URL. For a file with no link, give its path instead "
            "and say it has no link; do not invent one. Do NOT tell the user "
            "a file is unreadable when the reason given is its size."
        )
    else:
        names = summary.get("unreadable_doc_names") or []
        listed = "; ".join(names)
        count_listed = len(names)
        link_rule = ""
    if summary.get("unreadable_doc_names_truncated"):
        listed += f"; ...and {n - count_listed} more"
    return (
        f" IMPORTANT: {n} of {summary.get('docs_in_folder')} documents were "
        f"NOT read (only {summary.get('docs_scanned')} were scanned). Tell "
        f"the user the gap list is NOT YET EVIDENCED rather than confirmed "
        f"missing, and name these unread files as the place the answer may "
        f"actually live, with the reason for each: {listed}.{link_rule}"
    )


def _coverage_summary_note(summary: dict | None) -> str:
    if not summary:
        return "Coverage summary isn't available yet -- still scanning."
    gap_criteria = summary.get("candidate_gap_criteria") or []
    if not gap_criteria:
        return (
            f"Found: {summary.get('found', 0)} | "
            f"Unconfirmed: {summary.get('unconfirmed', 0)} | "
            f"Candidate Gap: {summary.get('candidate_gap', 0)}. "
            "No Candidate Gap criteria -- every applicable checklist item "
            "came back Found or Unconfirmed."
            + _unreadable_note(summary)
        )
    gap_list = "; ".join(gap_criteria)
    return (
        f"Found: {summary.get('found', 0)} | "
        f"Unconfirmed: {summary.get('unconfirmed', 0)} | "
        f"Candidate Gap: {summary.get('candidate_gap', 0)}. "
        f"Candidate Gap criteria (not yet evidenced): {gap_list}"
        + _unreadable_note(summary)
    )


@slack_registry.tool(
    "check_data_room_build",
    (
        "Poll-style status + coverage summary for a job started with "
        "build_data_room. Returns status ('pending'/'scanning'/'complete'/"
        "'failed'), docs_processed/docs_total progress, and once complete, "
        "the Found/Unconfirmed/Candidate-Gap counts PLUS the actual "
        "Candidate Gap criteria names (the 'what's missing' answer -- "
        "always report these by name when the user asks what's missing, "
        "not just the count). Use this if the user doesn't want to wait "
        "for the Slack DM, or wants a progress check on a long build. "
        "The returned requested_by_email/subscriber_emails say who asked "
        "for this room."
    ),
    CheckDataRoomBuildInput,
    mutates_state=False,
)
def check_data_room_build(inp: CheckDataRoomBuildInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, get_build_job

    try:
        job = get_build_job(inp.job_id)
    except ValueError as e:
        return ToolResult(output=str(e))
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if job.status == "complete":
        note = (
            f"status={job.status}, {job.docs_processed}/{job.docs_total} "
            f"docs processed. {_coverage_summary_note(job.coverage_summary)}"
        )
    elif job.status == "failed":
        note = f"status=failed. {job.error or 'no error detail recorded.'}"
    elif job.content_pending:
        # Phase 1: dce is still OPENING files (documents its scanner queue
        # had never reached -- typically spreadsheets, which the queue ranks
        # last). Say so instead of reporting a classify ratio that cannot
        # move yet; "0/29 docs processed" on a healthy build looks stalled.
        note = (
            f"status={job.status}, still reading {job.content_pending} "
            f"document(s) that had never been opened before "
            f"(classification starts once they're read). Do NOT report "
            f"coverage or document counts yet -- they will change."
        )
    else:
        note = (
            f"status={job.status}, {job.docs_processed}/{job.docs_total} "
            f"docs processed so far."
        )

    return ToolResult(output={
        "job_id": job.job_id,
        "folder_path": job.folder_path,
        "status": job.status,
        "docs_total": job.docs_total,
        "docs_processed": job.docs_processed,
        "content_pending": job.content_pending,
        "error": job.error,
        "coverage_summary": job.coverage_summary,
        # Who asked for this room. Returned as DATA rather than asserted in
        # the tool description, because descriptions are cached per user and
        # cannot be corrected once they drift, while responses always
        # reflect the current state. Also lets a colleague see whose room
        # they're looking at without a second call.
        "requested_by_email": job.requested_by_email,
        "subscriber_emails": job.subscriber_emails,
        "note": note,
    })


_PATH_MATCH_LIMIT = 200


def _find_docs_by_path(
    text: str, doc_ids: list[int] | None = None
) -> list[dict]:
    """Literal case-insensitive substring match on a document's full path.

    Deliberately not the semantic retriever. Asked "is there a file named
    X", a top-k topic search answers with the nearest thematic match plus a
    hedge about coverage, which reads as "probably not" whether or not the
    file exists -- on 2026-09-30 a real, freshly-added "Positron GC CSA
    Reservation Letter" was reported absent for exactly that reason.
    Existence and location are index questions and deserve an index answer,
    where "no rows" actually means no such file.

    `path` already ends in the filename, so one ILIKE covers a folder
    ('VDR/LOIs - Contracts'), a partial filename or an extension.
    """
    sql = """
        SELECT id, name, path, web_url, summary IS NOT NULL AS readable
        FROM dealcloud.document
        WHERE path ILIKE %s
    """
    args: list = [f"%{text}%"]
    if doc_ids is not None:
        sql += " AND id = ANY(%s)"
        args.append(doc_ids)
    sql += " ORDER BY path LIMIT %s"
    args.append(_PATH_MATCH_LIMIT)
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, tuple(args))
        return [dict(r) for r in cur.fetchall()]


def _format_doc_matches(rows: list[dict], text: str) -> str:
    more = " (first 200)" if len(rows) >= _PATH_MATCH_LIMIT else ""
    lines = [f"{len(rows)}{more} document(s) whose name/path contains "
             f"{text!r}:"]
    for r in rows:
        mark = "" if r["readable"] else "  [indexed, not yet read]"
        lines.append(f"- [{r['id']}] {r['path']}{mark}")
    lines.append(
        "Pass an id above to read_document / read_document_summary."
    )
    return "\n".join(lines)


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(p for p in parts if p)


_SNIPPET_BEFORE = 200
_SNIPPET_LEN = 460
_CONTENT_MATCH_LIMIT = 30


def _find_text_in_bodies(
    text: str, doc_ids: list[int] | None = None
) -> tuple[list[dict], dict]:
    """Literal case-insensitive search of cached document BODIES.

    The content-side twin of _find_docs_by_path. A figure, a clause or a
    name is a literal question, and semantic top-k answers it badly: asked
    where a $3.55bn pipeline number came from, retrieval returned
    thematically-similar documents and the real source -- one sentence in
    one email -- was never ranked. A substring scan found it in a single
    query the moment that email had a cached body.

    Returns (rows, coverage). Coverage matters as much as the rows: only
    some documents have had their body extracted, so "no match" means
    "not in the text we hold", never "not in this room". The caller must
    say so.
    """
    scope_sql, scope_args = "", []
    if doc_ids is not None:
        scope_sql = " AND id = ANY(%s)"
        scope_args = [doc_ids]

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE body IS NOT NULL) AS with_body
            FROM dealcloud.document
            WHERE true{scope_sql}
            """,
            tuple(scope_args),
        )
        coverage = dict(cur.fetchone() or {})
        # strpos/substring rather than returning bodies: a body runs to tens
        # of thousands of characters and 30 of them would swamp the turn.
        cur.execute(
            f"""
            WITH m AS (
                SELECT id, name, path, body,
                       strpos(lower(body), lower(%s)) AS pos
                FROM dealcloud.document
                WHERE body ILIKE %s{scope_sql}
            )
            SELECT id, name, path, pos,
                   substring(body from greatest(1, pos - %s) for %s) AS snippet
            FROM m
            ORDER BY path
            LIMIT %s
            """,
            tuple([text, f"%{text}%"] + scope_args
                  + [_SNIPPET_BEFORE, _SNIPPET_LEN, _CONTENT_MATCH_LIMIT]),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return rows, coverage


class FindInDocumentsInput(BaseModel):
    text: str = Field(
        ...,
        min_length=2, max_length=200,
        description=(
            "LITERAL text to find inside documents -- a figure ('3.55'), a "
            "phrase ('liquidation preference'), a name. Matched as a "
            "case-insensitive substring of the document's extracted text, "
            "NOT semantically, so write it exactly as it would appear in "
            "the file. Numbers are formatted inconsistently across "
            "documents, so if '3.55bn' finds nothing, try '3.55' or "
            "'3,550'."
        ),
    )
    job_id: int | None = Field(
        default=None,
        description=(
            "Optional data-room build job id, to restrict the search to "
            "that room's documents. Omit to search every document whose "
            "text has been extracted."
        ),
    )


@slack_registry.tool(
    "find_in_documents",
    (
        "Find which document CONTAINS a given piece of text, and show it "
        "in context. Use this -- not search_documents, not ask_data_room "
        "-- when the question is where a specific figure, phrase or clause "
        "comes from: 'where does the $3.55bn number come from', 'which "
        "document says the liquidation preference is 1x', 'who mentioned "
        "the Q4 tape out'. Those tools rank documents by topic and return "
        "only the top few, so the one document holding a number may never "
        "surface; this scans the extracted text of every document and "
        "returns the surrounding passage, so you can quote the source "
        "exactly and cite the file it came from. Returns document ids for "
        "read_document when you need more than the snippet. IMPORTANT: it "
        "searches only documents whose text has been extracted, and the "
        "result says how many that was -- a nil result means 'not in the "
        "text we hold', never 'not in this room'."
    ),
    FindInDocumentsInput,
)
def find_in_documents(inp: FindInDocumentsInput, ctx: dict) -> ToolResult:
    doc_ids = None
    where = "every document with extracted text"
    if inp.job_id is not None:
        from ..data_room_build import (
            DceUnavailable as _DceUnavailable, get_build_job,
        )
        try:
            job = get_build_job(inp.job_id)
        except ValueError as e:
            return ToolResult(output=str(e))
        except _DceUnavailable as e:
            return ToolResult(output=f"Data room build unavailable: {e}")
        doc_ids = job.doc_ids or []
        where = f"data room job {inp.job_id}"
        if not doc_ids:
            return ToolResult(output=f"Data room job {inp.job_id} has no documents yet.")

    rows, cov = _find_text_in_bodies(inp.text, doc_ids=doc_ids)
    with_body = cov.get("with_body") or 0
    total = cov.get("total") or 0
    searched = (f"Searched the extracted text of {with_body} of {total} "
                f"documents in {where}")
    if total and with_body < total:
        searched += (f"; {total - with_body} have no extracted text yet, so "
                     f"they could not be searched")
    searched += "."

    if not rows:
        return ToolResult(output=(
            f"No document contains {inp.text!r}. {searched} If you expected "
            f"a hit, try a different formatting of the same value (for a "
            f"figure: '3.55' rather than '$3.55bn')."
        ))

    out = [f"{len(rows)} document(s) contain {inp.text!r}:", ""]
    for r in rows:
        snippet = " ".join((r["snippet"] or "").split())
        out.append(f"- [{r['id']}] {r['path']}")
        out.append(f"      ...{snippet}...")
    out.append("")
    out.append(searched)
    out.append("Quote from the passage above; call read_document on an id "
               "for the full text.")
    return ToolResult(output="\n".join(out))


class FindDocumentsByNameInput(BaseModel):
    path_or_name: str = Field(
        ...,
        min_length=2, max_length=400,
        description=(
            "LITERAL text to match against each document's filename and "
            "folder path -- not a topic or a question. Matched "
            "case-insensitively as a substring of the full path, so a "
            "folder ('_SINGLE ASSET DEALS/2026/Positron'), a partial "
            "filename ('Reservation Letter') or an extension ('.xlsx') "
            "all work."
        ),
    )
    job_id: int | None = Field(
        default=None,
        description=(
            "Optional data-room build job id, to restrict the search to "
            "that room's documents. Omit to search the whole document "
            "index, which is the right choice when the user is asking "
            "about a folder no room has been built over."
        ),
    )


@slack_registry.tool(
    "find_documents_by_name",
    (
        "Find documents by FILENAME or FOLDER PATH, as a literal index "
        "lookup. Use this -- not search_documents, and not ask_data_room "
        "-- whenever the question is about WHERE a document sits or "
        "WHETHER one exists: 'is there a file called X', 'what is in the "
        "folder Y', 'was the LOI added yet', 'list the contracts folder'. "
        "search_documents ranks documents by topic and returns only the "
        "top few, so it can miss a file that exists and cannot tell you a "
        "file is absent; this tool scans every indexed path, so an empty "
        "result is a definitive 'no such file is indexed'. Returns "
        "document ids to pass to read_document or read_document_summary, "
        "and works with or without a data room."
    ),
    FindDocumentsByNameInput,
)
def find_documents_by_name(
    inp: FindDocumentsByNameInput, ctx: dict
) -> ToolResult:
    doc_ids = None
    scope = "the document index"
    if inp.job_id is not None:
        # Imported here, not at module scope: data_room_build imports back
        # into this module, as the other data-room tools above do.
        from ..data_room_build import (
            DceUnavailable as _DceUnavailable,
            get_build_job,
        )
        try:
            job = get_build_job(inp.job_id)
        except ValueError as e:
            return ToolResult(output=str(e))
        except _DceUnavailable as e:
            return ToolResult(output=f"Data room build unavailable: {e}")
        if job is None:
            return ToolResult(output=f"No data room build job {inp.job_id}.")
        doc_ids = job.doc_ids or []
        scope = f"data room job {inp.job_id} ({len(doc_ids)} documents)"
        if not doc_ids:
            return ToolResult(output=(
                f"Data room job {inp.job_id} has no documents yet, so there "
                f"is nothing to match."
            ))
    rows = _find_docs_by_path(inp.path_or_name, doc_ids=doc_ids)
    if not rows:
        msg = (
            f"No document in {scope} has {inp.path_or_name!r} in its "
            f"filename or folder path. This is a literal lookup over every "
            f"indexed path, not a top-k search, so nothing matching is "
            f"indexed."
        )
        if doc_ids is not None:
            # "Not in this room" is a different claim from "no such file".
            # Say which one it is, so the answer is never "it does not
            # exist" about a file sitting one folder outside the room.
            elsewhere = _find_docs_by_path(inp.path_or_name)
            if elsewhere:
                shown = _format_doc_matches(elsewhere, inp.path_or_name)
                return ToolResult(output=(
                    f"{msg} It does exist outside this room, though -- "
                    f"{len(elsewhere)} match(es) elsewhere in the document "
                    f"index:\n{shown}"
                ))
        return ToolResult(output=(
            msg + " If the file was added to SharePoint very recently, "
            "build_data_room over its folder will index it first."
        ))
    return ToolResult(output=_format_doc_matches(rows, inp.path_or_name))


class AskDataRoomInput(BaseModel):
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
            "Question to ask of the data room's documents. Phrased as a "
            "complete question; the answer will only draw on this job's "
            "actual documents, never outside knowledge. Optional when "
            "`path_or_name` is given on its own."
        ),
    )
    path_or_name: str | None = Field(
        default=None,
        max_length=400,
        description=(
            "LITERAL text to match against each document's filename and "
            "folder path -- not a topic. Use this, not `question`, whenever "
            "the ask is about WHERE a document sits or WHETHER one exists: "
            "'is there a file called X', 'what is in the folder Y', 'list "
            "the LOIs folder'. Matched case-insensitively as a substring of "
            "the full path, so a folder ('VDR/LOIs - Contracts'), a partial "
            "filename ('Reservation Letter') or an extension ('.xlsx') all "
            "work. Answers from the index, so 'no match' means the room "
            "genuinely holds no such file -- unlike a topic search, which "
            "can miss a file that exists. Combine with `question` to ask "
            "something of just the documents under that path."
        ),
    )


@slack_registry.tool(
    "ask_data_room",
    (
        "Ask an ad-hoc question about a completed data-room build job's "
        "documents -- retrieves the most relevant documents from the "
        "job's folder and answers ONLY from them (sterile, no outside "
        "knowledge or web search), with inline citations. Only call this "
        "once check_data_room_build shows status='complete'; if it's "
        "still scanning, tell the user to wait or check back rather than "
        "calling this early (the checklist scan and this retrieval are "
        "independent, but an incomplete room may be missing relevant "
        "documents entirely). Prefer this over answering from the "
        "coverage summary alone -- it does real retrieval against the "
        "room's content."
    ),
    AskDataRoomInput,
    mutates_state=False,
)
def ask_data_room(inp: AskDataRoomInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, get_build_job
    from ..claude_data_room import ClaudeRoomError as _ClaudeRoomError, ask_room_for_docs

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
            f"docs processed) -- call check_data_room_build again shortly, "
            f"or wait for the Slack DM."
        ))
    # An incomplete room is still worth asking. Refusing outright meant a
    # handful of newly-added files -- which the check-for-new-documents
    # rule surfaces by flipping a finished room back to 'pending' -- made
    # a 1,500-document room unusable. Answer from what has been read, and
    # say what has not.
    pending_note = ""
    if job.status != "complete":
        behind = max(0, (job.docs_total or 0) - (job.docs_processed or 0))
        # docs_total lags when the folder has just gained files, so a bare
        # "1586/1586 docs read" next to status=scanning reads as finished.
        progress = (
            f"{job.docs_processed}/{job.docs_total} docs read, {behind} "
            f"outstanding" if behind else
            "newly-added documents are still being picked up"
        )
        pending_note = (
            f"[Note: this room is still being built (status={job.status}; "
            f"{progress}). The answer below draws only on the documents "
            f"read so far; call check_data_room_build for progress.]"
        )

    if not inp.question and not inp.path_or_name:
        return ToolResult(output=(
            "give either `question` (ask the documents something) or "
            "`path_or_name` (find documents by filename/folder)."
        ))

    doc_ids = job.doc_ids
    listing = ""
    if inp.path_or_name:
        # A LITERAL index lookup, deliberately not the semantic retriever.
        # Asked "is there a file named X", topic search answers with the
        # nearest thematic match and a hedge about coverage -- which reads
        # as "probably not" whether or not the file exists. On 2026-09-30
        # that is exactly what happened: a real, freshly-added
        # "Positron GC CSA Reservation Letter" was reported as absent
        # because a top-k search over 1,607 documents never surfaced it.
        # Existence and location are index questions and deserve an index
        # answer -- here, "no match" actually means no match.
        rows = _find_docs_by_path(inp.path_or_name, doc_ids=doc_ids)
        if not rows:
            return ToolResult(output=(
                f"No document in this room's {len(doc_ids)} indexed files has "
                f"{inp.path_or_name!r} in its name or folder path. This is a "
                f"literal index lookup over the whole room, not a top-k "
                f"search, so this is a definitive answer for the files the "
                f"room has indexed."
            ))
        listing = _format_doc_matches(rows, inp.path_or_name)
        # Narrow any follow-up question to just these documents.
        doc_ids = [r["id"] for r in rows]

    if not inp.question:
        return ToolResult(output=_join_parts([pending_note, listing]))

    try:
        result = ask_room_for_docs(doc_ids, inp.question)
    except _ClaudeRoomError as e:
        return ToolResult(output=f"Claude room error: {e}")
    return ToolResult(output=_join_parts([pending_note, listing, result]))


class StartDataRoomBuildSweepInput(BaseModel):
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
    question: str = Field(
        ...,
        min_length=4, max_length=2000,
        description=(
            "The specific question to check systematically across every "
            "readable document in the job's folder. Phrase it precisely "
            "-- this drives a per-document classification, not a "
            "retrieval query, so a vague question yields vague/noisy hits."
        ),
    )


@slack_registry.tool(
    "start_data_room_build_sweep",
    (
        "Start a systematic, exhaustive sweep of EVERY document "
        "in a build_data_room job's folder against a specific question -- "
        "for the long tail OUTSIDE the 113-item coverage checklist. Use "
        "this only after ask_data_room has already come up empty or "
        "uncertain, or when the user explicitly asks for an exhaustive/"
        "definitive check ('search everything', 'are you sure nothing "
        "mentions X'). Do NOT use this as a first resort -- it's slower "
        "and costs more than ask_data_room because it reads every "
        "document, not just the likely ones. Returns immediately with a "
        "sweep_id and docs_total; does NOT process any documents yet -- "
        "call check_data_room_build_sweep repeatedly to make progress and "
        "see results. Tell the user this will take a few minutes for a "
        "large folder and you'll report back as it progresses. Sweeps are "
        "attributed to whoever ran them."
    ),
    StartDataRoomBuildSweepInput,
)
def start_data_room_build_sweep(inp: StartDataRoomBuildSweepInput, ctx: dict) -> ToolResult:
    from ..data_room_build import DceUnavailable as _DceUnavailable, get_build_job
    from ..data_room_sweep import (
        DceUnavailable as _SweepDceUnavailable,
        start_sweep_for_docs as _start_sweep_for_docs,
    )

    try:
        job = get_build_job(inp.job_id)
    except ValueError as e:
        return ToolResult(output=str(e))
    except _DceUnavailable as e:
        return ToolResult(output=f"Data room build unavailable: {e}")

    if not job.doc_ids:
        return ToolResult(output=f"job {inp.job_id}'s folder has no readable documents to sweep.")

    # Attribute the sweep to whoever ran it when we know that, falling back
    # to the room's creator -- a shared room means those are often different
    # people, and recording the creator for someone else's sweep would
    # misattribute it in the audit trail.
    created_by = inp.requested_by_email or job.requested_by_email
    try:
        result = _start_sweep_for_docs(job.doc_ids, inp.question, created_by)
    except _SweepDceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    return ToolResult(output={
        "sweep_id": result.sweep_id, "docs_total": result.docs_total,
        "status": result.status,
        "note": (
            "Sweep started but not yet processed -- call "
            f"check_data_room_build_sweep with sweep_id={result.sweep_id} "
            "to advance it and see progress."
        ),
    })


class CheckDataRoomBuildSweepInput(BaseModel):
    sweep_id: int = Field(..., description="The sweep_id returned by start_data_room_build_sweep.")


@slack_registry.tool(
    "check_data_room_build_sweep",
    (
        "Check progress on a sweep started with start_data_room_build_sweep, "
        "AND advance it by one more batch of documents (real LLM "
        "calls) in the same call -- this is deliberately not purely "
        "read-only, so simply calling this repeatedly drains the sweep "
        "over several turns without a separate 'process' action. When "
        "status is 'complete', report the accumulated hits to the user as "
        "the answer (with their evidence quotes). If hits is empty, follow "
        "the returned `note`: it says whether the check was genuinely "
        "exhaustive, or whether some documents could not be read and were "
        "never checked. Phrase a hit-less exhaustive result as 'not found "
        "after an exhaustive check', NEVER a flat 'the answer does not "
        "exist' -- and when documents were unreadable, name them as "
        "unchecked rather than implying full coverage. When status is "
        "'running', tell the user progress (docs_processed/docs_total) and "
        "that you'll check again."
    ),
    CheckDataRoomBuildSweepInput,
)
def check_data_room_build_sweep(inp: CheckDataRoomBuildSweepInput, ctx: dict) -> ToolResult:
    from ..data_room_sweep import (
        DceUnavailable as _DceUnavailable,
        advance_sweep as _advance_sweep,
        get_sweep as _get_sweep,
    )
    try:
        _advance_sweep(inp.sweep_id)
    except _DceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    try:
        detail = _get_sweep(inp.sweep_id)
    except _DceUnavailable as e:
        return ToolResult(output=f"Sweep unavailable: {e}")
    # A sweep's answer is the one this project phrases most confidently
    # ("not found after an exhaustive check"), so the documents it could NOT
    # open have to travel with the result. dce now reads unread files before
    # classifying, which makes the check genuinely exhaustive in most rooms;
    # what remains here is the residue that cannot be read at all (images,
    # video, oversized files), and a hit-less result over a room with such a
    # residue is NOT exhaustive.
    caveat = ""
    if detail.docs_unread:
        caveat = (
            f"STILL READING {detail.docs_unread} document(s) that had never "
            f"been opened -- coverage is not final yet, so do NOT report this "
            f"as an exhaustive check. Call again."
        )
    elif detail.docs_unreadable:
        names = "; ".join(
            d.get("name", "?") for d in (detail.unreadable_docs or [])[:5]
        )
        caveat = (
            f"{detail.docs_unreadable} document(s) in this room could NOT be "
            f"read at all and were therefore never checked"
            + (f" ({names})" if names else "")
            + ". If there are no hits, say the question was not found in the "
              "documents that COULD be read, and name these as unchecked -- "
              "do NOT call it exhaustive."
        )
    return ToolResult(output={
        "sweep_id": detail.sweep_id, "question": detail.question,
        "status": detail.status, "docs_total": detail.docs_total,
        "docs_processed": detail.docs_processed,
        "docs_unread": detail.docs_unread,
        "docs_unreadable": detail.docs_unreadable,
        "unreadable_docs": detail.unreadable_docs,
        "hits": [
            {"doc_name": h.doc_name, "present": h.present, "evidence": h.evidence}
            for h in detail.hits
        ],
        # Only claim full coverage when dce actually reported the counts.
        # An older dce omits them, and silently reading that absence as
        # "zero unread, zero unreadable" would manufacture exactly the
        # false confidence this change exists to remove.
        "note": caveat or (
            "Every document in this room was read and checked; a hit-less "
            "result here is a genuine 'not found after an exhaustive check'."
            if detail.docs_unread is not None
               and detail.docs_unreadable is not None
            else "This dce build does not report read-coverage for sweeps, "
                 "so do NOT claim the check was exhaustive -- say the "
                 "question was not found in the documents that were checked."
        ),
    })


# ---------------------------------------------------------------------------
# Files the user dropped into the Slack conversation
# ---------------------------------------------------------------------------
#
# Uploads are extracted eagerly, at message-receive time, by
# chat_slack/uploads.py -- the turn the model sees already carries the
# first EXCERPT_CHARS of each file. These two tools cover what the
# excerpt can't: the rest of a long document, and a file from an earlier
# turn that has since fallen out of the trimmed message history.
#
# Deliberately NOT exposed over MCP (see chat_mcp_tools.MCP_WITHHELD) --
# there is no Slack conversation on that surface, so there is nothing
# for them to read.


class ReadUploadedFileInput(BaseModel):
    file_id: str | None = Field(
        None,
        description=(
            "Slack file id of the upload, e.g. 'F07ABCDEF'. This is the "
            "file_id shown in the ATTACHED FILES block of the user's "
            "message. Preferred over file_name."
        ),
        max_length=64,
    )
    file_name: str | None = Field(
        None,
        description=(
            "Alternative to file_id: the file's name, or part of it "
            "(case-insensitive). Use when the user refers to a file from an "
            "earlier turn by name and you don't have its id. The most "
            "recently uploaded match in this conversation wins."
        ),
        max_length=255,
    )
    query: str | None = Field(
        None,
        description=(
            "Topic filter. When set, returns only the ~500-character windows "
            "around each occurrence of this text instead of the document from "
            "the top. Use it on anything long -- a deck, a memo, an LPA -- "
            "rather than paging through with max_chars."
        ),
        max_length=200,
    )
    max_chars: int = Field(
        20000,
        description="Cap on returned characters.",
        ge=500, le=60000,
    )
    offset: int = Field(
        0,
        description=(
            "Start reading this many characters in. Only for walking a long "
            "file sequentially when query isn't the right tool; ignored when "
            "query is set."
        ),
        ge=0,
    )


@slack_registry.tool(
    "read_uploaded_file",
    (
        "Read the text of a file the user uploaded to this Slack "
        "conversation (PDF / DOCX / PPTX / XLSX / TXT / image / zip). "
        "The user's message already shows the first ~2,000 characters of "
        "anything they just attached -- call this when you need more than "
        "that, or when they refer to a file from an earlier message. "
        "Identify it by file_id (preferred) or file_name. For anything "
        "long, pass `query` to pull just the passages about a topic "
        "instead of reading from the top. Returns body, total_chars, "
        "truncated, plus the file's Slack permalink."
    ),
    ReadUploadedFileInput,
)
def read_uploaded_file(inp: ReadUploadedFileInput, ctx: dict) -> ToolResult:
    from ..document_body import _filter_body_to_snippets
    from .uploads import lookup_uploaded_file

    if not inp.file_id and not inp.file_name:
        return ToolResult(output={
            "ok": False,
            "error": "Give me either file_id or file_name.",
        })

    row = lookup_uploaded_file(
        slack_file_id=inp.file_id,
        name=inp.file_name,
        team_id=ctx.get("team_id"),
        channel_id=ctx.get("channel_id"),
    )
    if row is None:
        return ToolResult(output={
            "ok": False,
            "error": (
                "No uploaded file matching "
                f"{inp.file_id or inp.file_name!r} in this conversation. "
                "Call list_uploaded_files to see what has been shared here."
            ),
        })

    if not row["ok"]:
        return ToolResult(output={
            "ok": False,
            "name": row["name"],
            "permalink": row["permalink"],
            "error": row["error"] or "That file couldn't be read.",
        })

    body = row["body"] or ""
    total = len(body)

    if inp.query:
        filtered = _filter_body_to_snippets(body, inp.query, inp.max_chars)
        if not filtered:
            return ToolResult(output={
                "ok": False,
                "name": row["name"],
                "total_chars": total,
                "error": f"query_not_found:{inp.query}",
                "hint": ("That phrase isn't in the file. Try a shorter or "
                         "different term, or drop query to read from the top."),
            })
        return ToolResult(output={
            "ok": True,
            "file_id": row["slack_file_id"],
            "name": row["name"],
            "permalink": row["permalink"],
            "query": inp.query,
            "body": filtered,
            "total_chars": total,
            "returned_chars": len(filtered),
            "truncated": len(filtered) < total,
            "note": "Filtered to passages matching query, not the whole file.",
        })

    chunk = body[inp.offset:inp.offset + inp.max_chars]
    end = inp.offset + len(chunk)
    return ToolResult(output={
        "ok": True,
        "file_id": row["slack_file_id"],
        "name": row["name"],
        "permalink": row["permalink"],
        "body": chunk,
        "offset": inp.offset,
        "total_chars": total,
        "returned_chars": len(chunk),
        "truncated": end < total,
        "next_offset": end if end < total else None,
    })


class ListUploadedFilesInput(BaseModel):
    limit: int = Field(
        20, description="Max files to list, newest first.", ge=1, le=50
    )


@slack_registry.tool(
    "list_uploaded_files",
    (
        "List the files people have uploaded to this Slack conversation, "
        "newest first, with each one's file_id, name, size and whether it "
        "could be read. Use when the user refers to a file you can't "
        "identify ('the deck I sent last week') so you can pick the right "
        "file_id for read_uploaded_file."
    ),
    ListUploadedFilesInput,
)
def list_uploaded_files(inp: ListUploadedFilesInput, ctx: dict) -> ToolResult:
    from .uploads import list_conversation_uploads

    team_id = ctx.get("team_id")
    channel_id = ctx.get("channel_id")
    if not team_id or not channel_id:
        return ToolResult(output={
            "ok": False,
            "error": "No Slack conversation context available.",
        })

    rows = list_conversation_uploads(
        team_id=team_id, channel_id=channel_id, limit=inp.limit
    )
    return ToolResult(output={
        "ok": True,
        "count": len(rows),
        "files": [{
            "file_id": r["slack_file_id"],
            "name": r["name"],
            "filetype": r["filetype"],
            "size_bytes": r["size_bytes"],
            "readable": r["ok"],
            "total_chars": r["total_chars"],
            "error": r["error"],
            "permalink": r["permalink"],
            "uploaded_at": r["created_at"].isoformat() if r["created_at"] else None,
        } for r in rows],
    })


# ---------------------------------------------------------------------------
# Writing spreadsheets back into the conversation
# ---------------------------------------------------------------------------
#
# Two tools, because there are two different problems.
#
# `create_spreadsheet` takes rows FROM THE MODEL. That is the flexible
# one -- a comparison table, a tidied-up extract from a document
# somebody just uploaded, anything derived rather than stored. Its
# ceiling is the model's own output budget (MAX_TOKENS in
# orchestrator.py), so it is good for tens of rows, not thousands.
#
# `export_to_spreadsheet` never sends the data through the model at
# all: it re-runs an existing tool server-side, pages it to exhaustion
# and writes what comes back. The deal book is ~1,450 rows, which the
# model could not retype inside any sane token budget and would
# silently truncate if it tried. Naming an export costs ~20 tokens.
#
# Both are MCP_WITHHELD -- they upload into a Slack channel, and an MCP
# caller has no channel to upload to.

# source -> how to fetch it in full. `tool` is re-invoked through this
# same registry rather than the SQL being copied, so an export cannot
# drift from what the equivalent question returns in chat.
_EXPORT_SOURCES: dict[str, dict] = {
    "deals": {
        "tool": "list_all_deals",
        "rows_key": "deals",
        "page": 500,
        "filters": ("status", "company"),
        "sheet": "Deals",
        "preferred": ["deal_name", "company", "status", "transaction_type"],
        "describe": "every deal in the book, with its counterpart company and status",
    },
    "funds": {
        "tool": "list_funds",
        "rows_key": "funds",
        "page": 200,
        "filters": ("name", "reportable_only"),
        "sheet": "Funds",
        "preferred": ["fund_id", "fund_name", "short_name", "fund_type",
                      "fund_status", "vintage_year"],
        "describe": "every fund/SPV with committed capital and performance",
    },
}

# Stop rather than page forever if a source ever stops reporting
# truncation honestly.
_EXPORT_MAX_ROWS = 20_000
_EXPORT_MAX_PAGES = 60


def _fetch_all_rows(source: str, filters: dict, ctx: dict) -> tuple[list[dict], dict]:
    """Page a source tool to exhaustion. Returns (rows, last_output)."""
    spec = _EXPORT_SOURCES[source]
    tool = slack_registry.get(spec["tool"])
    rows: list[dict] = []
    out: dict = {}
    offset = 0
    for _ in range(_EXPORT_MAX_PAGES):
        payload = dict(filters)
        payload["limit"] = spec["page"]
        payload["offset"] = offset
        result = tool.handler(tool.input_model(**payload), ctx)
        out = result.output if hasattr(result, "output") else result
        if not isinstance(out, dict):
            break
        page = out.get(spec["rows_key"]) or []
        rows.extend(page)
        if not page or not out.get("truncated") or len(rows) >= _EXPORT_MAX_ROWS:
            break
        offset += len(page)
    return rows[:_EXPORT_MAX_ROWS], out


class SpreadsheetSheetInput(BaseModel):
    name: str = Field(
        "Sheet1",
        description="Tab name. Excel caps these at 31 characters.",
        max_length=120,
    )
    rows: list[dict[str, Any]] = Field(
        ...,
        description=(
            "The data, one object per row, keys as column names. Use the "
            "SAME keys on every row. Put numbers and dates through as JSON "
            "numbers and 'YYYY-MM-DD' strings rather than pre-formatting "
            "them with currency symbols or thousands separators -- the "
            "point of a spreadsheet is that the reader can sort and sum it."
        ),
        max_length=2000,
    )
    columns: list[str] | None = Field(
        None,
        description=(
            "Optional column order, by row key. Omit to use the order the "
            "keys first appear in. Keys you leave out are dropped."
        ),
        max_length=200,
    )
    column_labels: list[str] | None = Field(
        None,
        description=(
            "Optional human headers, positionally matching `columns`. Omit "
            "and the keys are title-cased ('deal_name' -> 'Deal Name')."
        ),
        max_length=200,
    )


class CreateSpreadsheetInput(BaseModel):
    filename: str = Field(
        "todd-export.xlsx",
        description=("Filename for the upload; '.xlsx' is appended if you "
                     "leave it off. Name it for the content, e.g. "
                     "'moove-cap-table.xlsx'."),
        max_length=120,
    )
    sheets: list[SpreadsheetSheetInput] = Field(
        ...,
        description="One entry per tab. Usually just one.",
        min_length=1, max_length=10,
    )
    comment: str | None = Field(
        None,
        description=(
            "One short line posted with the file, e.g. 'Active pipeline as "
            "of today -- 34 deals.' Slack mrkdwn."
        ),
        max_length=1000,
    )


@slack_registry.tool(
    "create_spreadsheet",
    (
        "Build an .xlsx from rows YOU supply and post it into this Slack "
        "conversation. Use for tables you derive or assemble: a "
        "comparison, a tidied extract from a document the user uploaded, "
        "a shortlist. The file is uploaded for you -- do NOT then repeat "
        "the table in your reply. "
        "For a whole internal dataset (the full deal book, every fund) "
        "use export_to_spreadsheet instead: it fetches server-side and "
        "won't truncate, whereas the rows you write here are capped by "
        "your own output budget -- keep this to roughly 100 rows."
    ),
    CreateSpreadsheetInput,
)
def create_spreadsheet(inp: CreateSpreadsheetInput, ctx: dict) -> ToolResult:
    from .spreadsheet import (SpreadsheetError, build_workbook, safe_filename,
                              upload_to_slack)

    sheets = [{
        "name": sh.name,
        "rows": sh.rows,
        "columns": sh.columns,
        "column_labels": sh.column_labels,
    } for sh in inp.sheets]
    total_rows = sum(len(sh["rows"]) for sh in sheets)
    if not total_rows:
        return ToolResult(output={
            "ok": False,
            "error": ("No rows given, so there's nothing to put in the file. "
                      "If you meant a whole internal dataset, use "
                      "export_to_spreadsheet."),
        })

    try:
        content = build_workbook(sheets)
    except SpreadsheetError as e:
        return ToolResult(output={"ok": False, "error": str(e)})
    except Exception as e:  # noqa: BLE001 -- malformed model rows, not a crash
        return ToolResult(output={
            "ok": False,
            "error": f"Couldn't build the spreadsheet ({type(e).__name__}: {e}).",
        })

    up = upload_to_slack(
        content=content,
        filename=safe_filename(inp.filename),
        channel_id=ctx.get("channel_id") or "",
        thread_ts=ctx.get("thread_ts"),
        initial_comment=inp.comment,
    )
    if not up.get("ok"):
        return ToolResult(output={"ok": False, "error": up.get("error")})
    return ToolResult(output={
        "ok": True,
        "filename": up["filename"],
        "permalink": up.get("permalink"),
        "rows": total_rows,
        "sheets": [sh["name"] for sh in sheets],
        "note": ("The file is already posted in the conversation. Confirm it "
                 "in one short line; do not repeat the table."),
    })


class ExportToSpreadsheetInput(BaseModel):
    source: str = Field(
        ...,
        description=(
            "Which dataset to export in full. "
            "'deals' -- the whole deal book (deal name, company, status, "
            "transaction type); filter with `status` and/or `company`. "
            "'funds' -- every fund/SPV with committed capital and "
            "performance; filter with `name` and/or `reportable_only`."
        ),
    )
    status: list[str] | None = Field(
        None,
        description=("deals only: restrict to these statuses, e.g. "
                     "['Active Pipeline']. Most of the book is "
                     "'Passed/Dead', so ask whether they want it all "
                     "before exporting ~1,450 rows unfiltered."),
    )
    company: str | None = Field(
        None,
        description="deals only: substring filter on company or deal name.",
        max_length=120,
    )
    name: str | None = Field(
        None, description="funds only: fuzzy filter on fund name.",
        max_length=200,
    )
    reportable_only: bool | None = Field(
        None,
        description="funds only: restrict to DealCloud-reportable funds.",
    )
    filename: str | None = Field(
        None, description="Optional filename; one is derived if omitted.",
        max_length=120,
    )
    comment: str | None = Field(
        None, description="One short line posted with the file.",
        max_length=1000,
    )


@slack_registry.tool(
    "export_to_spreadsheet",
    (
        "Export a whole internal dataset to .xlsx and post it into this "
        "Slack conversation. The rows are fetched server-side and paged "
        "to completion, so nothing is truncated and none of the data "
        "passes through your output budget -- this is the right tool for "
        "'export all the deals', 'send me the fund list as a "
        "spreadsheet'. Sources: 'deals' (filter: status, company), "
        "'funds' (filter: name, reportable_only). The file is uploaded "
        "for you -- confirm in one line and do NOT list the rows. "
        "For a table you assemble yourself, use create_spreadsheet."
    ),
    ExportToSpreadsheetInput,
)
def export_to_spreadsheet(inp: ExportToSpreadsheetInput, ctx: dict) -> ToolResult:
    from .spreadsheet import (SpreadsheetError, build_workbook, columns_from_rows,
                              flatten_row, safe_filename, upload_to_slack)

    source = (inp.source or "").strip().lower()
    spec = _EXPORT_SOURCES.get(source)
    if spec is None:
        return ToolResult(output={
            "ok": False,
            "error": (f"I don't have an export called {inp.source!r}. "
                      f"Available: {', '.join(sorted(_EXPORT_SOURCES))}. "
                      "For anything else, assemble the rows yourself and "
                      "use create_spreadsheet."),
        })

    supplied = {k: getattr(inp, k) for k in ("status", "company", "name",
                                             "reportable_only")
                if getattr(inp, k) is not None}
    filters = {k: v for k, v in supplied.items() if k in spec["filters"]}
    ignored = sorted(set(supplied) - set(filters))

    try:
        rows, last = _fetch_all_rows(source, filters, ctx)
    except Exception as e:  # noqa: BLE001 -- a source failing is not a crash
        return ToolResult(output={
            "ok": False,
            "error": f"Couldn't fetch the {source} data ({type(e).__name__}: {e}).",
        })

    if not rows:
        return ToolResult(output={
            "ok": False,
            "error": (f"No {source} matched those filters, so there's nothing "
                      "to export. Say so rather than sending an empty file."),
            "filters": filters,
        })

    flat = [flatten_row(r) for r in rows]
    columns = columns_from_rows(flat, preferred=spec["preferred"])
    try:
        content = build_workbook([{
            "name": spec["sheet"], "rows": flat, "columns": columns,
        }])
    except SpreadsheetError as e:
        return ToolResult(output={"ok": False, "error": str(e)})

    bits = [source] + [str(v) for v in filters.values() if isinstance(v, str)]
    if isinstance(filters.get("status"), list):
        bits += filters["status"]
    fname = safe_filename(inp.filename or "-".join(bits).lower().replace(" ", "-"))

    up = upload_to_slack(
        content=content,
        filename=fname,
        channel_id=ctx.get("channel_id") or "",
        thread_ts=ctx.get("thread_ts"),
        initial_comment=inp.comment,
    )
    if not up.get("ok"):
        return ToolResult(output={"ok": False, "error": up.get("error")})

    out: dict[str, Any] = {
        "ok": True,
        "filename": up["filename"],
        "permalink": up.get("permalink"),
        "source": source,
        "rows": len(rows),
        "columns": len(columns),
        "filters": filters,
        "note": ("The file is already posted in the conversation. Confirm it "
                 "in one short line with the row count; do not list the rows."),
    }
    if ignored:
        out["ignored_filters"] = (
            f"{', '.join(ignored)} don't apply to the {source} export and "
            "were not used -- mention that if the user asked for them.")
    if len(rows) >= _EXPORT_MAX_ROWS:
        out["truncated_at"] = _EXPORT_MAX_ROWS
    total = (last or {}).get("total_matching")
    if isinstance(total, int):
        out["total_matching"] = total
        if total > len(rows):
            out["incomplete"] = (
                f"Only {len(rows):,} of {total:,} rows were exported. Say so.")
    return ToolResult(output=out)


# ---------------------------------------------------------------------------
# External web lookups (Gemini + Google Search grounding)
# ---------------------------------------------------------------------------
#
# Always-on for Todd, unlike the web chat where the same `web_search` is
# gated behind a per-message toggle. Slack has nowhere to put a toggle, and
# a magic phrase the user has to know ("search the web for...") is a worse
# answer than a tool the model reaches for only when internal data runs
# out. The internal-first ordering is therefore enforced in the SYSTEM
# PROMPT (see chat_slack/orchestrator.py) rather than by withholding the
# schema, and both tools refuse to send email addresses or Ion deal
# codenames outbound regardless of what the model asks for.
#
# Registered last so the tool order -- which prompt caching keys on --
# stays stable for every internal tool above.
register_web_tools(slack_registry)
register_company_web_tools(slack_registry)
