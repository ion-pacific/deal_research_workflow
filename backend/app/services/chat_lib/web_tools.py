"""External web tools for chat. Backed by Gemini 3.1 Pro with Google
Search grounding -- same pattern slack_helper/run_ai_updates.py uses for
the AI news digest.

Two tools, registered separately because the two surfaces want different
subsets:

  * `web_search(query)` via `register_web_tools` -- freeform factual
    lookup. In the drw web chat this is registered into a phase registry
    only when the user has explicitly enabled external sources via the
    per-message toggle, so a turn with the toggle off sees no
    `web_search` schema in the tools array at all.
  * `research_company_web(company)` via `register_company_web_tools` --
    a structured public profile of ONE company (what it does, funding,
    people, news, flags). Exists because the recurring ask is "tell me
    about this company, we have nothing internal on them", and letting
    the model improvise that from freeform searches gave inconsistent,
    patchily-sourced answers. Todd (chat_slack) registers both; the web
    chat currently registers only `web_search`.

Output shape: a brief text answer synthesised by Gemini plus a `sources`
list of `{url, title}` so the Claude orchestrator can cite them as
clickable markdown links. We do NOT scrape page bodies -- the synthesised
answer + snippets that Gemini's grounding already gives are enough to
answer most factual lookups without a separate fetch tool.

## Egress

Every call here leaves our perimeter for Google. The PII scrubber that
wraps the MCP server is OUTPUT-only, so it does nothing for an outbound
query -- a tool that sends data out has to police its own INPUT. Hence
`_check_egress`, which hard-refuses queries carrying the two things that
are both confidential and useless to a search engine: email addresses,
and Ion deal codenames ("Project Ostrich"). The model gets told to
rephrase rather than silently having its query rewritten.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from ...config import settings
from .tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Gemini's pricing is favourable for grounded calls relative to Claude's
# `web_search` server-tool, and we already depend on it for slack_helper.
# Pro because we want the same quality bar as the consolidation work that
# validated Pro >> Flash Lite > Sonnet for grounded judgement tasks.
# gemini-2.5-pro was retired by Google (2026-10); 3.1 Pro is its named
# successor and is preview-only -- there is no stable 3.x Pro yet.
GEMINI_MODEL = "gemini-3.1-pro-preview"

# How many grounding chunks (URL + title pairs) to surface to Claude.
# Keep it tight -- Claude only needs enough to cite, not a wall of refs.
MAX_SOURCES = 8

# A grounded Pro call is slow (10-40s is normal, multi-search profiles
# longer). Cap it so one hung request can't sit on a Slack turn forever --
# Todd's tool loop has no wall clock of its own, and the user is watching a
# "searching..." breadcrumb with no way to cancel.
REQUEST_TIMEOUT_MS = 120_000

# More sources on a company profile: it's several searches' worth of
# material and the model should be able to cite per-section.
MAX_PROFILE_SOURCES = 15

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Ion's deal-naming convention: "Project <Codename>". These are internal
# names for live transactions -- searching one tells Google what we're
# working on and returns nothing useful anyway.
#
# Matched case-insensitively, which knowingly over-blocks generic English
# ("project management software"). That direction is the cheap one: the
# model is told how to rephrase and loses one call, whereas a leaked
# codename can't be recalled. The precise alternative -- matching against
# dealcloud.deal.name -- would put a DB dependency in chat_lib, which is
# deliberately persistence-agnostic; revisit via ctx if the false
# positives ever actually bite.
_CODENAME_RE = re.compile(r"\bproject\s+[A-Z][A-Za-z-]{2,}", re.IGNORECASE)


def _check_egress(text: str) -> str | None:
    """Return a refusal message when `text` must not leave the perimeter,
    else None. Deliberately narrow: this catches the two categories we
    know are both confidential and worthless as search terms, and lets
    everything else (company names, sectors, public figures) through --
    a broad filter here would just make the tool unusable."""
    if _EMAIL_RE.search(text):
        return (
            "Refused: the query contains an email address. Never send "
            "personal email addresses to an external search engine. "
            "Re-run with the company or person's NAME instead."
        )
    if m := _CODENAME_RE.search(text):
        return (
            f"Refused: the query contains what looks like an internal Ion "
            f"deal codename ({m.group(0)!r}). Codenames are confidential "
            f"and meaningless to a search engine. Re-run with the actual "
            f"company name -- use find_organizations or list_deals to "
            f"resolve the codename to a company first. (If {m.group(0)!r} "
            f"is genuinely a public phrase and not one of our codenames, "
            f"just rephrase the query without the word 'project'.)"
        )
    return None


def _grounded_call(prompt: str, *, max_sources: int) -> tuple[str, list[dict], dict | None]:
    """One Gemini call with Google Search grounding. Returns
    `(answer, sources, error_output)` -- when `error_output` is not None
    it's a ready-to-return ToolResult output dict and the other two are
    empty. Every failure is reported to the model as data (never raised)
    so a web hiccup degrades to "answer from internal data" instead of
    killing the turn."""
    api_key = settings.gemini_api_key_billed or settings.gemini_api_key
    if not api_key:
        return "", [], {
            "error": "Web search is not configured on this server "
                     "(GEMINI_API_KEY missing). Tell the user external "
                     "sources are unavailable right now and answer from "
                     "internal data only.",
        }

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "", [], {
            "error": "google-genai package not installed on the server. "
                     "Web search disabled.",
        }

    try:
        # http_options is guarded: the pinned floor in requirements.txt is
        # old enough that HttpOptions.timeout may not exist there, and no
        # timeout beats no web search.
        try:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
        except Exception:
            logger.warning("HttpOptions(timeout=...) unsupported; "
                           "falling back to no request timeout")
            client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception as e:
        logger.exception("grounded Gemini call failed")
        return "", [], {"error": f"Web search failed: {type(e).__name__}: {e}"}

    return (response.text or "").strip(), _extract_sources(response, max_sources), None


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Search query in natural language. Gemini with Google Search "
            "grounding will fetch real-time web results and synthesise a "
            "brief answer plus citations. Keep the query focused -- one "
            "factual question per call. Examples: 'What is Soma Capital "
            "AUM in 2025?', 'Who is the CEO of Bitmovin?'."
        ),
        min_length=1,
        max_length=400,
    )


def web_search(inp: WebSearchInput, ctx: dict) -> ToolResult:
    """Synchronous Gemini call -- chat_lib runs this in asyncio.to_thread
    so the SSE stream isn't blocked. Returns a structured object so the
    orchestrator can cite distinctly from internal sources."""
    if refusal := _check_egress(inp.query):
        return ToolResult(output={"error": refusal})

    answer, sources, err = _grounded_call(inp.query, max_sources=MAX_SOURCES)
    if err is not None:
        return ToolResult(output=err)

    return ToolResult(output={
        "answer": answer,
        "sources": sources,
        "note": (
            "These are EXTERNAL web sources. Cite them in your reply as "
            "markdown links: `[title](url)`. Distinguish them from "
            "internal data-room sources by mentioning they came from a "
            "web search."
        ),
    })


def _extract_sources(response: Any, max_sources: int = MAX_SOURCES) -> list[dict]:
    """Pull (title, url) pairs out of Gemini's grounding metadata. The
    SDK shape is a little nested -- candidate -> grounding_metadata ->
    grounding_chunks[] -> web.{uri, title}. Be defensive: any field can
    be None on edge-case responses."""
    out: list[dict] = []
    try:
        candidates = response.candidates or []
        if not candidates:
            return out
        gm = getattr(candidates[0], "grounding_metadata", None)
        if gm is None:
            return out
        chunks = getattr(gm, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            url = getattr(web, "uri", None) or ""
            title = getattr(web, "title", None) or url
            if not url:
                continue
            out.append({"url": url, "title": title})
            if len(out) >= max_sources:
                break
    except Exception:
        logger.exception("failed to extract grounding sources")
    return out


def register_web_tools(registry: ToolRegistry) -> None:
    """Add the web_search tool to an existing registry. Caller should
    clone the cached phase registry first so we don't mutate the
    module-level base across turns."""
    registry.tool(
        "web_search",
        (
            "OPTIONAL external source. Search the public web via Gemini "
            "with Google Search grounding. Returns a synthesised answer "
            "plus a list of `{url, title}` sources. Use this only when "
            "the internal data room / dossier / database genuinely "
            "doesn't have the answer (e.g. recent news, public market "
            "data, regulatory filings, third-party coverage). Always "
            "cite the returned URLs as markdown links `[title](url)` "
            "and mention that the information came from a web search."
        ),
        WebSearchInput,
    )(web_search)


# ---------------------------------------------------------------------------
# Structured company profile
# ---------------------------------------------------------------------------

class ResearchCompanyWebInput(BaseModel):
    company: str = Field(
        ...,
        description=(
            "The company's name, as publicly known. Prefer the legal or "
            "brand name over an internal shorthand. Do NOT pass an Ion "
            "deal codename -- resolve it to the real company first."
        ),
        min_length=1,
        max_length=160,
    )
    website: str | None = Field(
        None,
        description=(
            "The company's domain (e.g. 'moove.io') when you know it. "
            "Strongly recommended -- it's what separates the right "
            "company from same-named ones, which is the main failure "
            "mode of this tool. Our own records often carry it (a "
            "contact's email domain is a reliable source); never invent "
            "one you haven't seen."
        ),
        max_length=120,
    )
    context: str | None = Field(
        None,
        description=(
            "Short PUBLIC disambiguator when you have no domain -- "
            "sector, HQ country, or founding era ('Singapore digital "
            "bank', 'Nigerian fleet-financing'). Public facts only: this "
            "string is sent to Google, so never put deal, valuation, or "
            "relationship details here."
        ),
        max_length=200,
    )
    focus: str | None = Field(
        None,
        description=(
            "Optional extra angle to dig into on top of the standard "
            "profile -- e.g. 'latest funding round and valuation', "
            "'who their CFO is', 'any litigation'. Leave empty for the "
            "general 'who are these people' lookup."
        ),
        max_length=300,
    )


# Fixed section list, so two lookups of two different companies come back
# comparable and the model isn't inventing a shape each time. "Not found"
# is required explicitly: a grounded model with no instruction will fill a
# gap with plausible prose, and for a thin-data company MOST sections are
# genuinely gaps.
_PROFILE_SECTIONS = (
    "1. WHAT THEY DO -- one short paragraph: product, customers, "
    "business model.",
    "2. BASICS -- founded year, HQ city/country, approximate headcount, "
    "website, public/private status (and ticker if listed).",
    "3. FUNDING -- each disclosed round: date, stage, amount, lead and "
    "notable investors; total raised; latest known valuation. Say if "
    "funding is undisclosed.",
    "4. OWNERSHIP / EXITS -- parent, major shareholders, any acquisition, "
    "IPO, merger, or announced exit process.",
    "5. KEY PEOPLE -- founders, current CEO/CFO/CTO, and any senior hire "
    "or departure in the last 18 months (with dates).",
    "6. RECENT DEVELOPMENTS -- dated bullets from the last 18 months: "
    "launches, expansion, partnerships, financial results.",
    "7. FLAGS -- anything an investor would want to know: litigation, "
    "regulatory action, insolvency or restructuring, mass layoffs, "
    "fraud allegations, sanctions. Say 'none found' if you find none.",
)

_PROFILE_RULES = (
    "Rules:\n"
    "- Ground every factual claim in a search result. If you cannot "
    "source a section, write 'Not found' for it -- do NOT guess, "
    "estimate, or pad. A mostly-'Not found' profile is a useful and "
    "acceptable answer.\n"
    "- Attach a date to every event and figure, and name the currency "
    "on every amount.\n"
    "- If the search results look like they describe a DIFFERENT company "
    "with a similar name, say so explicitly at the top instead of "
    "blending the two.\n"
    "- Be terse. Bullets, not prose. No preamble, no closing summary."
)


def research_company_web(inp: ResearchCompanyWebInput, ctx: dict) -> ToolResult:
    """Public-web profile of one company, in a fixed section layout.
    Synchronous -- chat_lib runs handlers in asyncio.to_thread."""
    probe = " ".join(filter(None, (inp.company, inp.website, inp.context, inp.focus)))
    if refusal := _check_egress(probe):
        return ToolResult(output={"error": refusal})

    ident = inp.company
    if inp.website:
        ident += f" (website: {inp.website})"
    if inp.context:
        ident += f" ({inp.context})"

    prompt = (
        f"Research the company {ident} using web search and produce a "
        f"factual profile. Today is {date.today().isoformat()}.\n\n"
        + "\n".join(_PROFILE_SECTIONS)
        + (f"\n8. FOCUS -- {inp.focus}\n" if inp.focus else "\n")
        + "\n" + _PROFILE_RULES
    )

    profile, sources, err = _grounded_call(prompt, max_sources=MAX_PROFILE_SOURCES)
    if err is not None:
        return ToolResult(output=err)
    if not profile:
        return ToolResult(output={
            "company": inp.company,
            "error": (
                f"The web search returned nothing usable for "
                f"'{inp.company}'. Tell the user we couldn't find public "
                f"information under that name and ask for the company's "
                f"website or country to narrow it down -- don't fill the "
                f"gap from prior knowledge."
            ),
        })

    return ToolResult(output={
        "company": inp.company,
        "website_hint": inp.website,
        "profile": profile,
        "sources": sources,
        "note": (
            "This is EXTERNAL, public-web information -- it is NOT from "
            "our records and has not been verified by anyone at Ion. Say "
            "so when you use it, and keep it visibly separate from what "
            "the internal tools returned. Cite sources as markdown links "
            "`[title](url)`. Sections reading 'Not found' mean the search "
            "found nothing -- report them as unknown rather than filling "
            "them in yourself. Where the profile conflicts with our "
            "internal data, flag the conflict; don't silently prefer "
            "either side."
        ),
    })


def register_company_web_tools(registry: ToolRegistry) -> None:
    """Add `research_company_web`. Separate from `register_web_tools` so
    the web chat's per-message toggle keeps exposing exactly the one tool
    it always has, while Todd can opt into both."""
    registry.tool(
        "research_company_web",
        (
            "EXTERNAL public-web profile of ONE company: what it does, "
            "founding/HQ/headcount, funding rounds and investors, "
            "ownership and exits, key people and recent senior moves, "
            "dated recent developments, and risk flags (litigation, "
            "insolvency, layoffs, regulatory). Backed by Gemini with "
            "Google Search grounding; returns a sectioned `profile` plus "
            "`sources` to cite. USE THIS when the internal tools come up "
            "thin -- find_organizations finds no match, or the org exists "
            "but has almost no documents, contacts, or deal history -- and "
            "whenever the user asks what a company actually does or who "
            "backs it. Pass `website` when you know the domain; same-named "
            "companies are the main way this goes wrong. Sections the "
            "search couldn't source come back as 'Not found' -- that is a "
            "real answer, report it as unknown. Nothing here is verified "
            "by Ion: always label it as a web lookup, keep it separate "
            "from internal facts, and cite the URLs."
        ),
        ResearchCompanyWebInput,
    )(research_company_web)
