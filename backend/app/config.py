"""Settings loaded from .env / environment. Single source of truth for config."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    # Connection pool size for psycopg2.pool.ThreadedConnectionPool. Default
    # 10 covers normal SSE concurrency on a single Render API instance
    # (each chat turn holds 1-3 conns briefly). Bump in env if SSE streams
    # ever queue waiting for a conn.
    db_pool_min: int = 1
    db_pool_max: int = 10

    anthropic_api_key: str = ""

    azure_client_id: str = ""
    azure_client_secret: str = ""
    azure_tenant_id: str = ""
    flask_secret_key: str = ""

    # Todd the Walrus -- Slack app credentials. Empty in local dev means
    # the Slack endpoints will refuse signed requests (good: prevents
    # accidental staging access without proper config).
    slack_signing_secret: str = ""
    slack_bot_token: str = ""
    # User token (xoxp) for READING the weekly "Deals Tracker" xlsx files
    # in #existing_pipeline. The bot token can't (no files:read scope, not
    # a channel member); the user token has files:read. Used only by
    # chat_slack/deals_tracker.py; blank disables find_new_deals_to_discuss.
    slack_user_token: str = ""

    allowed_origins: str = ""

    # ToltIQ -- Phase 4 ad-hoc query against built data rooms. Read at
    # runtime by services/toltiq_adhoc.py via os.environ; declared here
    # only so pydantic-settings doesn't whine on the .env load. Blank
    # values cause the chat tool to surface a friendly "not configured"
    # message.
    toltiq_base_url: str = ""
    toltiq_api_key: str = ""
    toltiq_org_id: str = ""

    # OpenAI -- query-time embeddings for hybrid org search (and later
    # document search). Empty value forces search_organizations to
    # silently fall back to trigram-only (no error, just no semantic
    # leg) so local dev without a key still functions.
    openai_api_key: str = ""

    # Gemini -- powers the opt-in `web_search` tool in chat (Google Search
    # grounding). Empty value forces the tool to return a friendly
    # "not configured" message rather than 500ing.
    gemini_api_key: str = ""
    # Corp-billed Gemini key; preferred over gemini_api_key when set. The
    # old key is free-tier and has NO allowance (`limit: 0`) on Pro-class
    # models, so the grounded web tools only work on this one.
    gemini_api_key_billed: str = ""

    # deal_cloud_enhancer internal API -- shared secret + base URL for
    # the /internal/document-body/{id} endpoint that triggers lazy
    # extraction of a document's text body. Empty values cause
    # read_document to surface a friendly "not configured" message.
    dce_internal_url: str = ""
    dce_internal_secret: str = ""

    # Public webpage that reads the SAME scenario_agent schema this MCP
    # writes to (Load Deal / ?org_id= deep link auto-loads a company's
    # current strategy+eventuality breakdown). Not a secret -- defaulted
    # directly rather than left blank -- used by finalize_strategy_
    # agreement/run_scenario_simulation to hand the analyst a direct link
    # to view/continue a deal there once it's been agreed/simulated.
    deal_scenario_modeler_url: str = "https://deal-scenario-modeler.onrender.com"

    @property
    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
