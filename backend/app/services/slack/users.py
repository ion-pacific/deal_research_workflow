"""Slack user <-> Ion email lookup.

Slack's users.info / users.lookupByEmail are rate-limited (~100/min,
Tier 4) so both directions cache in-process with a 1-hour TTL. Cache
misses on cold worker / cache expiry hit Slack's API; subsequent lookups
are free.

A user without an @ionpacific.com email (external collaborator, bot,
profile email not visible to the bot) is rejected by `is_ion_email`.
"""
import logging
import time
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from ...config import settings

logger = logging.getLogger(__name__)

_CLIENT: WebClient | None = (
    WebClient(token=settings.slack_bot_token) if settings.slack_bot_token else None
)
_CACHE: dict[str, tuple[str | None, float]] = {}
_TTL_SECONDS = 60 * 60

# Separate cache for the reverse (email -> user_id) direction -- keyed
# opposite of _CACHE above, same TTL/miss semantics.
_EMAIL_TO_ID_CACHE: dict[str, tuple[str | None, float]] = {}


def slack_user_to_email(slack_user_id: str) -> Optional[str]:
    """Returns the Slack user's primary email, or None if not visible."""
    cached = _CACHE.get(slack_user_id)
    if cached and time.time() - cached[1] < _TTL_SECONDS:
        return cached[0]

    if _CLIENT is None:
        return None

    try:
        resp = _CLIENT.users_info(user=slack_user_id)
        email = resp["user"].get("profile", {}).get("email")
    except SlackApiError:
        email = None

    _CACHE[slack_user_id] = (email, time.time())
    return email


# user_id -> display name, for rendering `<@U...>` mentions in stored
# Slack text (read_slack_conversation). Same TTL/miss semantics as _CACHE.
_NAME_CACHE: dict[str, tuple[str | None, float]] = {}


def slack_user_display_name(slack_user_id: str) -> Optional[str]:
    """The user's real name (falling back to display name), or None when
    there's no bot token or the lookup fails -- never raises."""
    cached = _NAME_CACHE.get(slack_user_id)
    if cached and time.time() - cached[1] < _TTL_SECONDS:
        return cached[0]

    if _CLIENT is None:
        return None

    try:
        user = _CLIENT.users_info(user=slack_user_id)["user"]
        profile = user.get("profile", {})
        name = (profile.get("real_name") or user.get("real_name")
                or profile.get("display_name") or user.get("name")) or None
    except Exception as e:  # noqa: BLE001 - fail soft, see docstring
        logger.info("slack_user_display_name: lookup failed for %s: %s", slack_user_id, e)
        name = None

    _NAME_CACHE[slack_user_id] = (name, time.time())
    return name


def is_ion_email(email: str | None) -> bool:
    return bool(email) and email.lower().endswith("@ionpacific.com")


def email_to_slack_user_id(email: str) -> Optional[str]:
    """Outbound direction: resolve an Ion email to its Slack user_id, for
    DMing someone we only know by email (e.g. a data_room_build_job's
    requested_by_email). Reference: tech_task_management/notify.py's
    _resolve_user_id does the same users_lookupByEmail call for the
    AI-agent runner's Slack notifications. Returns None on any lookup
    failure (unknown email, no Slack account, API error) -- never raises."""
    cached = _EMAIL_TO_ID_CACHE.get(email)
    if cached and time.time() - cached[1] < _TTL_SECONDS:
        return cached[0]

    if _CLIENT is None:
        return None

    try:
        resp = _CLIENT.users_lookupByEmail(email=email)
        user_id = resp["user"]["id"]
    except Exception as e:  # noqa: BLE001 - fail soft, see notify_slack_dm docstring
        logger.info("email_to_slack_user_id: lookup failed for %s: %s", email, e)
        user_id = None

    _EMAIL_TO_ID_CACHE[email] = (user_id, time.time())
    return user_id


def notify_slack_dm(email: str, text: str) -> bool:
    """Best-effort DM `email` on Slack -- never raises, same fail-soft
    contract as tech_task_management/notify.py's notify(): no
    SLACK_BOT_TOKEN configured, or the email doesn't resolve to a Slack
    user, just logs and returns False. Callers (e.g. the data-room-build
    job-finished handler) should treat this purely as a side effect, not
    something the request's success depends on."""
    if _CLIENT is None:
        logger.info("notify_slack_dm: SLACK_BOT_TOKEN not configured; skipping DM to %s", email)
        return False

    user_id = email_to_slack_user_id(email)
    if not user_id:
        logger.info("notify_slack_dm: could not resolve Slack user for %s; skipping DM", email)
        return False

    try:
        _CLIENT.chat_postMessage(channel=user_id, text=text, mrkdwn=True)
        return True
    except Exception as e:  # noqa: BLE001 - deliberately swallow, see docstring
        logger.warning("notify_slack_dm: chat_postMessage failed for %s: %s", email, e)
        return False
