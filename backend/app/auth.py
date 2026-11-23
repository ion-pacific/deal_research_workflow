"""User identity for protected routes.

`require_user` returns a UserCtx; route code never cares which backend
produced it. The backend is chosen by AUTH_MODE:

  stub   -- V0: trust the X-User-Email header. Anyone who can reach the
            API can impersonate anyone. Local dev only.
  both   -- migration: accept a valid Entra bearer token; if no
            Authorization header is sent, fall back to X-User-Email.
            A PRESENT-but-invalid bearer is always rejected (never
            silently downgraded to the header).
  entra  -- production: a valid Entra ID access token is REQUIRED.

Entra validation: RS256 signature against the tenant's JWKS, issuer =
https://login.microsoftonline.com/<tenant>/v2.0, audience = this API
(ENTRA_API_AUDIENCE, default api://<AZURE_CLIENT_ID>), delegated scope
ENTRA_REQUIRED_SCOPE present in `scp`, and the user's email in
ALLOWED_EMAIL_DOMAIN. The token's tenant must be ours (single-tenant).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Header, HTTPException, status

from .config import settings

log = logging.getLogger(__name__)

VALID_MODES = {"stub", "both", "entra"}


@dataclass
class UserCtx:
    email: str
    name: str | None = None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _mode() -> str:
    mode = (settings.auth_mode or "stub").strip().lower()
    if mode not in VALID_MODES:
        # Fail CLOSED on a typo -- never fall back to the stub.
        log.error("Unknown AUTH_MODE %r; treating as 'entra'", mode)
        return "entra"
    return mode


def _audiences() -> list[str]:
    if settings.entra_api_audience:
        return [a.strip() for a in settings.entra_api_audience.split(",") if a.strip()]
    cid = settings.azure_client_id
    return [f"api://{cid}", cid] if cid else []


@lru_cache(maxsize=1)
def _jwks_client() -> jwt.PyJWKClient:
    # PyJWKClient caches signing keys and refetches on an unknown `kid`
    # (Entra rotates keys), so this is a single network call per process
    # in the common case.
    return jwt.PyJWKClient(
        f"https://login.microsoftonline.com/{settings.azure_tenant_id}"
        "/discovery/v2.0/keys",
        cache_keys=True,
        lifespan=3600,
    )


def verify_entra_token(token: str) -> UserCtx:
    """Validate an Entra v2 access token for this API; return the user.
    Raises HTTPException(401) on any failure."""
    tenant = settings.azure_tenant_id
    audiences = _audiences()
    if not tenant or not audiences:
        log.error("Entra auth enabled but AZURE_TENANT_ID / audience not configured")
        raise _unauthorized("Authentication is not configured on this server.")
    try:
        key = _jwks_client().get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=audiences,
            issuer=f"https://login.microsoftonline.com/{tenant}/v2.0",
            options={"require": ["exp", "iat", "iss", "aud"]},
            leeway=60,
        )
    except jwt.PyJWTError as e:
        log.info("Rejected bearer token: %s: %s", type(e).__name__, e)
        raise _unauthorized("Invalid or expired token.") from e

    if claims.get("tid") != tenant:
        raise _unauthorized("Token is from another tenant.")

    required = settings.entra_required_scope
    if required and required not in (claims.get("scp") or "").split():
        raise _unauthorized(f"Token lacks the '{required}' scope.")

    email = (
        claims.get("preferred_username")
        or claims.get("email")
        or claims.get("upn")
        or ""
    ).lower().strip()
    domain = settings.allowed_email_domain.lower().lstrip("@")
    if not email or (domain and not email.endswith("@" + domain)):
        raise _unauthorized("Account is not allowed to use this app.")

    return UserCtx(email=email, name=claims.get("name"))


def _stub_user(x_user_email: str | None) -> UserCtx:
    if not x_user_email:
        raise _unauthorized("Missing credentials (Authorization bearer token or X-User-Email).")
    return UserCtx(email=x_user_email.lower().strip())


def require_user(
    authorization: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None, alias="X-User-Email"),
) -> UserCtx:
    mode = _mode()

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise _unauthorized("Authorization header must be 'Bearer <token>'.")
        if mode == "stub":
            # Stub mode has no verifier; don't pretend a token was checked.
            return _stub_user(x_user_email)
        return verify_entra_token(token.strip())

    if mode == "entra":
        raise _unauthorized("Sign-in required.")
    return _stub_user(x_user_email)
