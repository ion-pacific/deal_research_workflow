"""Tests for app/auth.py. Tokens are signed with a throwaway RSA key and
the JWKS lookup is stubbed, so no network is needed.

Run from backend/:  DATABASE_URL=postgresql://x python -m pytest tests/test_auth.py
"""
import os
import time

os.environ.setdefault("DATABASE_URL", "postgresql://unused")

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app import auth
from app.config import settings

TENANT = "11111111-1111-1111-1111-111111111111"
CLIENT = "22222222-2222-2222-2222-222222222222"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _FakeJWKS:
    def get_signing_key_from_jwt(self, token):
        return type("K", (), {"key": KEY.public_key()})()


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.setattr(settings, "azure_tenant_id", TENANT)
    monkeypatch.setattr(settings, "azure_client_id", CLIENT)
    monkeypatch.setattr(settings, "entra_api_audience", "")
    monkeypatch.setattr(settings, "entra_required_scope", "access_as_user")
    monkeypatch.setattr(settings, "allowed_email_domain", "ionpacific.com")
    monkeypatch.setattr(auth, "_jwks_client", lambda: _FakeJWKS())


def _token(key=KEY, **over):
    now = int(time.time())
    claims = {
        "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
        "aud": f"api://{CLIENT}",
        "tid": TENANT,
        "iat": now,
        "exp": now + 3600,
        "scp": "access_as_user",
        "preferred_username": "Analyst@IonPacific.com",
        "name": "An Analyst",
    }
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256")


def _call(mode, monkeypatch, authorization=None, email=None):
    monkeypatch.setattr(settings, "auth_mode", mode)
    return auth.require_user(authorization=authorization, x_user_email=email)


def _rejects(mode, monkeypatch, **kw):
    with pytest.raises(HTTPException) as e:
        _call(mode, monkeypatch, **kw)
    assert e.value.status_code == 401


# --- happy paths -----------------------------------------------------------

@pytest.mark.parametrize("mode", ["entra", "both"])
def test_valid_token_accepted(mode, monkeypatch):
    u = _call(mode, monkeypatch, authorization=f"Bearer {_token()}")
    assert u.email == "analyst@ionpacific.com"
    assert u.name == "An Analyst"


def test_bare_client_id_audience_accepted(monkeypatch):
    u = _call("entra", monkeypatch, authorization=f"Bearer {_token(aud=CLIENT)}")
    assert u.email == "analyst@ionpacific.com"


def test_stub_mode_keeps_header(monkeypatch):
    assert _call("stub", monkeypatch, email="X@ionpacific.com").email == "x@ionpacific.com"


def test_both_mode_falls_back_to_header_without_bearer(monkeypatch):
    assert _call("both", monkeypatch, email="x@ionpacific.com").email == "x@ionpacific.com"


# --- the hole this change closes -----------------------------------------

def test_entra_mode_ignores_forged_header(monkeypatch):
    _rejects("entra", monkeypatch, email="ceo@ionpacific.com")


def test_bearer_wins_over_header_in_both_mode(monkeypatch):
    u = _call("both", monkeypatch, authorization=f"Bearer {_token()}", email="ceo@ionpacific.com")
    assert u.email == "analyst@ionpacific.com"


def test_invalid_bearer_never_downgrades_to_header(monkeypatch):
    _rejects("both", monkeypatch, authorization="Bearer not-a-jwt", email="ceo@ionpacific.com")


def test_unknown_mode_fails_closed(monkeypatch):
    _rejects("entr", monkeypatch, email="ceo@ionpacific.com")


# --- token validation ------------------------------------------------------

@pytest.mark.parametrize("over", [
    {"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},   # expired
    {"aud": "api://someone-else"},                                     # wrong audience
    {"iss": "https://login.microsoftonline.com/evil/v2.0"},            # wrong issuer
    {"tid": "33333333-3333-3333-3333-333333333333"},                   # other tenant
    {"scp": "User.Read"},                                              # missing scope
    {"scp": None},                                                     # app-only token
    {"preferred_username": "someone@gmail.com"},                       # outside domain
    {"preferred_username": "x@ionpacific.com.evil.io"},                # suffix trick
    {"preferred_username": None},                                      # no identity
])
def test_bad_claims_rejected(over, monkeypatch):
    _rejects("entra", monkeypatch, authorization=f"Bearer {_token(**over)}")


def test_wrong_signing_key_rejected(monkeypatch):
    _rejects("entra", monkeypatch, authorization=f"Bearer {_token(key=OTHER_KEY)}")


def test_alg_none_rejected(monkeypatch):
    t = jwt.encode({"preferred_username": "x@ionpacific.com"}, None, algorithm="none")
    _rejects("entra", monkeypatch, authorization=f"Bearer {t}")


def test_non_bearer_scheme_rejected(monkeypatch):
    _rejects("both", monkeypatch, authorization="Basic abc", email="x@ionpacific.com")


def test_unconfigured_tenant_rejects(monkeypatch):
    monkeypatch.setattr(settings, "azure_tenant_id", "")
    _rejects("entra", monkeypatch, authorization=f"Bearer {_token()}")


# --- v1 access tokens (manifest requestedAccessTokenVersion unset) --------

def _v1(**over):
    base = {"iss": f"https://sts.windows.net/{TENANT}/", "preferred_username": None,
            "upn": "Analyst@ionpacific.com", "ver": "1.0"}
    base.update(over)
    return _token(**base)


def test_v1_token_accepted_with_upn(monkeypatch):
    u = _call("entra", monkeypatch, authorization=f"Bearer {_v1()}")
    assert u.email == "analyst@ionpacific.com"


def test_v1_unique_name_fallback(monkeypatch):
    t = _v1(upn=None, unique_name="analyst@ionpacific.com")
    assert _call("entra", monkeypatch, authorization=f"Bearer {t}").email == "analyst@ionpacific.com"


def test_v1_issuer_of_other_tenant_rejected(monkeypatch):
    _rejects("entra", monkeypatch, authorization=f"Bearer {_v1(iss='https://sts.windows.net/evil/')}")


def test_rejection_is_logged_without_token(monkeypatch, capsys):
    t = _token(aud="api://someone-else")
    _rejects("entra", monkeypatch, authorization=f"Bearer {t}")
    out = capsys.readouterr().out
    assert "[auth] rejected bearer" in out and "api://someone-else" in out and t not in out
