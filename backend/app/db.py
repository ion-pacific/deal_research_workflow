"""Postgres connection helper.

Uses a process-wide psycopg2.pool.ThreadedConnectionPool. FastAPI runs
sync route handlers in a thread pool, so blocking DB calls inside an
otherwise-async app are fine for V0 -- but every connect was paying a
~50-200ms TLS handshake to Neon. The pool keeps live conns warm so
each request reuses one rather than dialing fresh.

Every connection gets `SET search_path TO research, dealcloud, public`
on checkout (NOT just once on creation): Neon's connection pooler can
rotate the backing Postgres process between transactions (see
neon_pooler_search_path_drift), which drops the GUC. The 1 round-trip
on checkout is cheap (~5ms) and immune to that drift.

If a connection errors out mid-transaction we close it on putback so
the pool never hands out a poisoned conn to the next caller.
"""
from contextlib import contextmanager
from threading import Lock

import psycopg2
import psycopg2.extras
from psycopg2 import pool as _pg_pool

from .config import settings


_pool: _pg_pool.ThreadedConnectionPool | None = None
_pool_lock = Lock()


def _get_pool() -> _pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _pg_pool.ThreadedConnectionPool(
                    minconn=settings.db_pool_min,
                    maxconn=settings.db_pool_max,
                    dsn=settings.database_url,
                    # TCP keepalives, so an idle pooled conn is kept alive
                    # (or noticed dead) by the OS instead of being silently
                    # dropped by Neon / the network and discovered only
                    # when the next caller's first statement hits it.
                    **_KEEPALIVES,
                )
    return _pool


_KEEPALIVES = dict(keepalives=1, keepalives_idle=30,
                   keepalives_interval=10, keepalives_count=3)

#: How many dead pooled conns one checkout will discard before giving up.
_CHECKOUT_ATTEMPTS = 3


def _checkout(pool):
    """A pooled conn that has PROVEN it is alive, with session settings
    applied.

    ★★ `conn.closed` is not a liveness check. psycopg2 only sets it AFTER an
    operation fails, so a conn whose socket Neon (or the network) dropped
    while it sat idle in the pool still reads as open -- and the first
    statement on it raised "SSL SYSCALL error: EOF detected" straight into
    whatever tool asked for it (seen live on MCP ask_data_room, 2026-10-28).

    `_prepare`'s `SET search_path` is that first statement, and it runs
    before the caller has done anything, so failing it is free to retry:
    discard the dead conn and take another. Only connection-level errors
    are retried; anything else is a real problem and is raised."""
    for attempt in range(_CHECKOUT_ATTEMPTS):
        conn = pool.getconn()
        try:
            if conn.closed:
                raise psycopg2.InterfaceError("pooled connection already closed")
            _prepare(conn)
            return conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass
            if attempt == _CHECKOUT_ATTEMPTS - 1:
                raise
        except Exception:
            pool.putconn(conn, close=True)
            raise


def _prepare(conn) -> None:
    """Apply per-request session settings. Run on every pool checkout so
    Neon's pooler rotating the backend mid-pool-lifetime can't leave us
    with the wrong search_path."""
    with conn.cursor() as cur:
        cur.execute("SET search_path TO research, dealcloud, public")
    conn.commit()


@contextmanager
def get_conn():
    """Context-managed connection from the pool. Commits on clean exit,
    rolls back and discards on exception so the pool never reuses a
    broken conn."""
    pool = _get_pool()
    conn = _checkout(pool)
    poisoned = False
    try:
        yield conn
        conn.commit()
    except Exception:
        poisoned = True
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        # close=True drops the conn from the pool entirely; we do that on
        # exception OR if it's actually closed already.
        try:
            pool.putconn(conn, close=poisoned or conn.closed != 0)
        except Exception:
            pass


@contextmanager
def get_cursor():
    """Convenience for read paths: connection + dict cursor in one go."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur


def close_pool() -> None:
    """Close all pooled connections. Called from FastAPI shutdown so
    the worker doesn't leak FDs on reload."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            finally:
                _pool = None
