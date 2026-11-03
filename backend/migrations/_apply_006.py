"""One-shot apply for 006_portfolio_via_manager.sql. Reads DATABASE_URL
from the deal_cloud_enhancer .env file (same pattern as _apply_005.py).

Usage:
    python _apply_006.py
"""
import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Walk up so this also works from a git worktree (one level deeper).
for parent in ROOT.parents:
    if (parent / "deal_cloud_enhancer" / ".env").exists():
        load_dotenv(parent / "deal_cloud_enhancer" / ".env")
        break

sql = (ROOT / "006_portfolio_via_manager.sql").read_text(encoding="utf-8")

conn = psycopg2.connect(os.environ["DATABASE_URL"])
try:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("MIGRATION APPLIED")

    with conn.cursor() as cur:
        cur.execute("SELECT dealcloud.org_portfolio_status(ARRAY[11957])")
        print(json.dumps(cur.fetchone()[0], indent=2)[:800])
finally:
    conn.close()
