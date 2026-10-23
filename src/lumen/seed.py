"""Idempotent demo seed: creates (or reuses) a fixed demo workspace and admin login so a fresh
clone can be used immediately. The credentials are public; never seed an internet-facing deploy.

Usage: python -m lumen.seed"""

from __future__ import annotations

import asyncio

from lumen.api.deps import hash_password
from lumen.config import get_settings
from lumen.db import connect_mongo, ensure_indexes
from lumen.repositories import UserRepository, WorkspaceRepository

DEMO_EMAIL = "demo@lumen.local"
DEMO_PASSWORD = "Demo1234!"
DEMO_WORKSPACE_NAME = "Demo Workspace"


async def main() -> None:
    settings = get_settings()
    client, db = connect_mongo(settings)
    try:
        await ensure_indexes(db)
        users = UserRepository(db)
        if not await users.find_by_email(DEMO_EMAIL):
            workspace = await WorkspaceRepository(db, settings.workspace_default_monthly_cap_usd).create(
                DEMO_WORKSPACE_NAME
            )
            await users.create(DEMO_EMAIL, hash_password(DEMO_PASSWORD), str(workspace["_id"]), "admin")
        print(f"Lumen demo account ready: {DEMO_EMAIL} / {DEMO_PASSWORD} ({DEMO_WORKSPACE_NAME})")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
