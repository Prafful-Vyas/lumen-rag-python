"""Black-box smoke test against a running stack (API + arq worker + Mongo + Redis): logs in as the
seeded demo user, uploads a document, waits for the worker to move it through every stage, then
asks a question over SSE and checks the answer is cited.

Usage: python scripts/smoke.py [API_URL]   (run `python -m lumen.seed` first)"""

import json
import sys
import time

import httpx

API = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
DOC = b"""# Acme Remote Work Policy

## Eligibility

Employees may work remotely up to 3 days per week with manager approval.

## Equipment

The company reimburses home-office equipment up to $500 per year.
"""


def main() -> None:
    login = httpx.post(f"{API}/auth/login", json={"email": "demo@lumen.local", "password": "Demo1234!"}, timeout=30)
    login.raise_for_status()
    h = {"Authorization": f"Bearer {login.json()['token']}"}

    doc = httpx.post(
        f"{API}/documents", files={"file": ("remote-policy.md", DOC, "text/markdown")}, headers=h, timeout=30
    )
    doc.raise_for_status()
    doc_id = doc.json()["id"]
    print(f"uploaded {doc_id}: status={doc.json()['status']}")

    seen = []
    for _ in range(60):
        d = httpx.get(f"{API}/documents/{doc_id}", headers=h, timeout=30).json()
        if not seen or seen[-1] != d["status"]:
            seen.append(d["status"])
        if d["status"] in ("ready", "failed"):
            break
        time.sleep(0.25)
    print("statuses observed:", " -> ".join(seen))
    assert d["status"] == "ready", d

    events: list[tuple[str, dict]] = []
    with httpx.stream(
        "POST", f"{API}/query", json={"question": "How many days per week can I work remotely?"}, headers=h, timeout=60
    ) as res:
        event = None
        for line in res.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, json.loads(line[6:])))
    kinds = [e for e, _ in events]
    answer = "".join(d["t"] for e, d in events if e == "token")
    sources = next(d for e, d in events if e == "sources")["sources"]
    print("events:", kinds[0], "->", f"{kinds.count('token')} tokens", "->", kinds[-1])
    print("top source:", sources[0]["documentTitle"], "p.", sources[0]["pageNumber"], "|", sources[0]["snippet"][:60])
    print("answer:", answer)
    assert kinds[0] == "sources" and kinds[-1] == "done" and "[1]" in answer

    httpx.delete(f"{API}/documents/{doc_id}", headers=h, timeout=30).raise_for_status()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
