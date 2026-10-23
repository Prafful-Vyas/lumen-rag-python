"""Streamlit front end for the Lumen API: sign in, upload documents and watch them move through
the ingestion stages, ask questions with streamed, cited answers, and see workspace usage.

Run: streamlit run ui/streamlit_app.py   (LUMEN_API_URL defaults to http://localhost:8000)"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API = os.environ.get("LUMEN_API_URL", "http://localhost:8000")
PUBLIC_API = os.environ.get("LUMEN_PUBLIC_API_URL", API)

st.set_page_config(page_title="Lumen", page_icon=":mag:", layout="wide")


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api(method: str, path: str, **kwargs: Any) -> httpx.Response:
    return httpx.request(method, f"{API}{path}", headers=headers(), timeout=60, **kwargs)


def stream_query(question: str, document_ids: list[str] | None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Reads the POST + SSE stream frame by frame (EventSource cannot send auth headers)."""
    body: dict[str, Any] = {"question": question}
    if document_ids:
        body["filters"] = {"documentIds": document_ids}
    with httpx.stream("POST", f"{API}/query", json=body, headers=headers(), timeout=120) as res:
        if res.status_code != 200:
            yield "error", {"message": res.read().decode() or f"HTTP {res.status_code}"}
            return
        event = None
        for line in res.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event:
                yield event, json.loads(line.removeprefix("data: "))
                event = None


def login_view() -> None:
    st.title("Lumen")
    st.caption("Semantic search over your team's documents, with cited answers.")
    login_tab, signup_tab, join_tab = st.tabs(["Log in", "Create workspace", "Join with invite"])
    with login_tab, st.form("login"):
        email = st.text_input("Email", value="demo@lumen.local")
        password = st.text_input("Password", type="password", help="Seeded demo password: Demo1234!")
        if st.form_submit_button("Log in"):
            finish_auth(httpx.post(f"{API}/auth/login", json={"email": email, "password": password}, timeout=30))
    with signup_tab, st.form("signup"):
        email = st.text_input("Email ")
        password = st.text_input("Password (8+ characters)", type="password")
        workspace = st.text_input("Workspace name")
        if st.form_submit_button("Create workspace"):
            payload = {"email": email, "password": password, "workspaceName": workspace}
            finish_auth(httpx.post(f"{API}/auth/signup", json=payload, timeout=30))
    with join_tab, st.form("join"):
        email = st.text_input("Email  ")
        password = st.text_input("Password  ", type="password")
        invite = st.text_input("Invite code")
        if st.form_submit_button("Join workspace"):
            payload = {"email": email, "password": password, "inviteCode": invite}
            finish_auth(httpx.post(f"{API}/auth/join", json=payload, timeout=30))


def finish_auth(res: httpx.Response) -> None:
    if res.status_code >= 400:
        st.error(res.json().get("detail", res.text))
        return
    data = res.json()
    st.session_state.token = data["token"]
    st.session_state.user = data["user"]
    st.session_state.workspace = data["workspace"]
    st.rerun()


def documents_view() -> None:
    st.subheader("Documents")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader(
            "Upload PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
        )
        if st.form_submit_button("Upload") and files:
            for f in files:
                res = api(
                    "POST", "/documents", files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")}
                )
                if res.status_code >= 400:
                    st.error(f"{f.name}: {res.json().get('detail', res.text)}")
            st.rerun()

    docs = api("GET", "/documents").json()
    if not docs:
        st.info("No documents yet. Upload one to get started.")
        return
    for d in docs:
        cols = st.columns([4, 2, 2, 1])
        link = f"{PUBLIC_API}/documents/{d['id']}/file?token={st.session_state.token}"
        cols[0].markdown(f"**[{d['title']}]({link})**  \n{d['pageCount']} pages, {d['chunkCount']} chunks")
        if d["status"] == "failed":
            cols[1].error(f"failed at {d['error']['stage']}: {d['error']['message'][:80]}")
        else:
            cols[1].progress(d["progress"] / 100, text=d["status"])
        cols[2].caption(
            f"ingest cost ${d['ingestCost']['usd']:.5f}" + (f", OCR pages {d['ocrPages']}" if d["ocrPages"] else "")
        )
        if cols[3].button("Delete", key=f"del-{d['id']}"):
            api("DELETE", f"/documents/{d['id']}")
            st.rerun()
    if any(d["status"] not in ("ready", "failed") for d in docs):
        time.sleep(1.5)
        st.rerun()


def ask_view() -> None:
    st.subheader("Ask")
    docs = [d for d in api("GET", "/documents").json() if d["status"] == "ready"]
    scope = st.multiselect("Limit to documents (optional)", options=docs, format_func=lambda d: d["title"])
    question = st.chat_input("Ask a question about your documents")
    if not question:
        return
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        sources_box, answer_box, meta_box = st.empty(), st.empty(), st.empty()
        text, done = "", None
        for event, data in stream_query(question, [d["id"] for d in scope] or None):
            if event == "sources":
                # Sources render before the first token arrives.
                if data["sources"]:
                    lines = [
                        f"[{s['marker']}] **{s['documentTitle']}**, p.{s['pageNumber']}"
                        + (f" - {s['section']}" if s["section"] else "")
                        for s in data["sources"]
                    ]
                    sources_box.markdown("Sources: " + " | ".join(lines))
            elif event == "token":
                text += data["t"]
                answer_box.markdown(text + " ▌")
            elif event == "done":
                done = data
            elif event == "error":
                st.error(data["message"])
        answer_box.markdown(text)
        if done:
            meta_box.caption(
                f"{done['latencyMs']:.0f} ms{' (cached)' if done['cached'] else ''}, "
                f"{done['usage']['promptTokens']}+{done['usage']['completionTokens']} tokens, ${done['usage']['usd']:.5f}"
                + (f", {done['invalidCitations']} invalid citation(s) removed" if done["invalidCitations"] else "")
            )
            st.session_state.last_query_id = done["queryId"]
    if st.session_state.get("last_query_id"):
        up, down, _ = st.columns([1, 1, 8])
        for col, value, label in ((up, "up", "Helpful"), (down, "down", "Not helpful")):
            if col.button(label):
                api("POST", f"/query/{st.session_state.last_query_id}/feedback", json={"feedback": value})
                st.toast("Thanks for the feedback")


def admin_view() -> None:
    st.subheader("Workspace")
    ws = api("GET", "/workspaces/me").json()
    st.write(f"**{ws['name']}**, invite code `{ws['inviteCode']}`")
    st.dataframe(api("GET", "/workspaces/members").json(), width="stretch")
    if st.session_state.user["role"] == "admin":
        usage = api("GET", "/admin/usage").json()
        a, b, c = st.columns(3)
        a.metric("Spend this month", f"${usage['spendUsd']:.4f}")
        b.metric("Monthly cap", f"${usage['monthlyCapUsd']:.2f}")
        c.metric("Queries", usage["queries"])
        if usage["nearCap"]:
            st.warning("Workspace is above 80% of its monthly cost cap.")
    st.subheader("Recent questions")
    st.dataframe(
        [
            {k: q[k] for k in ("question", "refused", "invalidCitations", "feedback", "createdAt")}
            for q in api("GET", "/query").json()
        ],
        width="stretch",
    )


if "token" not in st.session_state:
    login_view()
else:
    with st.sidebar:
        st.markdown(f"**{st.session_state.workspace['name']}**  \n{st.session_state.user['email']}")
        page = st.radio("Go to", ["Ask", "Documents", "Workspace"], label_visibility="collapsed")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()
    {"Ask": ask_view, "Documents": documents_view, "Workspace": admin_view}[page]()
