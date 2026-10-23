from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import quote

from bson import ObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lumen.adapters.storage import storage_key_for
from lumen.api.deps import (
    AdminDep,
    AuthClaims,
    AuthDep,
    ContainerDep,
    hash_password,
    rate_limit,
    sign_token,
    verify_password,
)
from lumen.ingestion.extract import SUPPORTED_MIME_TYPES
from lumen.types import ChunkFilters

EXTENSION_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

auth_router = APIRouter(prefix="/auth", tags=["auth"])
documents_router = APIRouter(prefix="/documents", tags=["documents"])
query_router = APIRouter(prefix="/query", tags=["query"])
workspaces_router = APIRouter(prefix="/workspaces", tags=["workspaces"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])


def user_dto(u: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(u["_id"]), "email": u["email"], "role": u["role"]}


def workspace_dto(w: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(w["_id"]),
        "name": w["name"],
        "slug": w["slug"],
        "inviteCode": w["inviteCode"],
        "monthlyCapUsd": w["monthlyCapUsd"],
        "corpusVersion": w.get("corpusVersion", 0),
    }


def document_dto(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(d["_id"]),
        "title": d["title"],
        "filename": d["filename"],
        "mimeType": d["mimeType"],
        "sizeBytes": d["sizeBytes"],
        "pageCount": d.get("pageCount", 0),
        "status": d["status"],
        "progress": d.get("progress", 0),
        "error": d.get("error"),
        "chunkCount": d.get("chunkCount", 0),
        "failedChunks": d.get("failedChunks", 0),
        "ocrPages": d.get("ocrPages", []),
        "ingestCost": d.get("ingestCost"),
        "folderId": str(d["folderId"]) if d.get("folderId") else None,
        "createdAt": d["createdAt"].isoformat(),
        "readyAt": d["readyAt"].isoformat() if d.get("readyAt") else None,
    }


# Format check only: reserved domains such as .local (the demo login) must stay usable.
Email = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)]


class SignupIn(BaseModel):
    email: Email
    password: str = Field(min_length=8)
    workspaceName: str = Field(min_length=2)


class JoinIn(BaseModel):
    email: Email
    password: str = Field(min_length=8)
    inviteCode: str = Field(min_length=4)


class LoginIn(BaseModel):
    email: Email
    password: str


def _session(container, user: dict[str, Any], workspace: dict[str, Any]) -> dict[str, Any]:
    claims = AuthClaims(sub=str(user["_id"]), workspace_id=str(workspace["_id"]), role=user["role"])
    return {"token": sign_token(container, claims), "user": user_dto(user), "workspace": workspace_dto(workspace)}


@auth_router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(body: SignupIn, container: ContainerDep) -> dict[str, Any]:
    if await container.users.find_by_email(body.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    workspace = await container.workspaces.create(body.workspaceName)
    user = await container.users.create(body.email, hash_password(body.password), str(workspace["_id"]), "admin")
    return _session(container, user, workspace)


@auth_router.post("/join", status_code=status.HTTP_201_CREATED)
async def join(body: JoinIn, container: ContainerDep) -> dict[str, Any]:
    workspace = await container.workspaces.find_by_invite_code(body.inviteCode)
    if not workspace:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid invite code")
    if await container.users.find_by_email(body.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = await container.users.create(body.email, hash_password(body.password), str(workspace["_id"]), "member")
    return _session(container, user, workspace)


@auth_router.post("/login")
async def login(body: LoginIn, container: ContainerDep) -> dict[str, Any]:
    user = await container.users.find_by_email(body.email)
    if not user or not verify_password(body.password, user["passwordHash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    workspace = await container.workspaces.find_by_id(str(user["workspaceId"]))
    if not workspace:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "User references a workspace that no longer exists")
    return _session(container, user, workspace)


@auth_router.get("/me")
async def me(auth: AuthDep, container: ContainerDep) -> dict[str, Any]:
    user = await container.users.find_by_id(auth.sub)
    workspace = await container.workspaces.find_by_id(auth.workspace_id)
    if not user or not workspace:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found")
    return {"user": user_dto(user), "workspace": workspace_dto(workspace)}


def _resolve_mime(file: UploadFile) -> str:
    mime = (file.content_type or "").split(";")[0].strip()
    if mime in SUPPORTED_MIME_TYPES:
        return mime
    # Browsers and HTTP clients often send .md as application/octet-stream; trust the extension then.
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if ext in EXTENSION_MIME:
        return EXTENSION_MIME[ext]
    raise HTTPException(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type: {mime or ext}. Supported: PDF, DOCX, TXT, MD."
    )


@documents_router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    auth: AuthDep,
    container: ContainerDep,
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
    folderId: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    mime = _resolve_mime(file)
    data = await file.read(container.settings.max_upload_bytes + 1)
    if len(data) > container.settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File exceeds the upload size limit")
    if folderId and not ObjectId.is_valid(folderId):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid folderId")

    document_id = ObjectId()
    filename = file.filename or "upload"
    key = storage_key_for(auth.workspace_id, str(document_id), filename)
    await container.storage.put(key, data, mime)
    doc = await container.documents.create(
        document_id=document_id,
        workspace_id=auth.workspace_id,
        title=(title or "").strip() or filename,
        filename=filename,
        mime_type=mime,
        storage_key=key,
        size_bytes=len(data),
        uploaded_by=auth.sub,
        folder_id=folderId,
    )
    await container.queue.enqueue("extract", str(document_id), auth.workspace_id)
    fresh = await container.documents.find_by_id(auth.workspace_id, str(document_id))
    return document_dto(fresh or doc)


@documents_router.get("")
async def list_documents(
    auth: AuthDep,
    container: ContainerDep,
    folderId: str | None = None,
    uploadedBy: str | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[dict[str, Any]]:
    for value in (folderId, uploadedBy):
        if value and not ObjectId.is_valid(value):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid id filter")
    docs = await container.documents.list(
        auth.workspace_id, folder_id=folderId, uploaded_by=uploadedBy, after=after, before=before
    )
    return [document_dto(d) for d in docs]


async def _owned_document(container, auth: AuthClaims, document_id: str) -> dict[str, Any]:
    doc = await container.documents.find_by_id(auth.workspace_id, document_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@documents_router.get("/{document_id}")
async def get_document(document_id: str, auth: AuthDep, container: ContainerDep) -> dict[str, Any]:
    return document_dto(await _owned_document(container, auth, document_id))


@documents_router.get("/{document_id}/file")
async def get_document_file(document_id: str, auth: AuthDep, container: ContainerDep) -> Response:
    doc = await _owned_document(container, auth, document_id)
    data = await container.storage.get(doc["storageKey"])
    return Response(
        data,
        media_type=doc["mimeType"],
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(doc['filename'])}"},
    )


@documents_router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, auth: AuthDep, container: ContainerDep) -> Response:
    await _owned_document(container, auth, document_id)
    # Delist the chunks right away. Soft-deleting only the parent document would leave its chunks
    # fully searchable, an easy data leak to ship. Hard deletion is a retention job.
    await container.documents.soft_delete(auth.workspace_id, document_id)
    await container.chunks.delist_by_document(document_id)
    await container.workspaces.bump_corpus_version(auth.workspace_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class QueryFilters(BaseModel):
    documentIds: list[str] | None = None
    folderId: str | None = None
    after: datetime | None = None


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filters: QueryFilters | None = None


class FeedbackIn(BaseModel):
    feedback: Literal["up", "down"]
    note: str | None = Field(default=None, max_length=2000)


@query_router.post("", dependencies=[Depends(rate_limit(60, None, "query"))])
async def ask(body: QueryIn, auth: AuthDep, container: ContainerDep) -> StreamingResponse:
    """SSE over POST. The browser's EventSource cannot send an Authorization header or a JSON body,
    so clients read this stream with fetch/httpx instead."""
    f = body.filters or QueryFilters()
    ids = [i for i in (f.documentIds or []) + ([f.folderId] if f.folderId else []) if not ObjectId.is_valid(i)]
    if ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid id in filters")
    filters = ChunkFilters(document_ids=f.documentIds, folder_id=f.folderId, after=f.after)

    async def events() -> AsyncIterator[str]:
        try:
            async for evt in container.generation.stream_answer(auth.workspace_id, auth.sub, body.question, filters):
                yield f"event: {evt.event}\ndata: {json.dumps(evt.data)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc) or 'Unknown error generating answer'})}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform"}
    )


@query_router.get("")
async def recent_queries(auth: AuthDep, container: ContainerDep) -> list[dict[str, Any]]:
    return [
        {
            "id": str(q["_id"]),
            "question": q["question"],
            "answer": q["answer"],
            "refused": q["refused"],
            "invalidCitations": q["invalidCitations"],
            "feedback": q.get("feedback"),
            "usage": q.get("usage"),
            "createdAt": q["createdAt"].isoformat(),
        }
        for q in await container.queries.list_recent(auth.workspace_id, 50)
    ]


@query_router.post("/{query_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(query_id: str, body: FeedbackIn, auth: AuthDep, container: ContainerDep) -> Response:
    if not await container.queries.find_by_id(auth.workspace_id, query_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Query not found")
    await container.queries.set_feedback(auth.workspace_id, query_id, body.feedback, body.note or "")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@workspaces_router.get("/me")
async def my_workspace(auth: AuthDep, container: ContainerDep) -> dict[str, Any]:
    workspace = await container.workspaces.find_by_id(auth.workspace_id)
    if not workspace:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workspace not found")
    return workspace_dto(workspace)


@workspaces_router.get("/members")
async def members(auth: AuthDep, container: ContainerDep) -> list[dict[str, Any]]:
    return [
        {**user_dto(u), "createdAt": u["createdAt"].isoformat()}
        for u in await container.users.list_by_workspace(auth.workspace_id)
    ]


@admin_router.get("/usage")
async def usage(auth: AdminDep, container: ContainerDep) -> dict[str, Any]:
    workspace = await container.workspaces.find_by_id(auth.workspace_id)
    if not workspace:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workspace not found")
    spend, queries = await container.metering.get_usage(auth.workspace_id)
    cap = workspace["monthlyCapUsd"]
    return {
        "monthlyCapUsd": cap,
        "spendUsd": spend,
        "queries": queries,
        "capReached": spend >= cap,
        "nearCap": spend > cap * 0.8,
    }
