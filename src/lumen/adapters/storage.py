from __future__ import annotations

import asyncio
from pathlib import Path


def storage_key_for(workspace_id: str, document_id: str, filename: str) -> str:
    safe = Path(filename).name.replace("\\", "_") or "upload"
    return f"{workspace_id}/{document_id}-{safe}"


class LocalDiskStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()

    def _resolve(self, key: str) -> Path:
        full = (self.root / key).resolve()
        if not full.is_relative_to(self.root) or full == self.root:
            raise ValueError(f"Refusing to access storage key outside root: {key}")
        return full

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._resolve(key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._resolve(key).read_bytes)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._resolve(key).unlink, missing_ok=True)


class S3Storage:
    """Production storage target, behind the same port as LocalDiskStorage. Enabled with
    STORAGE_BACKEND=s3; AWS_ENDPOINT_URL also lets it target MinIO or another S3-compatible store."""

    def __init__(
        self,
        bucket: str,
        region: str | None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
        )

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self.client.put_object, Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )

    async def get(self, key: str) -> bytes:
        res = await asyncio.to_thread(self.client.get_object, Bucket=self.bucket, Key=key)
        return await asyncio.to_thread(res["Body"].read)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=key)
