import os
import uuid

import pytest

from lumen.config import Settings


def _make_pdf(pages: list[str]) -> bytes:
    """Minimal valid PDF with one text line per page (no PDF-writing dependency needed)."""
    objects: list[bytes] = []
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + len(pages) * 2
    for i, text in enumerate(pages):
        content_id = 4 + i * 2
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def tiny_pdf() -> bytes:
    return _make_pdf(
        [
            "Section 4: Limitation of Liability does not apply to gross negligence.",
            "Section 9: Governing law is the State of Delaware.",
        ]
    )


@pytest.fixture(scope="session")
def integration_settings(tmp_path_factory) -> Settings:
    """Isolated database and Redis db per test session, against the MONGO_URI/REDIS_URL infra
    (defaults: localhost). Integration tests are skipped when either is unreachable."""
    import pymongo
    import redis

    mongo_base = os.environ.get("TEST_MONGO_URI", "mongodb://127.0.0.1:27017")
    redis_url = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
    try:
        pymongo.MongoClient(mongo_base, serverSelectionTimeoutMS=1500).server_info()
        redis.Redis.from_url(redis_url, socket_connect_timeout=1.5).ping()
    except Exception as exc:
        pytest.skip(f"MongoDB/Redis not reachable for integration tests: {exc}")
    return Settings(
        _env_file=None,
        mongo_uri=f"{mongo_base}/lumen_test_{uuid.uuid4().hex[:8]}",
        redis_url=redis_url,
        storage_root=str(tmp_path_factory.mktemp("storage")),
        openai_api_key=None,
        jwt_secret="test-secret-at-least-32-bytes-long!!",
        embedding_dimensions=256,
        query_rate_limit_per_minute=1000,
    )
