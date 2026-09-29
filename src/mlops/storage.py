"""DigitalOcean Spaces, the object store that holds the dataset and every model version.

Spaces speaks the S3 protocol, so the standard boto3 client works with a DigitalOcean
endpoint. Layout of the bucket:

    data/data_tickets.parquet               the source dataset
    data/x_german/...                        the translated German pool (output of the GPU job)
    models/<version>/router.joblib           one fitted router per training run
    models/<version>/metrics.json            its validation metrics and reference statistics
    models/latest.json                       pointer to the version production should serve

latest.json carries the artifact's SHA-256. The inference service refuses to load an
artifact whose hash does not match, so a file swapped in the bucket cannot silently
change what production serves (tampering, the T in STRIDE).

Credentials come from the environment, never from code:
    SPACES_REGION   e.g. tor1
    SPACES_BUCKET   e.g. team3-ticket-routing
    SPACES_KEY / SPACES_SECRET   a key scoped to this bucket, read-only on inference
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

LATEST = "models/latest.json"


def configured() -> bool:
    return all(os.environ.get(k) for k in ("SPACES_REGION", "SPACES_BUCKET", "SPACES_KEY", "SPACES_SECRET"))


def _client():
    import boto3

    region = os.environ["SPACES_REGION"]
    return boto3.session.Session().client(
        "s3", region_name=region, endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=os.environ["SPACES_KEY"], aws_secret_access_key=os.environ["SPACES_SECRET"])


def _bucket() -> str:
    return os.environ["SPACES_BUCKET"]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(local: str | Path, key: str) -> None:
    """Upload a file as a private object (the bucket has no public access)."""
    _client().upload_file(str(local), _bucket(), key, ExtraArgs={"ACL": "private"})


def download(key: str, local: str | Path) -> Path:
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(_bucket(), key, str(local))
    return local


def exists(key: str) -> bool:
    try:
        _client().head_object(Bucket=_bucket(), Key=key)
        return True
    except Exception:
        return False


def read_json(key: str) -> dict | None:
    try:
        obj = _client().get_object(Bucket=_bucket(), Key=key)
    except Exception:
        return None
    return json.loads(obj["Body"].read())


def write_json(key: str, payload: dict) -> None:
    _client().put_object(Bucket=_bucket(), Key=key, Body=json.dumps(payload, indent=2).encode(),
                         ContentType="application/json", ACL="private")


def latest() -> dict | None:
    """The pointer production follows: {"version", "key", "sha256", "promoted_at", "metrics"}."""
    return read_json(LATEST)


def fetch_verified(pointer: dict, cache_dir: str | Path) -> Path:
    """Download the artifact a pointer names into cache_dir and check its hash. Raises on mismatch."""
    local = Path(cache_dir) / pointer["version"] / "router.joblib"
    if not local.exists() or sha256(local) != pointer["sha256"]:
        download(pointer["key"], local)
    digest = sha256(local)
    if digest != pointer["sha256"]:
        local.unlink(missing_ok=True)
        raise ValueError(f"artifact {pointer['key']} hash {digest[:12]} does not match the pointer "
                         f"{pointer['sha256'][:12]}, refusing to load it")
    return local
