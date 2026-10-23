"""Isolated diagnostic helper; no model import, request rewriting or compare override."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import stat
import time
from pathlib import Path

CASES = {"generation_prefill_cuda_graph_sm120", "dense_fp8pb_dynamic_sm120"}
_METADATA = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model", "model.safetensors.index.json")
_MAX_BYTES = 16 * 1024**3
_MAX_SECONDS = 120

def _json(path, data):
    content = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("diagnostic JSON file cap")
    current = sum(p.stat().st_size for p in path.parent.glob("*.json") if p.is_file() and not p.is_symlink())
    if current + len(content) > 16 * 1024 * 1024:
        raise ValueError("diagnostic JSON directory cap")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(content)

def _regular_file(root, path):
    # The declared root may be a NAS/model symlink; its resolved directory is the boundary.
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("identity file escapes resolved model/tokenizer root") from error
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("identity input is not a regular file")
    return resolved

def _open_regular(root, path):
    resolved = _regular_file(root, path)
    relative = resolved.relative_to(root)
    # Resolve first, then reopen each component from an owned root FD without following
    # symlinks. This also rejects a path swapped to a symlink between check and open.
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        file_fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError("identity input is not a regular file")
            file = os.fdopen(file_fd, "rb")
        except BaseException:
            os.close(file_fd)
            raise
        return file, resolved
    finally:
        os.close(descriptor)

def _hash_file(root, path, remaining, deadline):
    file, resolved = _open_regular(root, path)
    with file:
        before = os.fstat(file.fileno())
        if before.st_size > remaining:
            return {"path": str(path), "resolved": str(resolved), "bytes": before.st_size, "sha256": None, "reason": "identity byte cap"}, 0
        digest = hashlib.sha256()
        size = 0
        while chunk := file.read(1024 * 1024):
            if time.monotonic() > deadline or size + len(chunk) > remaining:
                return {"path": str(path), "resolved": str(resolved), "bytes": before.st_size, "sha256": None, "reason": "identity time/byte cap"}, size
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(file.fileno())
    current = _regular_file(root, path).stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    stable = identity(before) == identity(after) == identity(current)
    return {"path": str(path), "resolved": str(resolved), "bytes": size, "sha256": digest.hexdigest() if stable else None, "stat_stable": stable}, size

def model_identity(model_path, tokenizer_path, *, byte_limit=_MAX_BYTES, seconds=_MAX_SECONDS):
    declared = [Path(model_path), Path(tokenizer_path or model_path)]
    roots = [root.resolve(strict=True) for root in declared]
    if any(not root.is_dir() for root in roots):
        raise ValueError("identity root must resolve to a directory")
    paths = {}
    for root in roots:
        for name in _METADATA:
            path = root / name
            if path.exists() or path.is_symlink():
                _regular_file(root, path)
                paths[path] = root
    root = roots[0]
    index = root / "model.safetensors.index.json"
    if index.exists():
        file, _ = _open_regular(root, index)
        with file:
            raw = file.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("weight index metadata size cap")
        for name in set(json.loads(raw)["weight_map"].values()):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("weight index escapes model root")
            paths[root / relative] = root
    else:
        for path in list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin")):
            paths[path] = root
    deadline = time.monotonic() + seconds
    remaining = byte_limit
    rows = []
    for path, root in sorted(paths.items()):
        row, used = _hash_file(root, path, remaining, deadline)
        rows.append(row)
        remaining -= used
    weights = [r for r in rows if r["path"].endswith((".safetensors", ".bin"))]
    versions = {}
    for name in ("torch", "rtp-kernel", "deep-gemm", "transformers", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    has_config = roots[0] / "config.json" in paths
    has_tokenizer = any(roots[1] / name in paths for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    complete = bool(weights) and has_config and has_tokenizer and all(r.get("sha256") for r in rows)
    return {"model_path": str(declared[0]), "tokenizer_path": str(declared[1]), "resolved_roots": [str(root) for root in roots], "files": rows, "versions": versions, "complete": complete, "scope": "current listed weights/config/tokenizer files only; not historical golden identity or arbitrary remote model code", "historical_golden_identity": None, "byte_cap": byte_limit, "seconds_cap": seconds}

def prepare_case(name, task, outputs):
    if os.environ.get("RTP_SMOKE_CAUSAL_OBSERVATION") != "1":
        return None
    if name not in CASES:
        raise ValueError("diagnostic scope must be one of the two original complete cases")
    root = Path(outputs).resolve(strict=True)
    directory = root / "smoke_causal" / name
    parent = root / "smoke_causal"
    parent.mkdir(mode=0o700, exist_ok=True)
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("diagnostic parent is not an owned direct directory")
    directory.mkdir(mode=0o700, exist_ok=False)
    # This exact fresh path is inherited by the normal server process.
    os.environ["RTP_SMOKE_CAUSAL_DIR"] = str(directory)
    _json(directory / "task-original.json", task)
    try:
        identity = model_identity(task["model_path"], task.get("tokenizer_path"))
        _json(directory / "model-identity.json", identity)
    except Exception as error:
        _json(directory / "identity-error.json", {"type": type(error).__name__, "message": str(error)})
    return directory

def record_exchange(qr, request, response):
    if os.environ.get("RTP_SMOKE_CAUSAL_OBSERVATION") != "1":
        return
    directory = Path(os.environ["RTP_SMOKE_CAUSAL_DIR"])
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("invalid diagnostic directory")
    index = qr.get("_query_idx")
    if not isinstance(index, int) or index < 0:
        raise ValueError("missing exact original query index")
    # Per-visit number preserves any original retry without changing retry policy.
    for attempt in range(100):
        path = directory / f"query-{index}-exchange-{attempt}.json"
        if not path.exists():
            _json(path, {"query_index": index, "endpoint": qr.get("endpoint"), "task_info": qr.get("_taskinfo_rel_path"), "request": request, "raw_response": response})
            return
    raise ValueError("diagnostic exchange cap")
