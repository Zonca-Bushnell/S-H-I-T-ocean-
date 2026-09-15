from __future__ import annotations

import argparse
import ftplib
import json
import os
import posixpath
import time
from pathlib import Path


HOST = "ftp-access.aviso.altimetry.fr"
REMOTE_PREFIX = "auxiliary/tide_model"

FES2014_FILES: dict[str, list[tuple[str, ...]]] = {
    "readme": [
        ("fes2014_elevations_and_load", "readme_fes2014_elevation_and_load_v1.2.txt"),
        ("fes2014a_currents", "readme_fes2014_currents_v1.2.txt"),
    ],
    "elevations": [
        ("fes2014_elevations_and_load", "fes2014b_elevations", "ocean_tide.tar.xz"),
    ],
    "extrapolated": [
        ("fes2014_elevations_and_load", "fes2014b_elevations_extrapolated", "ocean_tide_extrapolated.tar.xz"),
    ],
    "currents": [
        ("fes2014a_currents", "eastward_velocity.tar.xz"),
        ("fes2014a_currents", "northward_velocity.tar.xz"),
    ],
    "load": [
        ("fes2014_elevations_and_load", "fes2014a_loadtide", "load_tide.tar.xz"),
    ],
}


def main() -> None:
    args = parse_args()
    dest_root = args.dest_root
    dest_root.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or dest_root / "download_fes2014.log"
    manifest_path = args.manifest_path or dest_root / "download_fes2014_manifest.json"

    user = args.user or os.environ.get("AVISO_USERNAME") or os.environ.get("AVISO_USER")
    password = args.password or os.environ.get("AVISO_PASSWORD") or os.environ.get("AVISO_PASS")
    if not user or not password:
        message = (
            "Missing AVISO credentials. Set AVISO_USERNAME and AVISO_PASSWORD, "
            "or pass --user and --password. FES2014 is on the authenticated "
            "AVISO FTP server."
        )
        append_log(log_path, message)
        manifest_path.write_text(
            json.dumps({"status": "missing_credentials", "message": message, "dest_root": str(dest_root)}, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(2)

    requested = ["readme"]
    for component in args.components:
        if component not in requested:
            requested.append(component)
    remote_paths: list[tuple[str, ...]] = []
    for component in requested:
        remote_paths.extend(FES2014_FILES[component])

    manifest: dict[str, object] = {
        "status": "running",
        "host": HOST,
        "remote_prefix": REMOTE_PREFIX,
        "dest_root": str(dest_root),
        "components": requested,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    append_log(log_path, f"Connecting to {HOST} with authenticated AVISO credentials")
    with ftplib.FTP(HOST, timeout=args.timeout) as ftp:
        try:
            ftp.login(user, password)
        except ftplib.error_perm as exc:
            message = f"AVISO FTP login failed: {exc}"
            append_log(log_path, message)
            manifest["status"] = "login_failed"
            manifest["message"] = message
            manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise SystemExit(3) from exc
        for remote_parts in remote_paths:
            remote_file = posixpath.join(REMOTE_PREFIX, *remote_parts)
            local_file = dest_root / Path(*remote_parts)
            result = download_one(ftp, remote_file, local_file, log_path, args.chunk_size)
            manifest["files"].append(result)  # type: ignore[index]
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest["status"] = "complete"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    append_log(log_path, f"Complete. Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FES2014 archives from authenticated AVISO FTP.")
    parser.add_argument("--dest-root", type=Path, default=Path(r"F:\Tide\FES2014"))
    parser.add_argument("--components", nargs="+", choices=["elevations", "currents", "load", "extrapolated"], default=["elevations", "currents", "load", "extrapolated"])
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    return parser.parse_args()


def download_one(ftp: ftplib.FTP, remote_file: str, local_file: Path, log_path: Path, chunk_size: int) -> dict[str, object]:
    local_file.parent.mkdir(parents=True, exist_ok=True)
    part_file = local_file.with_suffix(local_file.suffix + ".part")
    remote_size = ftp.size(remote_file)
    existing_size = part_file.stat().st_size if part_file.exists() else local_file.stat().st_size if local_file.exists() else 0
    if remote_size is not None and local_file.exists() and local_file.stat().st_size == remote_size:
        append_log(log_path, f"skip complete: {local_file} ({remote_size} bytes)")
        return {"remote": remote_file, "local": str(local_file), "status": "skipped_complete", "bytes": remote_size}
    if local_file.exists() and not part_file.exists():
        local_file.replace(part_file)
    mode = "ab" if part_file.exists() else "wb"
    rest = part_file.stat().st_size if part_file.exists() else 0
    append_log(log_path, f"download: ftp://{HOST}/{remote_file} -> {local_file} resume={rest}")
    with part_file.open(mode + "") as handle:
        ftp.retrbinary(f"RETR {remote_file}", handle.write, blocksize=chunk_size, rest=rest if rest > 0 else None)
    final_size = part_file.stat().st_size
    if remote_size is not None and final_size != remote_size:
        append_log(log_path, f"incomplete: {part_file} local={final_size} remote={remote_size}")
        return {"remote": remote_file, "local": str(local_file), "status": "incomplete", "bytes": final_size, "remote_bytes": remote_size}
    part_file.replace(local_file)
    append_log(log_path, f"done: {local_file} ({final_size} bytes)")
    return {"remote": remote_file, "local": str(local_file), "status": "downloaded", "bytes": final_size}


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")
    print(message, flush=True)


if __name__ == "__main__":
    main()
