import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath

from acc_config import (
    DATASET_ID,
    DATASET_VERSION,
    MANIFEST_DIR,
    PASSWORD_KEYS,
    RAW_DIR,
    SELECTED_FILE_LIST,
    USERNAME_KEYS,
    VARIABLES,
    build_time_windows,
    selected_bounds,
)


DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})?(\d{2})?(?!\d)")


def _first_non_empty_env(keys):
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _read_registry_env(var_name):
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            value, _ = winreg.QueryValueEx(key, var_name)
            return value or None
    except Exception:
        return None


def resolve_credentials():
    username = _first_non_empty_env(USERNAME_KEYS)
    password = _first_non_empty_env(PASSWORD_KEYS)

    if not username:
        for key in USERNAME_KEYS:
            username = _read_registry_env(key)
            if username:
                break
    if not password:
        for key in PASSWORD_KEYS:
            password = _read_registry_env(key)
            if password:
                break

    return username, password


def copernicusmarine_executable():
    executable = shutil.which("copernicusmarine")
    if executable:
        return executable

    scripts_dir = Path(os.sys.executable).resolve().parent / "Scripts"
    candidate = scripts_dir / "copernicusmarine.exe"
    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        "copernicusmarine CLI not found. Run this script after "
        "`mamba activate copernicus_downloading`."
    )


def month_starts(start, end):
    cursor = datetime(start.year, start.month, 1)
    while cursor < end:
        yield cursor
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)


def read_list(path):
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            lines.append(value)
    return lines


def write_list(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def dedupe_preserving_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def date_overlaps(path, start, end):
    filename_matches = list(DATE_RE.finditer(PurePosixPath(path).name))
    matches = filename_matches or list(DATE_RE.finditer(path))
    if not matches:
        return True

    for match in matches:
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        day = int(match.group(3) or 1)
        try:
            value = datetime(year, month, day)
        except ValueError:
            continue

        if match.group(3):
            if start <= value < end:
                return True
        elif match.group(2):
            next_month = datetime(year + (month == 12), 1 if month == 12 else month + 1, 1)
            if value < end and next_month > start:
                return True
        else:
            next_year = datetime(year + 1, 1, 1)
            if value < end and next_year > start:
                return True

    return False


def filter_selected_paths(paths, start, end):
    time_filtered = [path for path in paths if date_overlaps(path, start, end)]
    has_variable_split = any(any(var in path for var in VARIABLES) for path in time_filtered)
    if not has_variable_split:
        return time_filtered
    return [path for path in time_filtered if any(var in path for var in VARIABLES)]


def run_command(command):
    printable_parts = []
    redact_next = False
    for part in command:
        if redact_next:
            printable_parts.append("<redacted>")
            redact_next = False
            continue
        printable_parts.append(f'"{part}"' if " " in part else part)
        if part == "--password":
            redact_next = True
    printable = " ".join(printable_parts)
    print(f"Running: {printable}")
    result = subprocess.run(command)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {printable}")


def add_credentials(command):
    username, password = resolve_credentials()
    if username and password:
        command.extend(["--username", username, "--password", password])
    return command


def create_candidate_lists(args):
    exe = copernicusmarine_executable()
    start, end = selected_bounds(args.sample)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for month in month_starts(start, end):
        list_name = f"candidates_{month:%Y%m}.txt"
        command = [
            exe,
            "get",
            "-i",
            DATASET_ID,
            "--dataset-version",
            DATASET_VERSION,
            "--filter",
            f"*{month:%Y%m}*.nc",
            "--create-file-list",
            list_name,
            "-o",
            str(args.manifest_dir),
            "--disable-progress-bar",
        ]
        if args.log_level:
            command.extend(["--log-level", args.log_level])
        run_command(add_credentials(command))
        candidates.extend(read_list(args.manifest_dir / list_name))

    selected = dedupe_preserving_order(filter_selected_paths(candidates, start, end))
    write_list(args.file_list, selected)

    print(f"Candidate files: {len(candidates)}")
    print(f"Selected files: {len(selected)}")
    print(f"Selected file list: {args.file_list}")
    if selected:
        print("First selected files:")
        for path in selected[: min(5, len(selected))]:
            print(f"  {path}")
    else:
        raise RuntimeError("No remote files matched the configured time range and variables.")


def download_selected_files(args):
    if not args.file_list.exists():
        raise FileNotFoundError(f"File list does not exist: {args.file_list}")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    exe = copernicusmarine_executable()
    command = [
        exe,
        "get",
        "-i",
        DATASET_ID,
        "--dataset-version",
        DATASET_VERSION,
        "--file-list",
        str(args.file_list),
        "--skip-existing",
        "-o",
        str(args.raw_dir),
        "--disable-progress-bar",
    ]
    if args.max_concurrent_requests is not None:
        command.extend(["--max-concurrent-requests", str(args.max_concurrent_requests)])
    if args.log_level:
        command.extend(["--log-level", args.log_level])

    run_command(add_credentials(command))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Copernicus Marine file list and optionally download raw ACC files."
    )
    parser.add_argument("--sample", action="store_true", help="Use SAMPLE_START/SAMPLE_END.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files after creating or reusing the selected file list.",
    )
    parser.add_argument(
        "--reuse-file-list",
        action="store_true",
        help="Skip remote listing and use the existing selected file list.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--file-list", type=Path, default=SELECTED_FILE_LIST)
    parser.add_argument("--max-concurrent-requests", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.reuse_file_list:
        create_candidate_lists(args)
    if args.download:
        download_selected_files(args)
    else:
        print("Download not started. Add --download to fetch the selected files.")


if __name__ == "__main__":
    main()
