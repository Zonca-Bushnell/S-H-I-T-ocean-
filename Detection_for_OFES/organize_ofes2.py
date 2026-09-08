from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_ROOT = Path(r"F:\OFES\external_OFES2")
CTL_NAMES = {"eta", "pressur", "u", "v", "w", "temp", "salinity", "prho"}


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    kind: str
    variable: str
    year: int | None
    extension: str


@dataclass(frozen=True)
class MoveAction:
    source: str
    destination: str
    kind: str
    variable: str
    year: int | None
    size: int
    action: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize the local OFES2 data tree.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--reconstruct-applied-log", action="store_true")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    if args.reconstruct_applied_log:
        manifest_dir = root / "metadata" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        actions = reconstruct_applied_actions(root)
        write_actions(manifest_dir / "organize_applied_plan", actions)
        write_actions(manifest_dir / "organize_plan", actions)
        before_records = records_from_action_sources(root, actions)
        write_records(manifest_dir / "manifest_before_reconstructed", before_records)
        write_records(manifest_dir / "manifest_before", before_records)
        print(f"Reconstructed {len(actions)} applied move/copy records under {manifest_dir}")
        return

    before_records = scan_tree(root)
    manifest_dir = root / "metadata" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_records(manifest_dir / "manifest_before", before_records)

    actions = plan_actions(root)
    write_actions(manifest_dir / "organize_plan", actions)

    if args.dry_run:
        print(f"Dry-run complete. Planned {len([a for a in actions if a.action != 'noop'])} moves/copies.")
        print(f"Manifests written under {manifest_dir}")
        return

    apply_actions(actions)
    cleanup_legacy_empty_dirs(root)
    after_records = scan_tree(root)
    write_records(manifest_dir / "manifest_after", after_records)
    print(f"Applied {len([a for a in actions if a.action != 'noop'])} moves/copies.")
    print(f"Manifests written under {manifest_dir}")


def scan_tree(root: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        variable = infer_variable(path.name, rel)
        year = infer_year(path.name, rel)
        records.append(
            FileRecord(
                path=rel,
                size=path.stat().st_size,
                kind=infer_kind(path),
                variable=variable,
                year=year,
                extension=path.suffix.lower(),
            )
        )
    return records


def plan_actions(root: Path) -> list[MoveAction]:
    actions: list[MoveAction] = []

    for path in sorted(root.glob("*.ctl")):
        variable = path.stem
        if variable in CTL_NAMES:
            actions.append(make_action(path, root / "metadata" / "ctl" / path.name, "move", "ctl", variable, None))

    for path in sorted(root.glob("*.tgz")):
        variable = infer_variable(path.name, path.name)
        year = infer_year(path.name, path.name) or 1991
        actions.append(make_action(path, root / "compressed" / variable / str(year) / path.name, "move", "compressed", variable, year))

    ssh_note = root / "SSH说明.txt"
    if ssh_note.exists():
        actions.append(make_action(ssh_note, root / "docs" / ssh_note.name, "move", "doc", "ssh", None))

    read_data = root.parent / "read_data.m"
    if read_data.exists():
        actions.append(make_action(read_data, root / "docs" / "read_data.m", "copy", "doc", "matlab", None))

    legacy_daily = {
        "eta": root / "eta_y1991" / "daily",
        "pressur": root / "pressur_y1991" / "daily",
    }
    for variable, source_dir in legacy_daily.items():
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.dta")):
            year = infer_year(path.name, path.as_posix()) or 1991
            actions.append(make_action(path, root / "daily" / f"y{year}" / variable / path.name, "move", "daily", variable, year))

    return actions


def make_action(source: Path, destination: Path, action: str, kind: str, variable: str, year: int | None) -> MoveAction:
    if source.resolve() == destination.resolve():
        action = "noop"
    return MoveAction(
        source=str(source),
        destination=str(destination),
        kind=kind,
        variable=variable,
        year=year,
        size=source.stat().st_size,
        action=action,
    )


def apply_actions(actions: list[MoveAction]) -> None:
    for action in actions:
        if action.action == "noop":
            continue
        source = Path(action.source)
        destination = Path(action.destination)
        if not source.exists():
            raise FileNotFoundError(source)
        if destination.exists():
            if destination.stat().st_size == source.stat().st_size:
                print(f"Skip existing same-size destination: {destination}")
                continue
            raise FileExistsError(f"Destination exists with different size: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if action.action == "move":
            shutil.move(str(source), str(destination))
        elif action.action == "copy":
            shutil.copy2(source, destination)
        else:
            raise ValueError(f"Unknown action: {action.action}")


def reconstruct_applied_actions(root: Path) -> list[MoveAction]:
    actions: list[MoveAction] = []

    for path in sorted((root / "metadata" / "ctl").glob("*.ctl")):
        actions.append(make_reconstructed_action(root / path.name, path, "move", "ctl", path.stem, None))

    for path in sorted((root / "compressed").glob("*/*/*.tgz")):
        variable = path.parent.parent.name
        year = int(path.parent.name)
        actions.append(make_reconstructed_action(root / path.name, path, "move", "compressed", variable, year))

    for variable, legacy_dir in {"eta": "eta_y1991", "pressur": "pressur_y1991"}.items():
        for path in sorted((root / "daily" / "y1991" / variable).glob("*.dta")):
            actions.append(
                make_reconstructed_action(
                    root / legacy_dir / "daily" / path.name,
                    path,
                    "move",
                    "daily",
                    variable,
                    1991,
                )
            )

    ssh_note = root / "docs" / "SSH说明.txt"
    if ssh_note.exists():
        actions.append(make_reconstructed_action(root / "SSH说明.txt", ssh_note, "move", "doc", "ssh", None))

    read_data = root / "docs" / "read_data.m"
    if read_data.exists():
        actions.append(make_reconstructed_action(root.parent / "read_data.m", read_data, "copy", "doc", "matlab", None))

    return actions


def make_reconstructed_action(
    source: Path,
    destination: Path,
    action: str,
    kind: str,
    variable: str,
    year: int | None,
) -> MoveAction:
    return MoveAction(
        source=str(source),
        destination=str(destination),
        kind=kind,
        variable=variable,
        year=year,
        size=destination.stat().st_size,
        action=action,
    )


def records_from_action_sources(root: Path, actions: list[MoveAction]) -> list[FileRecord]:
    records: list[FileRecord] = []
    for action in actions:
        source = Path(action.source)
        try:
            rel = source.relative_to(root).as_posix()
        except ValueError:
            rel = source.as_posix()
        records.append(
            FileRecord(
                path=rel,
                size=action.size,
                kind=action.kind,
                variable=action.variable,
                year=action.year,
                extension=source.suffix.lower(),
            )
        )
    return records


def cleanup_legacy_empty_dirs(root: Path) -> None:
    for path in [
        root / "eta_y1991" / "daily",
        root / "eta_y1991",
        root / "pressur_y1991" / "daily",
        root / "pressur_y1991",
    ]:
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def write_records(prefix: Path, records: list[FileRecord]) -> None:
    write_json(prefix.with_suffix(".json"), [asdict(r) for r in records])
    write_csv(prefix.with_suffix(".csv"), [asdict(r) for r in records])


def write_actions(prefix: Path, actions: list[MoveAction]) -> None:
    write_json(prefix.with_suffix(".json"), [asdict(a) for a in actions])
    write_csv(prefix.with_suffix(".csv"), [asdict(a) for a in actions])


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def infer_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".ctl":
        return "ctl"
    if suffix == ".tgz":
        return "compressed"
    if suffix == ".dta":
        return "daily"
    if suffix in {".txt", ".m"}:
        return "doc"
    return "other"


def infer_variable(name: str, rel: str) -> str:
    lower_name = name.lower()
    lower_rel = rel.lower().replace("\\", "/")
    parts = [p for p in lower_rel.split("/") if p]

    stem = Path(lower_name).stem
    first_token = re.split(r"[._-]", stem)[0]
    direct = {
        "eta": "eta",
        "pressur": "pressur",
        "pressure": "pressur",
        "prho": "prho",
        "sali": "salinity",
        "salinity": "salinity",
        "temp": "temp",
        "u": "u",
        "v": "v",
        "w": "w",
    }
    if first_token in direct:
        return direct[first_token]

    for part in reversed(parts):
        if part in {"eta", "pressur", "prho", "salinity", "temp", "u", "v", "w"}:
            return part
        if part in {"u_vel", "v_vel", "w_vel"}:
            return part[0]

    archive_match = re.match(r"^(eta|pressur|prho|sali|salinity|temp|u|v|w)_y\d{4}", lower_name)
    if archive_match:
        return direct[archive_match.group(1)]
    return "unknown"


def infer_year(name: str, rel: str) -> int | None:
    match = re.search(r"(?:y|\.)(19\d{2}|20\d{2})(?:[_.\\/]|$)", f"{rel}/{name}")
    if match:
        return int(match.group(1))
    return None


if __name__ == "__main__":
    main()
