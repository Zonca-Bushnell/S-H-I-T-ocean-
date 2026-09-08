from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .ofes_io import ctl_path, data_path, parse_ctl, read_ssh, read_variable, summarize_array


DEFAULT_ROOT = Path(r"F:\OFES\external_OFES2")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "ofes2_sample_validation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one OFES2 eta/pressur/SSH sample.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--date", default="1991-01-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    eta_meta = parse_ctl(ctl_path(args.root, "eta"))
    pressur_meta = parse_ctl(ctl_path(args.root, "pressur"))

    eta_path = data_path(args.root, "eta", args.date)
    pressur_path = data_path(args.root, "pressur", args.date)
    assert_expected_size(eta_path, eta_meta.shape_xy)
    assert_expected_size(pressur_path, pressur_meta.shape_xy)

    eta = read_variable(args.root, "eta", args.date)
    pressur = read_variable(args.root, "pressur", args.date)
    ssh = read_ssh(args.root, args.date)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "root": str(args.root),
        "date": args.date,
        "eta_ctl": str(eta_meta.path),
        "pressur_ctl": str(pressur_meta.path),
        "eta_file": str(eta_path),
        "pressur_file": str(pressur_path),
        "eta_grid": eta_meta.grid_signature,
        "pressur_grid": pressur_meta.grid_signature,
        "ssh_formula": "SSH = eta - (pressur - 1000), units cm",
        "arrays": [
            summarize_array("eta", eta),
            summarize_array("pressur", pressur),
            summarize_array("ssh", ssh),
        ],
    }
    summary_path = output / f"summary_{args.date}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_preview(eta_meta.x.values, eta_meta.y.values, ssh, output / f"ssh_preview_{args.date}.png", args.date)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Preview written to {output / f'ssh_preview_{args.date}.png'}")


def assert_expected_size(path: Path, shape_xy: tuple[int, int]) -> None:
    expected = shape_xy[0] * shape_xy[1] * 4
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path} has {actual} bytes; expected {expected}.")


def plot_preview(lon: np.ndarray, lat: np.ndarray, ssh: np.ndarray, path: Path, date_label: str) -> None:
    stride = 4
    sampled = ssh[::stride, ::stride].T
    sampled = np.flipud(sampled)
    finite_sampled = np.nan_to_num(sampled, nan=0.0, posinf=150.0, neginf=-150.0)
    scaled = np.clip((finite_sampled + 150.0) / 300.0, 0.0, 1.0)
    rgb = _blue_white_red(scaled)
    rgb[~np.isfinite(sampled)] = np.array([190, 190, 190], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    image.save(path)

    sidecar = path.with_suffix(".txt")
    sidecar.write_text(
        "\n".join(
            [
                f"title: OFES2 SSH {date_label} (cm)",
                f"lon_min: {float(lon.min())}",
                f"lon_max: {float(lon.max())}",
                f"lat_min: {float(lat.min())}",
                f"lat_max: {float(lat.max())}",
                "color_scale_cm: -150 to 150",
                "nan_color: 190,190,190",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _blue_white_red(scaled: np.ndarray) -> np.ndarray:
    scaled = np.asarray(scaled, dtype=np.float32)
    rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
    lower = scaled <= 0.5
    upper = ~lower

    low_t = np.zeros_like(scaled)
    low_t[lower] = scaled[lower] / 0.5
    high_t = np.zeros_like(scaled)
    high_t[upper] = (scaled[upper] - 0.5) / 0.5

    blue = np.array([49, 130, 189], dtype=np.float32)
    white = np.array([247, 247, 247], dtype=np.float32)
    red = np.array([203, 24, 29], dtype=np.float32)

    rgb[lower] = (blue * (1.0 - low_t[lower][:, None]) + white * low_t[lower][:, None]).astype(np.uint8)
    rgb[upper] = (white * (1.0 - high_t[upper][:, None]) + red * high_t[upper][:, None]).astype(np.uint8)
    return rgb


if __name__ == "__main__":
    main()
