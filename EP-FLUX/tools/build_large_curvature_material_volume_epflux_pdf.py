from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
TEX_PATH = ROOT / "large_curvature_material_volume_ep_flux.tex"
PDF_PATH = ROOT / "large_curvature_material_volume_ep_flux.pdf"
BUILD_DIR = ROOT / "build" / "large_curvature_material_volume_ep_flux"
BUILD_TEX = BUILD_DIR / "large_curvature_material_volume_ep_flux.tex"
BUILD_PDF = BUILD_DIR / "large_curvature_material_volume_ep_flux.pdf"


def font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def arrow(draw: ImageDraw.ImageDraw, start, end, color="#334155", width=4, head=14) -> None:
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ang = math.atan2(dy, dx)
    pts = [
        end,
        (end[0] - head * math.cos(ang - 0.45), end[1] - head * math.sin(ang - 0.45)),
        (end[0] - head * math.cos(ang + 0.45), end[1] - head * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=color)


def centered(draw: ImageDraw.ImageDraw, rect, text: str, text_font, fill="#0f172a") -> None:
    lines = text.splitlines()
    bboxes = [draw.textbbox((0, 0), line, font=text_font) for line in lines]
    heights = [b[3] - b[1] for b in bboxes]
    total_h = sum(heights) + 8 * (len(lines) - 1)
    y = rect[1] + (rect[3] - rect[1] - total_h) / 2
    for line, bbox, h in zip(lines, bboxes, heights):
        w = bbox[2] - bbox[0]
        x = rect[0] + (rect[2] - rect[0] - w) / 2
        draw.text((x, y), line, font=text_font, fill=fill)
        y += h + 8


def box(draw: ImageDraw.ImageDraw, rect, text: str, text_font, fill="#f8fafc", outline="#334155") -> None:
    draw.rounded_rectangle(rect, radius=8, fill=fill, outline=outline, width=3)
    centered(draw, rect, text, text_font)


def save(img: Image.Image, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img.save(FIG_DIR / name, dpi=(170, 170))


def fig_curvature_failure() -> None:
    img = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(img)
    title = font(36)
    label = font(26)
    small = font(22)
    draw.text((70, 42), "Why first-order tube coordinates fail at large curvature", font=title, fill="#0f172a")

    draw.arc((120, 170, 580, 630), start=235, end=55, fill="#2563eb", width=8)
    for a in [250, 285, 320, 355, 30]:
        x = 350 + 230 * math.cos(math.radians(a))
        y = 400 + 230 * math.sin(math.radians(a))
        nx = math.cos(math.radians(a))
        ny = math.sin(math.radians(a))
        draw.line((x - 38 * ny, y + 38 * nx, x + 38 * ny, y - 38 * nx), fill="#60a5fa", width=4)
    draw.text((175, 690), "thin tube: kappa*r << 1", font=label, fill="#1d4ed8")
    draw.text((160, 725), "single-valued local map, J > 0", font=small, fill="#334155")

    draw.arc((890, 135, 1210, 455), start=205, end=20, fill="#b91c1c", width=8)
    draw.arc((990, 245, 1310, 565), start=205, end=20, fill="#b91c1c", width=8)
    for x, y in [(1035, 300), (1110, 350), (1175, 410), (1240, 480)]:
        draw.ellipse((x - 95, y - 40, x + 95, y + 40), outline="#ef4444", width=4)
        draw.line((x - 95, y, x + 95, y), fill="#fecaca", width=3)
    draw.text((925, 690), "observed regime: kappa*r = 10-35", font=label, fill="#991b1b")
    draw.text((905, 725), "folded map, J can change sign", font=small, fill="#334155")

    arrow(draw, (650, 400), (830, 400), "#64748b", width=5, head=18)
    draw.text((645, 330), "not a higher-order", font=small, fill="#334155")
    draw.text((650, 360), "correction problem", font=small, fill="#334155")
    save(img, "mv_ep_curvature_failure.png")


def fig_object_shift() -> None:
    img = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(img)
    title = font(36)
    label = font(25)
    small = font(21)
    draw.text((70, 42), "Theory object shift: from thin tube to coherent material volume", font=title, fill="#0f172a")

    box(draw, (70, 170, 485, 365), "thin curved tube\n(s, xi, eta)\nJ = 1 - kappa*x", label, "#eff6ff", "#2563eb")
    box(draw, (1015, 145, 1435, 390), "coherent material\nvolume V_c(t)\nCartesian x,y,z", label, "#f0fdf4", "#15803d")
    arrow(draw, (520, 270), (975, 270), "#475569", width=5, head=18)
    draw.text((610, 215), "replace coordinate foundation", font=small, fill="#334155")
    draw.text((650, 250), "not the EP idea", font=small, fill="#334155")

    pts = [(1030, 530), (1115, 460), (1250, 475), (1390, 545), (1335, 670), (1160, 690), (1045, 630)]
    draw.polygon(pts, fill="#dcfce7", outline="#16a34a")
    draw.line((1110, 610, 1180, 560, 1260, 585, 1340, 535), fill="#166534", width=5)
    for p in [(1110, 610), (1180, 560), (1260, 585), (1340, 535)]:
        draw.ellipse((p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8), fill="#166534")
    draw.text((1085, 720), "axis skeleton = diagnostic only", font=small, fill="#166534")

    draw.ellipse((170, 520, 390, 640), outline="#3b82f6", width=5)
    draw.line((270, 580, 380, 500), fill="#1d4ed8", width=5)
    draw.text((130, 710), "axis as coordinate backbone", font=small, fill="#1d4ed8")
    save(img, "mv_ep_object_shift.png")


def fig_tensor_forcing() -> None:
    img = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(img)
    title = font(36)
    label = font(26)
    small = font(22)
    draw.text((70, 42), "Cartesian material-volume EP forcing", font=title, fill="#0f172a")

    box(draw, (80, 185, 390, 330), "Reynolds stress\nR_ij = <u_i' u_j'>", label, "#f8fafc", "#475569")
    box(draw, (80, 430, 390, 575), "buoyancy flux\nB_i = <u_i' b'>", label, "#fff7ed", "#ea580c")
    box(draw, (80, 620, 390, 755), "PV flux\nP_i = <u_i' q'>", label, "#f0f9ff", "#0284c7")

    box(draw, (580, 280, 930, 535), "G_i = -d_j(rho0 R_ij)\n+ d_j(rho0 T_ij[B_j])", label, "#fef2f2", "#b91c1c")
    box(draw, (1100, 300, 1430, 515), "PV inversion\nmean velocity\ncentroid dynamics", label, "#f0fdf4", "#15803d")
    arrow(draw, (410, 255), (560, 360), "#475569", width=4)
    arrow(draw, (410, 500), (560, 440), "#ea580c", width=4)
    draw.line([(410, 690), (760, 690), (1085, 535)], fill="#0284c7", width=4)
    arrow(draw, (1085, 535), (1120, 535), "#0284c7", width=4, head=14)
    arrow(draw, (950, 410), (1080, 410), "#475569", width=5, head=18)

    save(img, "mv_ep_tensor_forcing.png")


def fig_axis_skeleton() -> None:
    img = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(img)
    title = font(36)
    label = font(26)
    small = font(22)
    draw.text((70, 42), "Axis skeleton is kinematic, not a coordinate map", font=title, fill="#0f172a")

    sections = [(350, 610, 210, 78), (490, 535, 235, 86), (650, 470, 250, 96), (830, 400, 265, 105), (1030, 340, 285, 115)]
    centers = []
    for cx, cy, w, h in sections:
        draw.ellipse((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), fill="#e0f2fe", outline="#0284c7", width=4)
        centers.append((cx, cy))
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill="#0f172a")
    draw.line(centers, fill="#0f172a", width=5)
    for start, end in zip(centers[:-1], centers[1:]):
        arrow(draw, start, end, "#0f172a", width=3, head=10)

    arrow(draw, (650, 470), (650, 325), "#dc2626", width=5, head=18)
    arrow(draw, (650, 470), (770, 435), "#16a34a", width=5, head=18)
    draw.text((675, 318), "P_perp d_s V_c", font=label, fill="#dc2626")
    draw.text((785, 420), "tangent part", font=small, fill="#166534")
    draw.text((790, 448), "reparameterizes", font=small, fill="#166534")
    draw.text((365, 690), "layer/shell PV centroids define the skeleton", font=small, fill="#334155")
    draw.text((375, 725), "EP/PV forcing changes V_c, then tilt/bend follows", font=small, fill="#334155")
    save(img, "mv_ep_axis_skeleton.png")


def generate_figures() -> None:
    fig_curvature_failure()
    fig_object_shift()
    fig_tensor_forcing()
    fig_axis_skeleton()


def clean_aux() -> None:
    if BUILD_DIR.exists():
        for path in BUILD_DIR.glob("*"):
            if path.suffix.lower() in {".aux", ".log", ".out", ".toc", ".xdv"}:
                path.unlink(missing_ok=True)


def compile_latex() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEX_PATH, BUILD_TEX)
    cmd = [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(BUILD_DIR),
        str(BUILD_TEX),
    ]
    for _ in range(2):
        subprocess.run(cmd, cwd=ROOT, check=True)
    shutil.copy2(BUILD_PDF, PDF_PATH)
    clean_aux()
    return PDF_PATH


def main() -> None:
    generate_figures()
    pdf = compile_latex()
    print(f"Generated {pdf}")


if __name__ == "__main__":
    main()
