from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
TEX_PATH = ROOT / "curved_tube_ep_flux.tex"
PDF_PATH = ROOT / "curved_tube_ep_flux.pdf"
BUILD_DIR = ROOT / "build" / "curved_tube_ep_flux"
BUILD_TEX = BUILD_DIR / "curved_tube_ep_flux.tex"
BUILD_PDF = BUILD_DIR / "curved_tube_ep_flux.pdf"

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]
FONT_PATH = next(path for path in FONT_PATHS if path.exists())


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def arrow(draw: ImageDraw.ImageDraw, start, end, color="#334155", width=4, head=14) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    p1 = (end[0] - head * math.cos(angle - math.pi / 6), end[1] - head * math.sin(angle - math.pi / 6))
    p2 = (end[0] - head * math.cos(angle + math.pi / 6), end[1] - head * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=color)


def centered(draw: ImageDraw.ImageDraw, rect, text: str, text_font, fill="#0f172a") -> None:
    lines = text.split("\n")
    line_h = int(text_font.size * 1.28)
    total_h = line_h * len(lines)
    y = rect[1] + (rect[3] - rect[1] - total_h) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=text_font)
        x = rect[0] + (rect[2] - rect[0] - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_h


def box(draw: ImageDraw.ImageDraw, rect, text: str, text_font, fill="#f8fafc", outline="#334155") -> None:
    draw.rounded_rectangle(rect, radius=12, fill=fill, outline=outline, width=3)
    centered(draw, rect, text, text_font)


def save(img: PILImage.Image, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img.save(FIG_DIR / name)


def fig_ctep_tube_frame() -> None:
    img = PILImage.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(img)
    title = font(48)
    label = font(40)
    small = font(34)
    draw.text((215, 35), "Curved material vortex tube and Bishop frame", font=title, fill="#0f172a")
    pts = []
    for i in range(13):
        x = 120 + i * 98
        y = 360 - 130 * math.sin(i / 12 * math.pi) + 34 * math.sin(i / 12 * 3 * math.pi)
        pts.append((x, y))
    draw.line(pts, fill="#2563eb", width=9, joint="curve")
    for i, p in enumerate(pts[1:-1:2], start=1):
        rx = 58 + 10 * math.sin(i)
        ry = 92 + 8 * math.cos(i)
        draw.ellipse((p[0] - rx, p[1] - ry, p[0] + rx, p[1] + ry), outline="#94a3b8", width=4)
        draw.line((p[0] - rx, p[1], p[0] + rx, p[1]), fill="#cbd5e1", width=2)
        draw.line((p[0], p[1] - ry, p[0], p[1] + ry), fill="#cbd5e1", width=2)
    p = pts[6]
    arrow(draw, p, (p[0] + 175, p[1] - 48), "#0f172a", 6)
    arrow(draw, p, (p[0] + 28, p[1] - 170), "#16a34a", 6)
    arrow(draw, p, (p[0] - 155, p[1] - 58), "#ea580c", 6)
    draw.text((p[0] + 182, p[1] - 68), "t", font=label, fill="#0f172a")
    draw.text((p[0] + 36, p[1] - 205), "e1", font=label, fill="#16a34a")
    draw.text((p[0] - 200, p[1] - 84), "e2", font=label, fill="#ea580c")
    draw.text((p[0] - 45, p[1] + 38), "r_c(s,t)", font=small, fill="#2563eb")
    draw.text((230, 545), "r = r_c + xi e1 + eta e2,    J = 1 - kappa_alpha x^alpha", font=small, fill="#334155")
    save(img, "ctep_tube_frame.png")


def fig_ctep_pv_centroid() -> None:
    img = PILImage.new("RGB", (1200, 660), "white")
    draw = ImageDraw.Draw(img)
    title = font(46)
    label = font(34)
    small = font(30)
    draw.text((190, 35), "PV-centroid axis on an arbitrary section", font=title, fill="#0f172a")
    cx, cy = 570, 320
    boundary = []
    for i in range(180):
        th = 2 * math.pi * i / 180
        r = 260 + 42 * math.sin(3 * th + 0.4) + 30 * math.cos(5 * th)
        boundary.append((cx + r * math.cos(th), cy + 0.72 * r * math.sin(th)))
    draw.polygon(boundary, fill="#eff6ff", outline="#2563eb")
    draw.line(boundary + [boundary[0]], fill="#2563eb", width=5)
    for ring, color in [(180, "#bfdbfe"), (112, "#60a5fa"), (52, "#1d4ed8")]:
        draw.ellipse((cx - ring, cy - 0.72 * ring, cx + ring, cy + 0.72 * ring), outline=color, width=5)
    draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill="#dc2626")
    draw.text((cx + 32, cy - 15), "r_c: q-weighted centroid", font=label, fill="#dc2626")
    draw.text((130, 570), "int_Omega x_perp q_c J dA = 0", font=small, fill="#334155")
    draw.text((130, 610), "Gamma(s,t): material PV contour, arbitrary closed section", font=small, fill="#334155")
    save(img, "ctep_pv_centroid.png")


def fig_ctep_tensor_upgrade() -> None:
    img = PILImage.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(img)
    title = font(44)
    label = font(34)
    small = font(30)
    draw.text((225, 35), "From zonal-mean E-P vector to curved-tube E-P tensor", font=title, fill="#0f172a")
    box(draw, (70, 190, 420, 365), "classical E-P\nF(y,z)", label, "#eef6ff", "#2563eb")
    box(draw, (525, 170, 875, 385), "material\nsection average", label, "#f8fafc", "#64748b")
    box(draw, (980, 150, 1330, 405), "CT-EP tensor\nF_CT^{a i}\ncovariant div", label, "#fef2f2", "#b91c1c")
    arrow(draw, (420, 278), (525, 278), "#334155", 6)
    arrow(draw, (875, 278), (980, 278), "#334155", 6)
    draw.text((90, 465), "F_y: momentum flux", font=small, fill="#334155")
    draw.text((90, 505), "F_z: buoyancy flux mapped by balance", font=small, fill="#334155")
    draw.text((990, 465), "a: flux direction", font=small, fill="#334155")
    draw.text((990, 505), "i: forced axis component", font=small, fill="#334155")
    draw.text((305, 575), "div_CT = J^{-1} partial_a(J F_CT^{a i}) + Gamma^i_{ka} F_CT^{a k}", font=small, fill="#0f172a")
    save(img, "ctep_tensor_upgrade.png")


def fig_ctep_axis_evolution() -> None:
    img = PILImage.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(img)
    title = font(46)
    label = font(36)
    small = font(30)
    draw.text((285, 35), "Flux divergence controls axial shear and tilt", font=title, fill="#0f172a")
    pts0 = [(130 + i * 130, 500 - i * 16) for i in range(8)]
    pts1 = [(130 + i * 130, 500 - i * 16 - 26 * i - 8 * i * i) for i in range(8)]
    draw.line(pts0, fill="#94a3b8", width=6)
    draw.line(pts1, fill="#2563eb", width=7)
    for p0, p1 in zip(pts0, pts1):
        draw.ellipse((p0[0] - 10, p0[1] - 10, p0[0] + 10, p0[1] + 10), fill="#94a3b8")
        draw.ellipse((p1[0] - 12, p1[1] - 12, p1[0] + 12, p1[1] + 12), fill="#2563eb")
        arrow(draw, (p0[0], p0[1] - 18), (p1[0], p1[1] + 18), "#ea580c", 3, 10)
    box(draw, (1010, 135, 1330, 305), "div F_CT\n+ PV flux P", label, "#fef2f2", "#b91c1c")
    arrow(draw, (1010, 305), (870, 380), "#b91c1c", 6)
    draw.text((145, 555), "initial centerline", font=small, fill="#64748b")
    draw.text((705, 135), "D_t t = P_perp partial_s V_c", font=label, fill="#0f172a")
    draw.text((430, 585), "perpendicular axial shear bends the material vortex axis", font=small, fill="#334155")
    save(img, "ctep_axis_evolution.png")


def generate_figures() -> None:
    fig_ctep_tube_frame()
    fig_ctep_pv_centroid()
    fig_ctep_tensor_upgrade()
    fig_ctep_axis_evolution()


def clean_aux() -> None:
    for suffix in [".aux", ".log", ".out", ".toc"]:
        artifact = BUILD_TEX.with_suffix(suffix)
        if artifact.exists():
            artifact.unlink()
    if BUILD_TEX.exists():
        BUILD_TEX.unlink()


def compile_latex() -> Path:
    xelatex = shutil.which("xelatex")
    if not xelatex:
        raise RuntimeError("xelatex was not found on PATH")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tex = TEX_PATH.read_text(encoding="utf-8")
    fig_path = FIG_DIR.as_posix() + "/"
    tex = tex.replace(r"\graphicspath{{assets/figures/}}", rf"\graphicspath{{{{{fig_path}}}}}")
    BUILD_TEX.write_text(tex, encoding="utf-8")
    for _ in range(2):
        subprocess.run([xelatex, "-interaction=nonstopmode", "-halt-on-error", BUILD_TEX.name], cwd=BUILD_DIR, check=True)
    if not BUILD_PDF.exists():
        raise RuntimeError(f"expected PDF not found: {BUILD_PDF}")
    shutil.copy2(BUILD_PDF, PDF_PATH)
    clean_aux()
    return PDF_PATH


def main() -> None:
    generate_figures()
    output = compile_latex()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
