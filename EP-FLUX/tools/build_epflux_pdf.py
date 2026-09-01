from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
TEX_PATH = ROOT / "E-P_flux\u7406\u8bba\u6574\u7406.tex"
PDF_PATH = ROOT / "E-P_flux\u7406\u8bba\u6574\u7406.pdf"
BUILD_DIR = ROOT / "build" / "latex"
BUILD_TEX = BUILD_DIR / "epflux.tex"
BUILD_PDF = BUILD_DIR / "epflux.pdf"

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]
FONT_PATH = next(path for path in FONT_PATHS if path.exists())


def pil_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def draw_centered(draw: ImageDraw.ImageDraw, rect, text: str, text_font, fill="#0f172a", spacing=6) -> None:
    lines = text.split("\n")
    widths = [draw.textbbox((0, 0), line, font=text_font)[2] for line in lines]
    line_height = int(text_font.size * 1.28)
    total_h = line_height * len(lines) + spacing * (len(lines) - 1)
    y = rect[1] + (rect[3] - rect[1] - total_h) / 2
    for line, width in zip(lines, widths):
        x = rect[0] + (rect[2] - rect[0] - width) / 2
        draw.text((x, y), line, font=text_font, fill=fill)
        y += line_height + spacing


def box(draw: ImageDraw.ImageDraw, rect, text: str, text_font, fill="#f8fafc", outline="#334155") -> None:
    draw.rounded_rectangle(rect, radius=10, fill=fill, outline=outline, width=3)
    draw_centered(draw, rect, text, text_font)


def arrow(draw: ImageDraw.ImageDraw, start, end, color="#334155", width=4) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    p1 = (end[0] - size * math.cos(angle - math.pi / 6), end[1] - size * math.sin(angle - math.pi / 6))
    p2 = (end[0] - size * math.cos(angle + math.pi / 6), end[1] - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=color)


def save_img(img: PILImage.Image, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img.save(FIG_DIR / name)


def fig1_tem_vs_eulerian() -> None:
    img = PILImage.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(img)
    title = pil_font(38)
    f = pil_font(36)
    small = pil_font(28)
    draw.text((380, 55), "Eulerian mean vs. TEM residual mean", font=title, fill="#0f172a")
    box(draw, (90, 190, 430, 350), "Eulerian\nmean equations", f)
    box(draw, (590, 100, 1040, 240), "Momentum flux\n-d_y overline(u'v')", small, "#eef6ff")
    box(draw, (590, 330, 1040, 470), "Heat flux\n-d_y overline(v'T')", small, "#fff7ed")
    box(draw, (1120, 210, 1430, 350), "TEM residual\n(v*, w*)", f, "#f0fdf4")
    box(draw, (1120, 500, 1430, 640), "net forcing\ndiv(F)", f, "#fef2f2")
    arrow(draw, (430, 260), (590, 170))
    arrow(draw, (430, 285), (590, 400))
    arrow(draw, (1040, 170), (1120, 270), "#2563eb")
    arrow(draw, (1040, 400), (1120, 280), "#ea580c")
    arrow(draw, (1275, 350), (1275, 500), "#16a34a")
    draw.text(
        (145, 690),
        "TEM absorbs heat-flux compensation into residual circulation; div(F) remains as wave forcing.",
        font=small,
        fill="#334155",
    )
    save_img(img, "fig1_tem_vs_eulerian.png")


def fig2_ep_flux_vector() -> None:
    img = PILImage.new("RGB", (1200, 840), "white")
    draw = ImageDraw.Draw(img)
    f = pil_font(32)
    small = pil_font(26)
    left, top, right, bottom = 150, 120, 1040, 690
    for i in range(7):
        x = left + i * (right - left) / 6
        draw.line((x, top, x, bottom), fill="#e2e8f0", width=2)
    for i in range(5):
        y = top + i * (bottom - top) / 4
        draw.line((left, y, right, y), fill="#e2e8f0", width=2)
    arrow(draw, (left, bottom), (right + 40, bottom), "#334155")
    arrow(draw, (left, bottom), (left, top - 50), "#334155")
    draw.text((right + 50, bottom - 20), "y", font=small, fill="#0f172a")
    draw.text((left - 35, top - 90), "z", font=small, fill="#0f172a")
    origin = (380, 560)
    end = (760, 250)
    arrow(draw, origin, end, "#2563eb", 8)
    arrow(draw, origin, (760, 560), "#64748b", 5)
    arrow(draw, origin, (380, 250), "#64748b", 5)
    draw.text((780, 235), "F", font=pil_font(46), fill="#2563eb")
    draw.text((495, 590), "F_y = -rho0 overline(u'v')", font=small, fill="#334155")
    draw.text((395, 365), "F_z ~ overline(v'T')", font=small, fill="#334155")
    box(draw, (790, 350, 1070, 470), "div(F)\nmean-flow force", small, "#fef2f2", "#b91c1c")
    draw.text((385, 45), "E-P flux in the meridional plane", font=f, fill="#0f172a")
    save_img(img, "fig2_ep_flux_vector.png")


def fig3_j1_j2_comparison() -> None:
    img = PILImage.new("RGB", (1500, 680), "white")
    draw = ImageDraw.Draw(img)
    f = pil_font(34)
    small = pil_font(29)
    draw.text((230, 65), "J1 and J2 are complementary: meridional flux vs. explicit vertical flux.", font=small, fill="#334155")
    box(draw, (140, 190, 560, 340), "J1:\nmeridional flux", f, "#eef6ff")
    draw_centered(draw, (145, 355, 555, 470), "- overline(u'v')\n+ overline(theta'v')", small)
    box(draw, (940, 190, 1360, 340), "J2:\nvertical flux", f, "#fff7ed")
    draw_centered(draw, (945, 355, 1355, 470), "- overline(u'w')\n+ overline(theta'w')", small)
    box(draw, (535, 500, 965, 630), "mean PV and\nbasic-flow change", f, "#f0fdf4")
    arrow(draw, (350, 470), (600, 530), "#2563eb")
    arrow(draw, (1150, 470), (900, 530), "#ea580c")
    save_img(img, "fig3_j1_j2_comparison.png")


def heat_color(value: float):
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        return (255, int(245 - 90 * value), int(245 - 160 * value))
    return (int(245 + 10 * value), int(248 + 40 * value), 255)


def fig4_qbar_feedback() -> None:
    img = PILImage.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(img)
    f = pil_font(31)
    small = pil_font(25)
    left, top, w, h = 145, 120, 980, 500
    for ix in range(w):
        ycoord = -3 + 6 * ix / max(1, w - 1)
        for iz in range(h):
            zcoord = 4 - 4 * iz / max(1, h - 1)
            val = 0.55 * math.sin(ycoord) * math.cos(1.4 * zcoord)
            val -= 0.45 * math.cos(0.9 * ycoord) * math.sin(1.8 * zcoord)
            draw.point((left + ix, top + iz), fill=heat_color(val))
    for i in range(7):
        x = left + i * w / 6
        draw.line((x, top, x, top + h), fill="#ffffff", width=1)
    for i in range(5):
        y = top + i * h / 4
        draw.line((left, y, left + w, y), fill="#ffffff", width=1)
    draw.rectangle((left, top, left + w, top + h), outline="#334155", width=3)
    draw.text((265, 45), "partial_t qbar = -partial_y div(J1) - partial_z div(J2)", font=f, fill="#0f172a")
    draw.text((left + 20, top + 25), "meridional change\n-partial_y div(J1)", font=small, fill="#0f172a", stroke_width=3, stroke_fill="white")
    draw.text((left + 620, top + 380), "vertical change\n-partial_z div(J2)", font=small, fill="#0f172a", stroke_width=3, stroke_fill="white")
    draw.text((left + w + 40, top + 210), "mean PV\ntendency", font=small, fill="#334155")
    save_img(img, "fig4_qbar_feedback.png")


def fig5_coherent_tilt() -> None:
    img = PILImage.new("RGB", (1200, 840), "white")
    draw = ImageDraw.Draw(img)
    f = pil_font(31)
    small = pil_font(25)
    left, top, right, bottom = 180, 110, 980, 690
    draw.rectangle((left, top, right, bottom), outline="#cbd5e1", width=3)
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        draw.line((left, y, right, y), fill="#e2e8f0", width=2)
    x0 = 390
    old = []
    new = []
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        old.append((x0, int(y)))
        new.append((x0 + 70 * i, int(y)))
    draw.line(old, fill="#64748b", width=4)
    draw.line(new, fill="#2563eb", width=5)
    for p0, p1 in zip(old, new):
        draw.ellipse((p0[0] - 10, p0[1] - 10, p0[0] + 10, p0[1] + 10), fill="#64748b")
        draw.ellipse((p1[0] - 12, p1[1] - 12, p1[0] + 12, p1[1] + 12), fill="#2563eb")
        arrow(draw, (p0[0] + 16, p0[1]), (p1[0] - 18, p1[1]), "#ea580c", 3)
    arrow(draw, (610, 450), (850, 250), "#16a34a", 6)
    draw.text((865, 232), "e_p", font=f, fill="#16a34a")
    draw.text((245, 45), "coherent tilt: accumulated partial_z V_c controls tilt", font=f, fill="#0f172a")
    draw.text((450, 720), "horizontal projected position", font=small, fill="#334155")
    draw.text((120, 95), "z", font=small, fill="#334155")
    draw.text((790, 640), "time-evolved tilt", font=small, fill="#2563eb")
    draw.text((220, 640), "initial vertical core", font=small, fill="#64748b")
    save_img(img, "fig5_coherent_tilt.png")


def generate_figures() -> None:
    fig1_tem_vs_eulerian()
    fig2_ep_flux_vector()
    fig3_j1_j2_comparison()
    fig4_qbar_feedback()
    fig5_coherent_tilt()


def clean_latex_aux() -> None:
    for suffix in [".aux", ".log", ".out"]:
        for artifact in [BUILD_TEX.with_suffix(suffix), TEX_PATH.with_suffix(suffix)]:
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
    clean_latex_aux()
    return PDF_PATH


def main() -> None:
    generate_figures()
    output = compile_latex()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
