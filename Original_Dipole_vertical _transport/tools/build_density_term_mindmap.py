"""Build a mind-map figure for W term and density-processing differences."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_W = 3000
CANVAS_H = 2300
MARGIN = 70


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(48, True)
FONT_SUBTITLE = font(25)
FONT_BOX_TITLE = font(29, True)
FONT_BODY = font(21)
FONT_BODY_BOLD = font(22, True)
FONT_FOOT = font(23)


def wrap_mixed(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    if text.startswith("•") or text[0].isdigit():
        prefix = text[:2] if text.startswith("•") else text[:3]
        wrapped = textwrap.wrap(text, width=max_chars, break_long_words=False)
        if len(wrapped) <= 1:
            return wrapped
        return [wrapped[0], *["  " + line for line in wrapped[1:]]]
    return textwrap.wrap(text, width=max_chars, break_long_words=False) or [text]


def rounded_box(draw, xy, fill, outline, width=4, radius=28):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_box(draw, xy, title, lines, fill, outline, title_color, max_chars=43):
    x0, y0, x1, y1 = xy
    rounded_box(draw, xy, fill, outline)
    draw.text(((x0 + x1) / 2, y0 + 26), title, font=FONT_BOX_TITLE, fill=title_color, anchor="ma")
    y = y0 + 78
    for raw in lines:
        if raw == "":
            y += 10
            continue
        section = not raw.startswith("•") and ("口径" in raw or "来源" in raw or "定义" in raw or "平滑" in raw or "形式" in raw or "检查" in raw)
        fnt = FONT_BODY_BOLD if section else FONT_BODY
        fill_color = title_color if section else "#1f2937"
        for line in wrap_mixed(raw, max_chars=max_chars):
            draw.text((x0 + 32, y), line, font=fnt, fill=fill_color)
            y += 29
        y += 2


def arrow(draw, start, end, fill="#475569", width=5):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) > abs(y1 - y0):
        sign = 1 if x1 > x0 else -1
        pts = [(x1, y1), (x1 - sign * 24, y1 - 13), (x1 - sign * 24, y1 + 13)]
    else:
        sign = 1 if y1 > y0 else -1
        pts = [(x1, y1), (x1 - 13, y1 - sign * 24), (x1 + 13, y1 - sign * 24)]
    draw.polygon(pts, fill=fill)


def build(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), "#f8fafc")
    draw = ImageDraw.Draw(image)

    draw.text(
        (CANVAS_W / 2, 50),
        "W 重建中 term1 / term2 与密度处理思维导图",
        font=FONT_TITLE,
        fill="#0f172a",
        anchor="ma",
    )
    draw.text(
        (CANVAS_W / 2, 118),
        "核心问题：整体等密面集合 vs BOA 背景扣除后的异常等密面起伏；同时检查差分、梯度、速度和符号链条",
        font=FONT_SUBTITLE,
        fill="#475569",
        anchor="ma",
    )

    term1 = (70, 210, 1415, 860)
    term2 = (1585, 210, 2930, 860)
    dens = (70, 985, 1415, 1745)
    algo = (1585, 985, 2930, 1745)
    focus = (360, 1860, 2640, 2185)

    draw_box(
        draw,
        term1,
        "Term1：涡旋传播诱导  c ∂zρ/∂x",
        [
            "物理含义：涡旋水平传播扫过倾斜等密面时产生垂向运动。",
            "",
            "前辈口径",
            "• 使用合成后的密度场 Den_compound(x,y,z)。",
            "• 对每个格点/深度取 rho0 = rho_comp(i,j,z0)。",
            "• 到左右相邻整条密度柱反插同一 rho0，得到 z_left/z_right。",
            "• 中心差分：dzdx = (z_right - z_left) / (2 dx)。",
            "• W_dzdt = c0 · dzdx，c0 多取纬向传播速度的中位/平均量。",
            "",
            "当前正式口径",
            "• rho0 = BOA(lon,lat,month,z0) 的局地气候态密度。",
            "• z′ρ = zρ(profile,rho0) - zρ(BOA,rho0)，正深度向下。",
            "• Cressman 合成 z′ρ(x/R,y/R,z)，再求 gradient。",
            "• term1 = + c_x_rel · ∂z′ρ/∂x，W 向上为正。",
        ],
        "#eef6ff",
        "#2563eb",
        "#1d4ed8",
    )
    draw_box(
        draw,
        term2,
        "Term2：水平流沿等密面坡度平流  u · ∇zρ",
        [
            "物理含义：水平速度穿过等密面坡度，等效产生垂向速度。",
            "",
            "前辈口径",
            "• Term2 的斜率常来自另一个合成密度场 Den_compound_all / ISAS 相关背景。",
            "• 先把密度场插到统一深度，再对整条密度柱做同 rho0 反插。",
            "• dzdx/dzdy 用左右/南北相邻柱的 z_rho 中心差分。",
            "• U_thw/V_thw 用热成风从密度梯度积分，并锚定 1000 m 速度。",
            "• W_is = U_thw · dzdx + V_thw · dzdy。",
            "",
            "当前正式口径",
            "• term2 与 term1 共用 BOA z′ρ 异常几何。",
            "• 速度用历史 Argo1000m I_Upk/I_Vpk，并用热成风延拓到 z。",
            "• 相对速度：u_rel = (u_tw - c_x_raw, v_tw)。",
            "• term2 = - u_rel · ∇z′ρ；负号来自 W 向上为正而 z 为正深度向下。",
        ],
        "#fff7ed",
        "#ea580c",
        "#c2410c",
    )
    draw_box(
        draw,
        dens,
        "平行链条 A：密度来源、背景和异常定义",
        [
            "前辈密度来源",
            "• Argo03_compound_*：Den_compound，用于 term1 斜率和热成风。",
            "• Argo04_*_ISAS_7Sample：Den_compound_all，用于 term2 斜率。",
            "• 这不是逐 profile 的 BOA 扣背景，而是合成密度场的整体几何。",
            "",
            "当前密度来源",
            "• ArgoData_SA_CT_PT_PDen_sigma.mat：逐 profile TEOS-10 密度。",
            "• Self_BOA_Argo_PotentialDensity：多年同月 BOA 背景。",
            "• 按 lon/lat/month/z0 取 rho_BOA，严格 bracket 反插。",
            "",
            "关键定义差异",
            "• 前辈：整体等密面集合 zρ[rho_comp(x,y,z0)]。",
            "• 当前：异常等密面起伏 z′ρ = z_profile - z_BOA。",
            "• 前者保留大尺度背景坡度；后者强调相对 BOA 背景的局地异常。",
        ],
        "#f0fdf4",
        "#16a34a",
        "#15803d",
    )
    draw_box(
        draw,
        algo,
        "平行链条 B：映射、差分、热成风和坐标风险",
        [
            "映射/平滑",
            "• 前辈：先有合成场，再二维平滑；部分程序用矩形窗口多次平滑。",
            "• 当前：样本点先 Cressman 到 x/R,y/R 网格，再支撑计数和平滑。",
            "• 图像规整度主要受 Rc、min_obs、smooth_passes、网格分辨率影响。",
            "",
            "差分形式",
            "• 前辈：手写中心差分，先反插相邻柱的 zρ，再 (right-left)/(2dx)。",
            "• 当前：先得到 z′ρ 网格场，再 gradient(z′ρ,dx,dy)。",
            "• 二者不等价的核心不是差分阶数，而是被求梯度的几何对象不同。",
            "",
            "坐标/符号检查",
            "• x/R 必须是局地东西向，y/R 必须是局地南北向。",
            "• MATLAB gradient 的行/列输出容易和 x/y 语义混淆，需显式校验。",
            "• W 向上为正；zρ、z′ρ 和图像深度仍按正深度向下保存/显示。",
        ],
        "#fef2f2",
        "#dc2626",
        "#b91c1c",
    )

    rounded_box(draw, (520, 890, 2480, 955), "#ffffff", "#cbd5e1", width=3, radius=20)
    draw.text(
        (1500, 910),
        "两条 term 都依赖同一个核心选择：到底对“整体 zρ”求坡度，还是对“扣除背景后的 z′ρ”求坡度？",
        font=FONT_BODY_BOLD,
        fill="#334155",
        anchor="ma",
    )

    arrow(draw, (742, 860), (742, 985), "#2563eb")
    arrow(draw, (2258, 860), (2258, 985), "#ea580c")
    arrow(draw, (1415, 1290), (1585, 1290), "#64748b")
    arrow(draw, (742, 1745), (1070, 1860), "#16a34a")
    arrow(draw, (2258, 1745), (1930, 1860), "#dc2626")

    draw_box(
        draw,
        focus,
        "需要优先验证的密度相关问题",
        [
            "1. 去掉的 BOA 背景等密面坡度是否在 1R-4R 内真的可忽略。",
            "2. z′ρ 异常几何是否会天然抑制深层反转或改变 term2 相位。",
            "3. term1/term2 是否应该共用同一个 z 几何，还是分别使用传播等密面和背景/整体等密面。",
            "4. 热成风速度使用 rho_anom 梯度还是 composite rho 梯度，会直接改变 u(z),v(z) 的垂向结构。",
            "5. 所有对照必须固定 x=东西、y=南北，避免偶极方向被坐标转置伪造。",
        ],
        "#fdf4ff",
        "#9333ea",
        "#7e22ce",
        max_chars=82,
    )

    png_path = output_root / "density_term_mindmap_term1_term2.png"
    image.save(png_path, optimize=True)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=r"E:\DATA\01_Eddy_correspond\05_Original_Dipole_vertical _transport",
        help="Directory where the mind-map PNG will be written.",
    )
    args = parser.parse_args()
    png_path = build(Path(args.output_root))
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
