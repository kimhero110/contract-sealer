"""生成合成测试素材（确定性，seed 固定）。

仓库不含真实合同/印章图片（安全决策：真实素材永不入库）。
本脚本生成的合成素材与真实素材同文件名，测试直接可用：

    python tests/fixtures/generate_fixtures.py
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑；缺失时退化为英文/图形


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _arc_text(draw: ImageDraw.ImageDraw, center, radius, text, font, fill, start_deg=180, span_deg=360):
    """沿圆弧排布文字（模拟公章环形字）。"""
    n = len(text)
    for i, ch in enumerate(text):
        ang = np.deg2rad(start_deg - span_deg * i / max(n - 1, 1) + 90)
        x = center[0] + radius * np.cos(ang)
        y = center[1] + radius * np.sin(ang)
        # 每个字旋转到切线方向
        tmp = Image.new("RGBA", (font.size * 2, font.size * 2), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        td.text((font.size // 2, font.size // 2), ch, font=font, fill=fill)
        rot = tmp.rotate(-(np.rad2deg(ang) + 90), resample=Image.BICUBIC, center=(font.size, font.size))
        draw._image.paste(rot, (int(x - font.size), int(y - font.size)), rot)


def make_round_seal(path: Path, company: str, code: str, size: int = 800, ring_ratio: float = 0.97):
    """合成圆形公章：外环 + 五角星 + 环形公司名 + 底部编号。RGBA 白底。"""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    red = (200, 30, 30, 255)
    c = size / 2
    r_out = size * ring_ratio / 2
    d.ellipse([c - r_out, c - r_out, c + r_out, c + r_out], outline=red, width=max(6, size // 90))
    # 五角星
    r_star = size * 0.16
    pts = []
    for i in range(10):
        r = r_star if i % 2 == 0 else r_star * 0.382
        ang = -np.pi / 2 + i * np.pi / 5
        pts.append((c + r * np.cos(ang), c + r * np.sin(ang)))
    d.polygon(pts, fill=red)
    # 环形公司名
    _arc_text(d, (c, c), r_out * 0.78, company, _font(int(size * 0.075)), red)
    # 底部编号
    f_small = _font(int(size * 0.05))
    w = d.textlength(code, font=f_small)
    d.text((c - w / 2, c + r_out * 0.72), code, font=f_small, fill=red)
    img.save(path)


def make_square_seal(path: Path, name: str, size: int = 500):
    """合成方形法人章：边框 + 名字 + "印"。"""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    red = (200, 30, 30, 255)
    m = size // 20
    d.rectangle([m, m, size - m, size - m], outline=red, width=max(5, size // 60))
    f = _font(int(size * 0.28))
    chars = list(name) + ["印"]
    per_row = 2
    for i, ch in enumerate(chars[:4]):
        row, col = divmod(i, per_row)
        x = size * (0.28 + col * 0.30)
        y = size * (0.16 + row * 0.34)
        d.text((x, y), ch, font=f, fill=red)
    img.save(path)


def make_signature(path: Path, seed: int, w: int = 900, h: int = 400):
    """合成手写签名：多条随机平滑曲线（黑/深蓝墨迹，白底）。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    ink_colors = [(20, 20, 20), (10, 10, 90)]
    for s in range(rng.integers(3, 6)):
        n_pts = rng.integers(4, 8)
        xs = np.sort(rng.uniform(0.05, 0.95, n_pts)) * w
        ys = (0.5 + rng.uniform(-0.3, 0.3, n_pts)) * h
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        # 样条插值使笔画平滑
        t = np.linspace(0, 1, n_pts)
        tt = np.linspace(0, 1, 120)
        sx = np.interp(tt, t, pts[:, 0])
        sy = np.interp(tt, t, pts[:, 1]) + np.sin(tt * rng.uniform(3, 9)) * h * 0.06
        curve = np.stack([sx, sy], axis=1).astype(np.int32)
        color = ink_colors[s % 2]
        cv2.polylines(img, [curve], False, color, thickness=int(rng.integers(2, 5)), lineType=cv2.LINE_AA)
    Image.fromarray(img).save(path)


def make_contract_page(path: Path, page_no: int, w: int = 2544, h: int = 3504):
    """合成 A4 扫描风格合同页（300DPI 像素尺寸，白底黑字）。"""
    img = np.full((h, w, 3), 252, dtype=np.uint8)
    rng = np.random.default_rng(page_no)
    # 模拟文字行：深浅不一的灰色横条
    y = 300
    f = _font(48)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.text((w // 2 - 200, 150), f"测试合同 第{page_no}页", font=_font(72), fill=(30, 30, 30))
    while y < h - 400:
        line_w = int(w * rng.uniform(0.55, 0.8))
        d.text((200, y), "这是一段用于测试的合同条款文字，仅作合成素材。" * 2, font=f, fill=(50, 50, 50))
        y += 90
    # 轻微扫描噪声与底色不均
    noise = rng.normal(0, 2.5, (h, w, 1)).astype(np.float32)
    out = np.array(pil).astype(np.float32) + noise
    img = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=92)


def main() -> None:
    make_contract_page(HERE / "1.jpg", 1)
    make_contract_page(HERE / "2.jpg", 2)
    make_contract_page(HERE / "3.jpg", 3)
    make_round_seal(HERE / "seal_company.png", "测试合成科技有限公司", "3200000000000")
    make_square_seal(HERE / "seal_person.png", "张三")
    make_signature(HERE / "sig_lsl.png", seed=11)
    make_signature(HERE / "sig_hxd.png", seed=22, w=700, h=900)
    print("合成素材已生成到", HERE)


if __name__ == "__main__":
    main()
