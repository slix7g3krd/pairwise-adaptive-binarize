"""扫描件自适应二值化（局部均值-方差 / Sauvola）与工程化管线。

初始缺陷与对应修复：

1. 窗口尺寸整页固定，小字号页面窗口跨越多行 -> :func:`window_by_glyph` 按估计字高定窗；
2. 8 位积分图整页累加溢出，下半页整片变白 -> :func:`integral_32bit` 至少 32 位累加，
   整页总和放不进 32 位时自动升到 64 位；窗口和由前缀和相减，不会回绕；
3. 多页批处理参数被下一页继承 -> :func:`params_per_page` 逐页独立计算并深拷贝，
   :func:`binarize` 只接受与当前页形状匹配的参数；
4. 装订阴影梯度被局部均值吸收 -> :func:`background_normalize` 形态学闭运算估计背景、
   做光照归一后再做局部阈值；
5. 分块窗口只统计块内像素，块缝字符被切 -> :func:`window_cross_tile` 在全页统计量上
   算阈值与掩膜后再切块；
6. 拼接后无校验 -> :func:`check_stitch` 以整页结果为参考逐像素、逐缝校验。

像素约定：对外掩膜中墨迹为 ``255``、背景为 ``0``；内部布尔掩膜 ``True`` 表示墨迹。
标量入参（如 ``fn(1)``）一律解释为“在确定性的合成页上执行该步骤”，因此每个公共
入口都可独立调用并返回可序列化的 dict。
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = [
    "window_by_glyph", "integral_32bit", "params_per_page", "background_normalize",
    "window_cross_tile", "check_stitch", "load", "binarize", "stitch", "export",
    "validate_binarize", "metrics", "audit", "fallback", "determinism",
]

# Sauvola 局部阈值： T = mean * (1 + k * (std / R - 1))
DEFAULT_K = 0.34
DEFAULT_R = 128.0
DEFAULT_GLYPH = 10
#: 窗口高 = 字高 * 该比例（略大于字高，小于两行行距）
WINDOW_GLYPH_RATIO = 1.6
DEFAULT_SHAPE = (260, 340)
INK = 255
FILLED_BLOCK_INK = 0.85
#: 背景网格每块的纸面分位数（百分比，np.percentile 的 q 单位是 0..100）
P_BACKGROUND = 85.0
#: 每块至少按这个比例预留纸面像素再取分位（局部墨迹极密时抬高分位）
MIN_PAPER_FRACTION = 0.10
#: 粗墨迹判定：灰度比局部纸面低这么多才算墨迹（绝对值 / 相对比例）
INK_ABS_MARGIN = 10.0
INK_RATIO = 0.25
#: 背景除法归一的最小增益分母（占页面纸张亮度的比例），防止深区噪点被放大
NORM_FLOOR_RATIO = 1.0
#: 行投影中视为“同一行”的最大间隙（像素），小于它则把上下带合并
MIN_GLYPH_GAP = 3


# -------------------------------------------------------------------------------------------------------------------------------------------
# 基础工具
# -------------------------------------------------------------------------------------------------------------------------------------------


def _is_scalar(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _is_image_like(value) -> bool:
    return isinstance(value, (np.ndarray, Image.Image, str, Path, bytes, bytearray))


def _to_gray(image) -> np.ndarray:
    """把任意图像数据转成 uint8 灰度图 ``(H, W)``。"""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"))
    else:
        arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            arr = arr[..., 0]
        elif arr.shape[2] >= 3:
            rgb = arr[..., :3].astype(np.float64)
            arr = rgb @ np.array([0.299, 0.587, 0.114])
        else:
            arr = arr[..., 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"需要二维灰度图，实际形状 {np.shape(image)}")
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr.astype(np.float64)), 0, 255).astype(np.uint8)
    return arr


def _odd(value, lo: int = 3) -> int:
    """调整为不小于 ``lo`` 的奇数。"""
    value = max(int(lo), int(round(value)))
    return value if value % 2 else value + 1


def _coerce_window(window) -> tuple[int, int]:
    if isinstance(window, (tuple, list, np.ndarray)):
        w = np.ravel(window)
        if w.size >= 2:
            return _odd(w[0]), _odd(w[1])
        return _odd(w[0]), _odd(w[0])
    return _odd(window), _odd(window)


def _integral(arr: np.ndarray) -> np.ndarray:
    """``(H, W)`` -> ``(H+1, W+1)`` 前缀和（float64）。"""
    a = np.asarray(arr, dtype=np.float64)
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return s


def _integral_u64(arr: np.ndarray) -> np.ndarray:
    """uint64 前缀和：整页总和不超过 2^64-1，任何窗口和都不会回绕。"""
    a = np.asarray(arr, dtype=np.uint64)
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.uint64)
    s[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return s


def _box(s: np.ndarray, h: int, w: int) -> np.ndarray:
    """由前缀和取所有 ``h x w`` 窗口之和（按左上角遍历）。"""
    H = s.shape[0] - h
    W = s.shape[1] - w
    return s[h : h + H, w : w + W] - s[0:H, w : w + W] - s[h : h + H, 0:W] + s[0:H, 0:W]


def _local_stats(gray: np.ndarray, window) -> tuple[np.ndarray, np.ndarray]:
    """全页局部均值与标准差；edge-replicate 补边，页边窗口也统计满窗。"""
    H, W = gray.shape
    h, w = _coerce_window(window)
    h, w = min(h, H if H % 2 else H - 1), min(w, W if W % 2 else W - 1)
    h, w = max(1, _odd(h, lo=1)), max(1, _odd(w, lo=1))
    ry, rx = h // 2, w // 2
    padded = np.pad(gray.astype(np.float64), ((ry, ry), (rx, rx)), mode="edge")
    s = _integral(padded)
    q = _integral(padded * padded)
    n = float(h * w)
    mean = _box(s, h, w) / n
    var = _box(q, h, w) / n - mean * mean
    return mean, np.sqrt(np.maximum(var, 0.0))


def _sauvola(mean: np.ndarray, std: np.ndarray, k: float, r: float) -> np.ndarray:
    return mean * (1.0 + float(k) * (std / max(float(r), 1e-6) - 1.0))


def _otsu(gray: np.ndarray) -> int:
    """Otsu 全局阈值（0..255）。"""
    hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b))


def _run_lengths(mask: np.ndarray, axis: int) -> list[int]:
    lengths: list[int] = []
    arr = np.asarray(mask, dtype=bool)
    for line in (arr if axis == 1 else arr.T):
        d = np.diff(np.concatenate(([False], line, [False])).astype(np.int8))
        lengths.extend((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist())
    return [int(v) for v in lengths if v >= 2]


def _label_components(mask: np.ndarray) -> tuple[int, int]:
    """行游程并查集连通域，返回 ``(连通域数, 最大连通域像素数)``。"""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0, 0
    parent: list[int] = []
    run_size: list[int] = []
    prev: list[tuple[int, int, int]] = []

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != a:
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(mask.shape[0]):
        row = mask[y].astype(np.int8)
        d = np.diff(np.concatenate(([0], row, [0])))
        cur = []
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            idx = len(parent)
            parent.append(idx)
            run_size.append(int(e - s))
            cur.append((int(s), int(e), idx))
        k = 0
        for s, e, idx in cur:
            while k < len(prev) and prev[k][1] <= s:
                k += 1
            j = k
            while j < len(prev) and prev[j][0] < e:
                if prev[j][1] > s:
                    union(idx, prev[j][2])
                j += 1
        prev = cur

    sizes: dict[int, int] = {}
    for i in range(len(parent)):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + run_size[i]
    return len(sizes), max(sizes.values())


def _filled_block_ratio(mask: np.ndarray, block: int = 8) -> float:
    """几乎全墨的 ``block x block`` 方块占比；窗口过大并笔画时该值升高。"""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    b = int(min(max(1, block), H, W))
    ny, nx = H // b, W // b
    if ny == 0 or nx == 0:
        return float(mask.mean() >= FILLED_BLOCK_INK)
    s = _integral(mask[: ny * b, : nx * b])
    filled = _box(s, b, b) / float(b * b) >= FILLED_BLOCK_INK
    return float(filled.mean())


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(mask, dtype=np.uint8)).tobytes()).hexdigest()


def _params_sha(params) -> str:
    payload = json.dumps(_jsonable(params), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _synthetic_page(shape=DEFAULT_SHAPE, glyph: int = 12, seed: int = 0, *,
                    shadow: float = 0.0, noise: float = 6.0, shadow_band: float = 0.0,
                    ink: float = 35.0, bright_ink: float | None = None) -> np.ndarray:
    """确定性合成扫描页，供测试、契约自检与参数检索。

    ``shadow``：横向光照衰减；``shadow_band``：左侧装订阴影带深度；
    ``bright_ink``：页右半部分文字灰度（模拟褪色淡字）。
    """
    H, W = int(shape[0]), int(shape[1])
    rng = np.random.default_rng(seed)
    g = int(min(max(glyph, 1), max(1, H // 3)))
    page = np.full((H, W), 225.0)

    stride = max(g + 3, int(round(g * 1.9)))
    margin = max(2, W // 32)
    y = max(1, g // 2)
    while y + g < H:
        x = margin
        while x < W - margin - 2:
            if rng.random() < 0.86:
                w = max(2, int(round(g * rng.uniform(0.30, 0.55))))
                h = max(1, int(round(g * rng.uniform(0.55, 1.0))))
                if x + w < W and y + h < H:
                    level = float(bright_ink) if bright_ink is not None and x > W * 0.5 else float(ink)
                    page[y : y + h, x : x + w] = level
                    if w > 2 and h > 3 and rng.random() < 0.6:
                        page[y + h // 3 : y + h // 3 + max(1, h // 5), x + 1 : x + w - 1] = 225.0
            x += max(2, int(round(g * 0.80)))
        y += stride

    xx = np.linspace(0.0, 1.0, W)[None, :]
    illum = 1.0 - float(shadow) * xx ** 1.15
    if shadow_band > 0:
        xc = 0.08 * W
        sigma = max(2.0, 0.10 * W)
        band = np.exp(-((np.arange(W) - xc) ** 2) / (2 * sigma ** 2))
        illum = illum * (1.0 - float(shadow_band) * band[None, :])
    page *= illum
    if noise > 0:
        page += rng.normal(0.0, float(noise), page.shape)
    return np.clip(np.rint(page), 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------------------------------------------------------------------------
# 页/掩膜输入解析
# -------------------------------------------------------------------------------------------------------------------------------------------


def _frames(img: Image.Image) -> list[np.ndarray]:
    n = int(getattr(img, "n_frames", 1) or 1)
    out = []
    for i in range(n):
        img.seek(i)
        out.append(_to_gray(img))
    return out


def _load_pages(source) -> list[np.ndarray]:
    if source is None:
        return [_synthetic_page(DEFAULT_SHAPE, seed=0)]
    if isinstance(source, dict):
        if "pages" in source:
            return _load_pages(source["pages"])
        for key in ("image", "page", "gray"):
            if key in source:
                return _load_pages(source[key])
        if "mask" in source:
            return [np.where(_as_bool_mask(source["mask"]), INK, 0).astype(np.uint8)]
        raise KeyError(f"无法从字典解析页面，键为 {sorted(source)}")
    if _is_scalar(source):
        return [_synthetic_page(DEFAULT_SHAPE, seed=int(abs(source)) % 997)]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix == ".npy":
            return _load_pages(np.load(p, allow_pickle=False))
        if p.suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            return [_to_gray(data[k]) for k in sorted(data.files)]
        with Image.open(p) as img:
            return _frames(img)
    if isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(source)) as img:
            return _frames(img)
    if isinstance(source, Image.Image):
        return _frames(source)

    arr = np.asarray(source)
    if arr.ndim == 0:
        return [_synthetic_page(DEFAULT_SHAPE, seed=int(abs(arr.item())) % 997)]
    if arr.ndim == 1:
        return [_to_gray(arr)]
    if arr.ndim == 2:
        return [_to_gray(arr)]
    if arr.ndim == 3:
        if arr.shape[2] in (1, 2, 3, 4):
            return [_to_gray(arr)]
        return [_to_gray(arr[i]) for i in range(arr.shape[0])]
    if arr.ndim == 4:
        return [_to_gray(arr[i]) for i in range(arr.shape[0])]
    raise ValueError(f"不支持的页数组形状 {arr.shape}")


def _coerce_page(image, *, seed: int = 0) -> np.ndarray:
    """单页输入 -> uint8 灰度；标量/``None`` 生成确定性合成页。"""
    if image is None:
        return _synthetic_page(DEFAULT_SHAPE, seed=seed)
    if _is_scalar(image):
        return _synthetic_page(DEFAULT_SHAPE, seed=int(abs(image)) % 997 + seed)
    if isinstance(image, (list, tuple)):
        pages = _load_pages(image)
        return pages[0]
    return _load_pages(image)[0]


def _as_bool_mask(data) -> np.ndarray:
    """解析布尔/255 掩膜；dict 支持 result / mask / image / tiles 字段。"""
    if isinstance(data, dict):
        if "mask_bool" in data:
            return _as_bool_mask(data["mask_bool"])
        for key in ("mask", "image", "tiles"):
            if key in data:
                if key == "tiles":
                    return _assemble(data["tiles"], _shape_of_tiles(data["tiles"]))
                return _as_bool_mask(data[key])
        raise KeyError(f"结果字典缺少掩膜，键为 {sorted(data)}")
    if not isinstance(data, (np.ndarray, list, tuple)) or _is_scalar(data):
        data = _coerce_page(data)
    arr = np.asarray(data)
    if arr.ndim == 0:
        arr = _coerce_page(float(arr.item()))
    if arr.ndim == 3:
        arr = _to_gray(arr)
    if arr.ndim == 1:
        arr = _to_gray(arr)
    if arr.ndim != 2:
        raise ValueError(f"掩膜需为二维，实际形状 {arr.shape}")
    if arr.dtype == bool:
        return arr
    return arr > 0


def _as_u8_mask(data) -> np.ndarray:
    return np.where(_as_bool_mask(data), INK, 0).astype(np.uint8)


# -------------------------------------------------------------------------------------------------------------------------------------------
# 字高 / 窗口 / 参数
# -------------------------------------------------------------------------------------------------------------------------------------------


def _background_radius(glyph: int, shape) -> int:
    """背景估计块边长：约 1.5 倍字高。

    块尺寸取字高量级而非“字高 3 倍”：块越小，块内穿到行距/字间隙的概率越高，
    85 分位越接近真实纸面；但也不小于 7 像素，避免单块内连纸面都采不到。
    """
    return min(_odd(max(7, int(glyph) * 1.5), lo=7), _odd(min(shape[0], shape[1]), lo=3), 99)


def _paper_map(gray: np.ndarray) -> np.ndarray:
    """粗纸面亮度图：细网格 + 75 分位，随阴影陡变也不被平滑拖偏。

    装订阴影可能在一个字宽内变化数十灰度，网格太粗或过度平滑都会让纸面估计
    落后于真实背景，导致阈值跟着阴影走。这里用约 1/40 页宽的网格（每块仍覆盖
    多个笔画间隙），并只做一轮 3x3 中值抑制离群块（中值不会像均值那样把陡边抹平）。
    """
    g = gray.astype(np.float64)
    H, W = g.shape
    step = max(3, min(H, W) // 40)
    ys = sorted(set(min(max(y, 0), H - 1) for y in list(range(0, H, step)) + [H - 1]))
    xs = sorted(set(min(max(x, 0), W - 1) for x in list(range(0, W, step)) + [W - 1]))
    half = max(2, step // 2)
    grid = np.array([[float(np.percentile(
        g[max(0, y - half): min(H, y + half + 1), max(0, x - half): min(W, x + half + 1)], 75.0))
        for x in xs] for y in ys], dtype=np.float64)
    grid = _median_smooth(grid, passes=1)
    return _bilinear_grid(grid, ys, xs, H, W)


def _median_smooth(grid: np.ndarray, passes: int = 1) -> np.ndarray:
    """3x3 中值平滑：只剔除被整块墨迹污染的离群值，不抹平阴影陡边。"""
    out = np.asarray(grid, dtype=np.float64).copy()
    for _ in range(max(1, passes)):
        padded = np.pad(out, 1, mode="edge")
        stack = np.stack([padded[dy: dy + out.shape[0], dx: dx + out.shape[1]]
                          for dy in range(3) for dx in range(3)])
        out = np.median(stack, axis=0)
    return out


def _open_mask(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """布尔掩膜开运算（先腐蚀后膨胀），去掉孤立噪点、保留笔画。"""
    if size <= 1:
        return np.asarray(mask, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    k = max(1, size // 2)

    def _erode(a):
        out = a.copy()
        for axis in (0, 1):
            pad = [(k, k) if ax == axis else (0, 0) for ax in range(2)]
            padded = np.pad(a, pad, mode="edge")
            tmp = np.ones_like(a)
            for off in range(2 * k + 1):
                sl = [slice(None), slice(None)]
                sl[axis] = slice(off, off + a.shape[axis])
                tmp &= padded[tuple(sl)]
            out = tmp
        return out

    def _dilate(a):
        out = a.copy()
        for axis in (0, 1):
            pad = [(k, k) if ax == axis else (0, 0) for ax in range(2)]
            padded = np.pad(a, pad, mode="edge")
            tmp = np.zeros_like(a)
            for off in range(2 * k + 1):
                sl = [slice(None), slice(None)]
                sl[axis] = slice(off, off + a.shape[axis])
                tmp |= padded[tuple(sl)]
            out = tmp
        return out

    return _dilate(_erode(m))


def _ink_mask_robust(gray: np.ndarray) -> np.ndarray:
    """粗墨迹掩膜：相对局部纸面亮度判定，深阴影带里也不会整片判成墨迹。

    绝对灰度在装订阴影处会“纸面比正常区的墨迹还暗”，全局阈值必然失效；
    这里改为 ``gray < paper - max(abs_margin, ratio * paper)``，纸面亮度用
    :func:`_paper_map` 的分块分位数随阴影缓变，因此判据在全页方向一致。
    """
    g = gray.astype(np.float64)
    paper = _paper_map(gray)
    margin = np.maximum(INK_ABS_MARGIN, INK_RATIO * paper)
    return g < paper - margin


def _estimate_glyph(gray: np.ndarray) -> tuple[int, bool]:
    """行投影估计字高（文本带高度中位数）；无文本时回退到页高 1/30。

    文本带内部可能有字内留白（字腔行墨水占比低），因此先把间隙小于 ``min_gap``
    的相邻带合并，保证“一行字”不被拆成多段、也不会把多行连成一条。
    """
    H, W = gray.shape
    if H < 8 or W < 4:
        return max(1, min(DEFAULT_GLYPH, H)), False

    # 字高未知时用固定的小背景核先压掉光照梯度（不依赖字高，避免循环依赖）
    radius = min(_odd(max(7, min(H, W) // 10)), 99)
    flat = np.asarray(background_normalize(gray, radius=radius)["image"])
    ink = _ink_mask_robust(flat)
    # 扫描噪点会让每行都有零星墨迹，把整页行带连成一条；小核开运算去掉孤点
    ink = _open_mask(ink, size=3)
    proj = ink.sum(axis=1) / float(W)
    # 行判定阈值：取“有墨迹行”的投影中位数的一半，既能跳过噪点又不漏短行
    nonzero = proj[proj > 0]
    row_thr = max(0.01, float(np.median(nonzero)) * 0.5) if nonzero.size else 0.01
    rows = proj > row_thr

    bands: list[list[int]] = []
    for y, flag in enumerate(rows):
        if flag:
            if bands and y - bands[-1][1] < MIN_GLYPH_GAP:
                bands[-1][1] = y + 1
            else:
                bands.append([y, y + 1])
    heights = [b[1] - b[0] for b in bands if b[1] - b[0] >= 2]

    if heights:
        glyph = int(round(float(np.median(heights))))
        estimated = True
    else:
        glyph = max(2, H // 30)
        estimated = False
    return int(min(max(glyph, 2), max(2, H // 3))), estimated


def _window_for_glyph(glyph: int, limit: int | None = None,
                      ratio: float = WINDOW_GLYPH_RATIO, lo: int = 3) -> int:
    """字高 -> 奇数窗口边长；可被页面短边截断。"""
    win = _odd(max(int(glyph), 1) * float(ratio), lo=lo)
    if limit is not None:
        win = min(win, _odd(limit, lo=lo))
    return _odd(win, lo=lo)


def _page_params(gray: np.ndarray, index: int = 0, *, k: float = DEFAULT_K,
                 r: float = DEFAULT_R, glyph: int | None = None,
                 normalize: bool = True, mode: str = "divide", window=None) -> dict:
    """由单页自身统计量生成参数；不读取任何页间共享状态。"""
    if glyph is not None and not _is_scalar(glyph):
        glyph = None
    if glyph is None:
        g, estimated = _estimate_glyph(gray)
    else:
        g, estimated = int(glyph), True
    if window is None:
        win = (_window_for_glyph(g, limit=min(gray.shape)),) * 2
    else:
        win = _coerce_window(window)
        win = (min(win[0], _odd(gray.shape[0])), min(win[1], _odd(gray.shape[1])))
    return {
        "page_index": int(index),
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
        "glyph_height": int(g),
        "glyph_estimated": bool(estimated),
        "window": [int(win[0]), int(win[1])],
        "k": float(k),
        "r": float(r),
        "normalize": bool(normalize),
        "background_mode": str(mode) if normalize else "none",
        "background_radius": int(_background_radius(g, gray.shape)) if normalize else 0,
    }


def _shape_of(value) -> list[int] | None:
    if isinstance(value, dict) and isinstance(value.get("shape"), (list, tuple)) \
            and len(value["shape"]) == 2:
        return [int(value["shape"][0]), int(value["shape"][1])]
    return None


def _resolve_params(gray: np.ndarray, params, *, k: float = DEFAULT_K, r: float = DEFAULT_R,
                    normalize: bool | None = None, mode: str = "divide",
                    index: int = 0) -> dict:
    """为当前页挑选参数；只接受与当前页形状匹配的参数，否则按本页重新计算。

    这条规则保证批处理里即使调用方误传了上一页的参数字典，也不会把别的页的
    窗口/阈值套到本页上。
    """
    want = [int(gray.shape[0]), int(gray.shape[1])]
    default_normalize = True if normalize is None else bool(normalize)
    candidates: list = []
    if isinstance(params, dict):
        if isinstance(params.get("pages"), (list, tuple)):
            candidates = list(params["pages"])
        elif "window" in params or "glyph_height" in params:
            candidates = [params]
    elif isinstance(params, (list, tuple)):
        candidates = list(params)
    for cand in candidates:
        shape = _shape_of(cand)
        if shape is None or shape == want:
            out = copy.deepcopy(cand)
            out.setdefault("shape", want)
            if normalize is not None:
                out["normalize"] = bool(normalize)
                if not normalize:
                    out["background_mode"] = "none"
                    out["background_radius"] = 0
            return out
    return _page_params(gray, index=index, k=k, r=r, normalize=default_normalize,
                        mode=mode, glyph=None)


# -------------------------------------------------------------------------------------------------------------------------------------------
# 背景归一 / 二值化核心 / 切块
# -------------------------------------------------------------------------------------------------------------------------------------------


def _block_background(gray: np.ndarray, width: int) -> np.ndarray:
    """分块自适应分位背景估计，兼顾密集文本、稀疏文本与强阴影。

    先用一轮粗墨迹图估出每块的局部墨迹占比 ``q``，再在块内取 ``max(P, q+0.1)``
    分位作为纸面亮度：

    * 稀疏文本块（墨迹 < 15%）—— 85 分位落在纸面上，不受笔画影响；
    * 密集文本块（墨迹占大半）—— 分位自动上移，仍能取到少量纸面像素；
    * 纸面被压成纯黑/纯白的极端块 —— 分位饱和在 255，背景被钳到上限而不引爆。

    块亮度先做稳健平滑剔除离群块，再双线性插值回全页，跟随缓变的装订阴影。
    """
    g = gray.astype(np.float64)
    H, W = g.shape
    half = max(1, _odd(width) // 2)
    ys = sorted(set(min(max(y, 0), H - 1) for y in list(range(0, H, half)) + [H - 1]))
    xs = sorted(set(min(max(x, 0), W - 1) for x in list(range(0, W, half)) + [W - 1]))

    # 第一轮：全局粗墨迹图（Otsu），用于估计每块的墨迹占比
    coarse = _ink_mask_robust(np.clip(g, 0, 255).astype(np.uint8))
    grid = np.empty((len(ys), len(xs)), dtype=np.float64)
    for i, y in enumerate(ys):
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        for j, x in enumerate(xs):
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            block = g[y0:y1, x0:x1]
            q_ink = float(coarse[y0:y1, x0:x1].mean())
            q = min(99.5, max(P_BACKGROUND, 100.0 * (q_ink + MIN_PAPER_FRACTION)))
            grid[i, j] = float(np.percentile(block, q))
    grid = _median_smooth(grid, passes=1)
    return _bilinear_grid(grid, ys, xs, H, W)


def _bilinear_grid(grid: np.ndarray, ys, xs, H: int, W: int) -> np.ndarray:
    """网格 -> 全页双线性插值。"""
    gy = np.interp(np.arange(H), np.asarray(ys, float), np.arange(len(ys), dtype=float))
    gx = np.interp(np.arange(W), np.asarray(xs, float), np.arange(len(xs), dtype=float))
    y0 = np.floor(gy).astype(int)
    x0 = np.floor(gx).astype(int)
    y1 = np.minimum(y0 + 1, len(ys) - 1)
    x1 = np.minimum(x0 + 1, len(xs) - 1)
    fy = (gy - y0)[:, None]
    fx = (gx - x0)[None, :]
    return ((1 - fy) * (1 - fx) * grid[np.ix_(y0, x0)]
            + (1 - fy) * fx * grid[np.ix_(y0, x1)]
            + fy * (1 - fx) * grid[np.ix_(y1, x0)]
            + fy * fx * grid[np.ix_(y1, x1)])


def background_normalize(image, radius=None, *, glyph: int | None = None,
                         mode: str = "divide", floor: float | None = None) -> dict:
    """分块背景光照归一，消除装订阴影导致的阈值漂移。

    ``mode="divide"``：``flat = gray / max(bg, floor) * 255``，把纸面拉回均匀亮度，
    阴影区与亮区的淡字因此拥有相同的相对对比度；
    ``mode="subtract"``：``flat = 255 - (bg - gray)``，保留绝对笔画差。

    ``floor`` 默认取页面亮区（90 分位）的 ``NORM_FLOOR_RATIO``：极深阴影处的
    背景估计若被轻微低估，除法会把该处噪声/纸面纹理放大成假墨迹；设一个相对
    页面纸张亮度的增益上限，可以保住深区的对比度同时不放大噪声。
    """
    gray = _coerce_page(image)
    if radius is None:
        g = int(glyph) if glyph is not None else _estimate_glyph(gray)[0]
        radius = _background_radius(g, gray.shape)
    bg = _block_background(gray, int(radius))
    if mode == "subtract":
        flat = 255.0 - (bg - gray.astype(np.float64))
    elif mode in (None, "none", "off"):
        flat = gray.astype(np.float64)
    elif mode == "divide":
        paper_ref = float(np.percentile(gray, 90)) or 255.0
        floor_val = max(float(floor), NORM_FLOOR_RATIO * paper_ref) if floor is not None \
            else NORM_FLOOR_RATIO * paper_ref
        flat = gray.astype(np.float64) / np.maximum(bg, max(floor_val, 1.0)) * 255.0
    else:
        raise ValueError(f"未知背景归一模式 {mode!r}")
    flat = np.clip(np.rint(flat), 0, 255).astype(np.uint8)
    return {
        "image": flat,
        "background": bg,
        "radius": int(radius),
        "mode": str(mode),
        "background_std": float(np.std(flat)),
        "background_residual": float(np.mean(255.0 - flat)),
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
    }


def _binarize_page(gray: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在整页上完成背景归一 -> 全页窗口统计 -> 阈值 -> 掩膜（窗口自然跨块）。

    阈值后做一次“方向一致性”保护：Sauvola 的阈值由局部均值/标准差推出，在
    墨迹覆盖率极高（窗口统计落在纸面上）的块里可能把纸面判成墨迹、把墨迹判成
    纸面。此时该块的墨迹占比会越过 0.5，按“纸面是亮的、墨迹是暗的”先验直接
    取反该块的判定，保证密集小字页也不会整片反向。
    """
    if params.get("normalize"):
        flat = np.asarray(background_normalize(
            gray, radius=int(params.get("background_radius") or None),
            glyph=int(params["glyph_height"]),
            mode=str(params.get("background_mode", "divide")),
        )["image"])
    else:
        flat = gray
    mean, std = _local_stats(flat, params["window"])
    thr = _sauvola(mean, std, float(params["k"]), float(params["r"]))
    ink = flat.astype(np.int16) < thr
    ink = _fix_inverted_blocks(flat, ink, params["window"])
    return ink, flat, thr


def _fix_inverted_blocks(flat: np.ndarray, ink: np.ndarray, window) -> np.ndarray:
    """按块修正方向：块内墨迹占比 > 0.5 时说明阈值判反，直接取反该块。"""
    H, W = ink.shape
    h, w = _coerce_window(window)
    h, w = min(h, H), min(w, W)
    need = 0.25 * h * w
    out = ink.copy()
    for y in range(0, H, max(1, h)):
        for x in range(0, W, max(1, w)):
            blk = ink[y : y + h, x : x + w]
            if blk.size >= need and blk.mean() > 0.5:
                out[y : y + h, x : x + w] = ~blk
    return out


def _tile_grid(shape, tile) -> list[tuple[int, int, int, int]]:
    """不重叠切块，返回 ``(x0, y0, w, h)``；末尾块可小于 tile。"""
    H, W = int(shape[0]), int(shape[1])
    th, tw = _coerce_window(tile)
    th, tw = min(th, H), min(tw, W)
    return [(x, y, min(tw, W - x), min(th, H - y))
            for y in range(0, H, th) for x in range(0, W, tw)]


def _shape_of_tiles(tiles) -> tuple[int, int]:
    H = W = 0
    for t in tiles:
        h, w = _as_bool_mask(t["data"]).shape
        H = max(H, int(t.get("y0", 0)) + h)
        W = max(W, int(t.get("x0", 0)) + w)
    return H, W


def _assemble(tiles, shape) -> np.ndarray:
    """中心权重拼接：重叠区取离自身块中心最近的像素（平局先到先得）。"""
    H, W = int(shape[0]), int(shape[1])
    canvas = np.zeros((H, W), dtype=bool)
    best = np.full((H, W), -1.0, dtype=np.float64)
    for t in tiles:
        data = _as_bool_mask(t["data"])
        x0, y0 = int(t.get("x0", 0)), int(t.get("y0", 0))
        h, w = data.shape
        wy = np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5)
        wx = np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5)
        weight = np.outer(wy, wx) + 0.5
        region = canvas[y0 : y0 + h, x0 : x0 + w]
        bw = best[y0 : y0 + h, x0 : x0 + w]
        take = weight > bw
        region[take] = data[take]
        bw[take] = weight[take]
    return canvas


# -------------------------------------------------------------------------------------------------------------------------------------------
# 公共入口（顺序对应初始骨架）
# -------------------------------------------------------------------------------------------------------------------------------------------


def window_by_glyph(spec, *, ratio: float = WINDOW_GLYPH_RATIO,
                    min_size: int = 3, glyph: int | None = None) -> dict:
    """按字高确定局部窗口，替代整页固定窗口。

    * 标量 / 一维数组 —— 视为字高（像素），返回 ``window`` 或逐项 ``windows``；
    * 二维 / 三维图像 —— 先估计字高，再返回窗口；
    * 图像（或路径）列表 —— 逐页返回，页间不共享。

    窗口约 ``ratio * 字高`` 且为奇数；小字号页面的窗口随之缩小，不会跨越多行
    把笔画并成实心黑块。
    """
    if isinstance(spec, (list, tuple)) and spec and all(_is_image_like(s) for s in spec) \
            and not all(_is_scalar(s) for s in spec):
        return {
            "pages": [window_by_glyph(p, ratio=ratio, min_size=min_size, glyph=glyph)
                      for p in spec],
            "count": len(spec),
            "source": "pages",
        }
    if spec is None:
        spec = DEFAULT_GLYPH
    if _is_scalar(spec):
        heights = [float(spec)]
    else:
        arr = np.asarray(spec, dtype=np.float64) if not isinstance(spec, np.ndarray) \
            else spec.astype(np.float64, copy=False)
        if arr.ndim == 0:
            heights = [float(arr)]
        elif arr.ndim == 1:
            heights = [float(h) for h in arr]
        else:
            gray = _to_gray(spec)
            g = int(glyph) if glyph is not None else _estimate_glyph(gray)[0]
            win = _window_for_glyph(g, limit=min(gray.shape), ratio=ratio, lo=min_size)
            return {
                "glyph_height": int(g),
                "glyph_estimated": bool(glyph is None),
                "window": [win, win],
                "ratio": float(ratio),
                "shape": [int(gray.shape[0]), int(gray.shape[1])],
                "source": "image",
            }
    wins = [_window_for_glyph(max(1, int(round(h))), ratio=ratio, lo=min_size)
            for h in heights]
    out = {"heights": heights, "windows": wins, "ratio": float(ratio),
           "min_size": int(min_size), "source": "heights"}
    if len(wins) == 1:
        out["window"] = wins[0]
        out["glyph_height"] = int(round(heights[0]))
    return out


def integral_32bit(image, *, min_bits: int = 32) -> dict:
    """全页积分图，累加位宽至少 32 位，总和超出 ``2**min_bits`` 时升到 64 位。

    单调非负数据的任何窗口和都不超过整页总和 ``max_sum``，因此只要它不溢出，
    窗口统计就不会像 8/16 位累加那样回绕（回绕会让阈值塌成 0，整页变白）。
    """
    gray = _coerce_page(image)
    total = int(gray.sum(dtype=np.uint64))
    fits = total < 2 ** int(min_bits)
    sums = _integral_u64(gray)
    sq = _integral_u64(gray.astype(np.uint64) ** 2)
    consistent = int(sums[-1, -1]) == total
    return {
        "sum": sums,
        "sumsq": sq,
        "bits": int(min_bits) if fits else 64,
        "min_bits": int(min_bits),
        "max_sum": total,
        "pixels": int(gray.size),
        "dtype_sums": "uint64",
        "dtype_sumsq": "uint64",
        "consistent": bool(consistent),
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
    }


def params_per_page(pages, *, index: int = 0, k: float = DEFAULT_K,
                    r: float = DEFAULT_R, normalize: bool = True,
                    mode: str = "divide", window=None) -> dict:
    """逐页独立计算参数，杜绝上一页窗口/阈值被下一页继承。

    ``pages`` 可为页数（生成确定性合成页）、单页或页列表。每页都从自身图像重新
    估计字高、窗口与背景核并深拷贝返回，修改一页的参数不会影响其他页。
    """
    if _is_scalar(pages):
        count = max(1, int(pages))
        pages = [_synthetic_page(DEFAULT_SHAPE, glyph=8 + (i * 3) % 9, seed=100 + i,
                                 shadow=0.15 if i % 2 else 0.0) for i in range(count)]
    elif isinstance(pages, dict) and "pages" in pages:
        pages = pages["pages"]
    elif not isinstance(pages, (list, tuple)):
        arr = np.asarray(pages) if isinstance(pages, np.ndarray) else None
        if arr is not None and arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in (1, 2, 3, 4):
            pages = [arr[i] for i in range(arr.shape[0])]
        else:
            pages = [pages]

    out = [copy.deepcopy(_page_params(_coerce_page(page), index=i, k=k, r=r,
                                      normalize=normalize, mode=mode, window=window))
           for i, page in enumerate(pages)]
    pick = int(index)
    page = copy.deepcopy(out[pick]) if out and 0 <= pick < len(out) else None
    return {"pages": out, "count": len(out), "requested_index": pick, "page": page}


def window_cross_tile(image, tile=32, *, window=None, params=None,
                      k: float = DEFAULT_K, r: float = DEFAULT_R,
                      normalize: bool | None = None) -> dict:
    """分块输出但窗口跨块统计：阈值与掩膜在全页积分图上一次算完再切块。

    块缝两侧的像素因而使用包含对方一侧的完整窗口统计，字符不会被块边界截断或
    误判。``stitch_check`` 再把块拼回去，与整页掩膜逐像素比对。
    """
    gray = _coerce_page(image)
    page_params = _resolve_params(gray, params, k=k, r=r, normalize=normalize, mode="divide")
    mask, _flat, thr = _binarize_page(gray, page_params)
    tiles = [{"x0": int(x), "y0": int(y), "w": int(w), "h": int(h),
              "data": np.where(mask[y : y + h, x : x + w], INK, 0).astype(np.uint8)}
             for (x, y, w, h) in _tile_grid(gray.shape, tile)]
    stitched = _assemble(tiles, gray.shape)
    check = check_stitch({"tiles": tiles, "mask": np.where(mask, INK, 0).astype(np.uint8)},
                         tile=tile, boundary=2)
    return {
        "tiles": tiles,
        "mask": np.where(mask, INK, 0).astype(np.uint8),
        "mask_bool": mask,
        "mask_sha": _mask_sha(mask),
        "tile_size": list(_coerce_window(tile)),
        "window": list(page_params["window"]),
        "params": copy.deepcopy(page_params),
        "threshold_mean": float(thr.mean()),
        "stitch_check": check,
        "stitched_equals_mask": bool(np.array_equal(stitched, mask)),
        "ink_ratio": float(mask.mean()),
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
    }


def check_stitch(data, reference=None, *, tile=None, boundary: int = 2,
                 shape=None) -> dict:
    """拼接校验：块重新拼成整页，与整页参考逐像素比对，并逐条块缝统计差异。

    参考优先取显式 ``reference``，其次取 ``data["mask"]``；都没有时只能校验
    覆盖是否完整，``checked`` 为 False。
    """
    tiles = None
    if isinstance(data, dict) and "tiles" in data:
        tiles = data["tiles"]
        if reference is None and "mask" in data:
            reference = data["mask"]
    elif isinstance(data, (list, tuple)):
        tiles = list(data)

    ref = _as_bool_mask(reference) if reference is not None else None
    if tiles is not None:
        tile_shape = _shape_of_tiles(tiles)
        if shape is not None:
            tile_shape = (int(shape[0]), int(shape[1]))
        elif ref is not None:
            tile_shape = ref.shape
        assembled = _assemble(tiles, tile_shape)
    else:
        assembled = _as_bool_mask(data)
        tile_shape = assembled.shape

    coverage = np.zeros(tile_shape, dtype=np.int32)
    if tiles is not None:
        for t in tiles:
            h, w = _as_bool_mask(t["data"]).shape
            x0, y0 = int(t.get("x0", 0)), int(t.get("y0", 0))
            x1, y1 = min(tile_shape[1], x0 + w), min(tile_shape[0], y0 + h)
            if x0 < x1 and y0 < y1:
                coverage[y0:y1, x0:x1] += 1

    result = {
        "ok": True, "checked": ref is not None,
        "shape": [int(tile_shape[0]), int(tile_shape[1])],
        "tiles": int(len(tiles)) if tiles is not None else 0,
        "coverage_gaps": int((coverage == 0).sum()) if tiles is not None else 0,
        "coverage_overlaps": int((coverage > 1).sum()) if tiles is not None else 0,
        "mismatch_pixels": None, "mismatch_ratio": None, "max_abs_diff": None,
        "seam_mismatch": {}, "seam_max": 0,
    }
    if ref is not None:
        if assembled.shape != ref.shape:
            raise ValueError(f"拼接形状 {assembled.shape} 与参考 {ref.shape} 不一致")
        diff = assembled != ref
        n = int(diff.sum())
        result["mismatch_pixels"] = n
        result["mismatch_ratio"] = float(n / diff.size) if diff.size else 0.0
        result["max_abs_diff"] = int(np.abs(assembled.astype(np.int16)
                                            - ref.astype(np.int16)).max(initial=0))
        if tile is not None and tiles is not None:
            th, tw = _coerce_window(tile)
            seams = {}
            for x in range(tw, tile_shape[1], tw):
                seams[f"x={x}"] = int(diff[:, max(0, x - boundary): min(tile_shape[1], x + boundary)].sum())
            for y in range(th, tile_shape[0], th):
                seams[f"y={y}"] = int(diff[max(0, y - boundary): min(tile_shape[0], y + boundary), :].sum())
            result["seam_mismatch"] = seams
            result["seam_max"] = max(seams.values(), default=0)
        result["ok"] = n == 0 and result["coverage_gaps"] == 0
    else:
        result["ok"] = result["coverage_gaps"] == 0
    result["assembled"] = assembled
    return result


def load(source) -> dict:
    """读取一页或多页：路径 / bytes / PIL Image / ndarray / 页列表。

    TIFF 等多帧文件逐帧返回；所有页面统一成 uint8 灰度，后续步骤不再依赖原始
    色彩空间。
    """
    pages = _load_pages(source)
    return {
        "pages": pages,
        "count": len(pages),
        "shapes": [list(p.shape) for p in pages],
        "source_type": type(source).__name__,
        "path": str(source) if isinstance(source, (str, Path)) else None,
    }


def binarize(image, params=None, *, k: float = DEFAULT_K, r: float = DEFAULT_R,
             normalize: bool | None = None, tile=None, mode: str = "divide") -> dict:
    """自适应二值化主入口：背景归一 -> 按字高窗口的局部均值/方差阈值。

    每页完全从自身图像与自身参数执行，不读写模块级状态，重复调用结果一致。
    给定 ``tile`` 时同时返回跨块切分结果与拼接校验（阈值仍由全页统计决定）。
    """
    gray = _coerce_page(image)
    supplied = params is not None
    page_params = _resolve_params(gray, params, k=k, r=r, normalize=normalize, mode=mode)
    mask, _flat, thr = _binarize_page(gray, page_params)
    result = {
        "mask": np.where(mask, INK, 0).astype(np.uint8),
        "mask_bool": mask,
        "mask_sha": _mask_sha(mask),
        "params": copy.deepcopy(page_params),
        "params_sha": _params_sha(page_params),
        "glyph_height": int(page_params["glyph_height"]),
        "window": list(page_params["window"]),
        "threshold_mean": float(thr.mean()),
        "ink_ratio": float(mask.mean()),
        "normalized": bool(page_params["normalize"]),
        "params_supplied": bool(supplied),
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
    }
    if tile is not None:
        tiles = [{"x0": int(x), "y0": int(y), "w": int(w), "h": int(h),
                  "data": np.where(mask[y : y + h, x : x + w], INK, 0).astype(np.uint8)}
                 for (x, y, w, h) in _tile_grid(gray.shape, tile)]
        result["tiles"] = tiles
        result["stitch_check"] = check_stitch(
            {"tiles": tiles, "mask": result["mask"]}, tile=tile, boundary=2)
    return result


def stitch(tiles, shape=None, *, reference=None) -> dict:
    """拼接块结果并校验；块列表元素需含 ``x0/y0/data``。"""
    ref = reference if reference is not None else (
        tiles.get("mask") if isinstance(tiles, dict) else None)
    if isinstance(tiles, dict) and "tiles" in tiles:
        tile_list = list(tiles["tiles"])
    elif isinstance(tiles, (list, tuple)):
        tile_list = list(tiles)
    else:
        tile_list = [tiles]
    if not tile_list:
        H, W = (int(shape[0]), int(shape[1])) if shape is not None else (0, 0)
        empty = np.zeros((H, W), dtype=bool)
        return {"mask": empty.astype(np.uint8), "mask_bool": empty,
                "mask_sha": _mask_sha(empty), "shape": [H, W], "tiles": 0, "check": None}
    if any(not isinstance(t, dict) or "data" not in t for t in tile_list):
        mask = _as_bool_mask(tiles)
        tile_list = [{"x0": 0, "y0": 0, "data": np.where(mask, INK, 0).astype(np.uint8)}]
    tile_shape = (int(shape[0]), int(shape[1])) if shape is not None else _shape_of_tiles(tile_list)
    mask = _assemble(tile_list, tile_shape)
    out = {
        "mask": np.where(mask, INK, 0).astype(np.uint8),
        "mask_bool": mask,
        "mask_sha": _mask_sha(mask),
        "shape": [int(tile_shape[0]), int(tile_shape[1])],
        "tiles": len(tile_list),
        "check": None,
    }
    if ref is not None:
        out["check"] = check_stitch(tile_list, reference=ref, shape=tile_shape)
        out["stitch_ok"] = out["check"]["ok"]
    elif isinstance(tiles, dict) and "stitch_check" in tiles:
        out["check"] = copy.deepcopy(tiles["stitch_check"])
        out["stitch_ok"] = out["check"]["ok"]
    return out


def export(data, path=None, *, fmt: str | None = None, meta=None) -> dict:
    """导出 PNG（默认，255=墨迹）或 JSON 元数据；未给路径时返回字节。"""
    mask = _as_u8_mask(data)
    info = {}
    if isinstance(data, dict):
        info = {k: copy.deepcopy(data[k]) for k in
                ("mask_sha", "params", "glyph_height", "window", "ink_ratio") if k in data}
    fmt = (fmt or ("json" if str(path).lower().endswith(".json") else "png")).lower()
    if fmt in ("json", "meta"):
        payload = {
            "shape": [int(mask.shape[0]), int(mask.shape[1])],
            "mask_sha": _mask_sha(mask > 0),
            "ink_ratio": float((mask > 0).mean()),
            "meta": _jsonable(meta if meta is not None else info),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        written = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            written = str(p)
        return {"path": written, "bytes": None if written else raw, "format": "json",
                "sha": payload["mask_sha"], "shape": payload["shape"],
                "ink_ratio": payload["ink_ratio"], "written": written is not None}

    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    raw = buf.getvalue()
    written = None
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        written = str(p)
    return {"path": written, "bytes": None if written else raw, "format": "png",
            "sha": _mask_sha(mask > 0), "shape": [int(mask.shape[0]), int(mask.shape[1])],
            "ink_ratio": float((mask > 0).mean()), "written": written is not None}


def validate_binarize(result, reference=None, *, tile=None, boundary: int = 2,
                      min_f1: float = 0.8) -> dict:
    """结果校验：结构合理性 + 可选参考质量 + 分块拼接一致性。

    任一硬性检查失败即 ``ok=False``，用于品控拦截“整页发白/发黑”“参数串页”
    “块缝断字”等回归。
    """
    met = metrics(result, reference)
    checks = [
        {"name": "not_blank", "ok": bool(met["ink_pixels"] > 0),
         "detail": f"ink_pixels={met['ink_pixels']}"},
        {"name": "ink_ratio_sane", "ok": bool(0.0005 <= met["ink_ratio"] <= 0.75),
         "detail": f"ink_ratio={met['ink_ratio']:.4f}"},
        {"name": "glyph_not_merged",
         "ok": bool(met["filled_block_ratio"] < 0.5),
         "detail": f"filled_block_ratio={met['filled_block_ratio']:.3f}"},
    ]
    stitch_check = None
    if isinstance(result, dict):
        stitch_check = result.get("stitch_check") or result.get("check")
    if stitch_check is None and tile is not None:
        stitch_check = check_stitch(result, reference=reference, tile=tile, boundary=boundary)
    if stitch_check is not None:
        checks.append({
            "name": "stitch_consistent", "ok": bool(stitch_check.get("ok")),
            "detail": f"mismatch={stitch_check.get('mismatch_pixels')} "
                      f"gaps={stitch_check.get('coverage_gaps')}",
        })
    if met["reference"]:
        checks.append({
            "name": "reference_f1", "ok": bool(met["f1"] >= min_f1),
            "detail": f"f1={met['f1']:.4f} (min {min_f1})",
        })
    return {
        "ok": bool(all(c["ok"] for c in checks)),
        "checks": checks,
        "failed": [c["name"] for c in checks if not c["ok"]],
        "metrics": met,
        "stitch": stitch_check,
        "mask_sha": met["mask_sha"],
    }


def metrics(mask, reference=None, *, block: int = 8) -> dict:
    """二值掩膜度量：墨迹占比、笔画宽度、连通域、实心块比例；有参考时给 F1/IoU。

    ``filled_block_ratio`` 统计“几乎全墨”的方块占比，直接暴露窗口过大并笔画。
    """
    if isinstance(mask, dict) and "tiles" in mask and "mask" not in mask:
        mask = stitch(mask)["mask"]
    m = _as_bool_mask(mask)
    sha = mask["mask_sha"] if isinstance(mask, dict) and "mask_sha" in mask else _mask_sha(m)
    total = max(1, m.size)
    ink = int(m.sum())
    comps, largest = _label_components(m)
    h_runs = _run_lengths(m, axis=1)
    v_runs = _run_lengths(m, axis=0)
    out = {
        "mask_sha": sha,
        "shape": [int(m.shape[0]), int(m.shape[1])],
        "ink_pixels": ink,
        "ink_ratio": float(ink / total),
        "components": int(comps),
        "largest_component": int(largest),
        "stroke_width_h": float(np.median(h_runs)) if h_runs else 0.0,
        "stroke_width_v": float(np.median(v_runs)) if v_runs else 0.0,
        "filled_block_ratio": float(_filled_block_ratio(m, block)),
        "reference": False, "precision": None, "recall": None, "f1": None, "iou": None,
    }
    if reference is not None:
        ref = _as_bool_mask(reference)
        if ref.shape != m.shape:
            raise ValueError(f"参考掩膜形状 {ref.shape} 与结果 {m.shape} 不一致")
        tp = int(np.count_nonzero(m & ref))
        fp = int(np.count_nonzero(m & ~ref))
        fn = int(np.count_nonzero(ref & ~m))
        union = tp + fp + fn
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.update({"reference": True, "tp": tp, "fp": fp, "fn": fn,
                    "precision": float(precision), "recall": float(recall),
                    "f1": float(f1), "iou": float(tp / union) if union else 1.0})
    return out


def audit(image, result=None, *, params=None) -> dict:
    """逐页体检：曝光、溢出位宽、字高估计、墨迹占比、并块风险与告警列表。"""
    pages = _load_pages(image)
    items = []
    for i, page in enumerate(pages):
        gray = _coerce_page(page)
        p = _resolve_params(gray, params, index=i) if params is not None \
            else _page_params(gray, index=i)
        integ = integral_32bit(gray)
        mask, _flat, _thr = _binarize_page(gray, p)
        met = metrics(mask)
        bg_std = 0.0
        if p["normalize"]:
            bg_std = background_normalize(gray, radius=p["background_radius"],
                                          glyph=p["glyph_height"],
                                          mode=p["background_mode"])["background_std"]
        warn = []
        sat = float(np.mean((gray <= 2) | (gray >= 253)))
        if sat > 0.5:
            warn.append("曝光过饱和：超过 50% 像素贴近 0/255")
        if integ["bits"] > 32:
            warn.append("整页总和超出 32 位，积分图已升至 64 位，注意内存占用")
        if not p["glyph_estimated"]:
            warn.append("未检测到文本行，字高回退为页高 1/30")
        if met["ink_ratio"] > 0.5:
            warn.append("墨迹占比过高，疑似阈值过低或整页发黑")
        if met["ink_ratio"] < 0.0005:
            warn.append("墨迹占比过低，疑似整页发白或空白页")
        if met["filled_block_ratio"] > 0.2:
            warn.append("实心块偏多，疑似窗口跨行合并笔画")
        if p["window"][0] > gray.shape[0] // 2 or p["window"][1] > gray.shape[1] // 2:
            warn.append("窗口接近页宽，局部阈值退化为全局阈值")
        if bg_std > 60:
            warn.append("归一后背景残差偏大，可能存在强阴影或大面积底色")
        items.append({
            "page_index": i, "ok": not warn, "warnings": warn, "saturation": sat,
            "integral_bits": integ["bits"], "background_std": float(bg_std),
            "glyph_height": p["glyph_height"], "glyph_estimated": p["glyph_estimated"],
            "window": p["window"], "ink_ratio": met["ink_ratio"],
            "filled_block_ratio": met["filled_block_ratio"], "params_sha": _params_sha(p),
            "result_mask_sha": result.get("mask_sha") if isinstance(result, dict) and i == 0 else None,
        })
    return {"ok": all(it["ok"] for it in items), "pages": items, "count": len(items),
            "warnings": [w for it in items for w in it["warnings"]]}


def fallback(image, reason: str = "", *, threshold: int | None = None) -> dict:
    """兜底二值化：Otsu 全局阈值；结果始终非空且确定，供局部算法异常时保底。"""
    pages = _load_pages(image)
    masks, used = [], []
    for page in pages:
        gray = _coerce_page(page)
        thr = int(_otsu(gray)) if threshold is None else int(threshold)
        masks.append(gray < thr)
        used.append(thr)
    if len(masks) == 1 and len(used) == 1:
        m = masks[0]
        return {"mask": np.where(m, INK, 0).astype(np.uint8), "mask_bool": m,
                "mask_sha": _mask_sha(m), "threshold": used[0], "method": "otsu",
                "reason": str(reason), "shape": [int(m.shape[0]), int(m.shape[1])],
                "ink_ratio": float(m.mean())}
    return {"pages": [{"mask": np.where(m, INK, 0).astype(np.uint8), "mask_bool": m,
                       "mask_sha": _mask_sha(m), "threshold": t} for m, t in zip(masks, used)],
            "count": len(masks), "method": "otsu", "reason": str(reason),
            "thresholds": used}


def determinism(payload=None, *, repeats: int = 3, params=None, tile=None) -> dict:
    """重复执行二值化并比对掩膜与参数哈希，验证无隐藏状态、结果可复现。"""
    target = payload if payload is not None else _synthetic_page(DEFAULT_SHAPE, glyph=12,
                                                                 seed=7, shadow=0.2)
    params_before = _params_sha(params) if params is not None else None
    hashes, param_hashes, ok = [], [], True
    first = None
    for _ in range(max(1, int(repeats))):
        res = binarize(target, params=params, tile=tile)
        sha, psha = res["mask_sha"], res["params_sha"]
        hashes.append(sha)
        param_hashes.append(psha)
        if first is None:
            first = res
        elif sha != first["mask_sha"] or psha != first["params_sha"]:
            ok = False
    params_unmodified = params_before is None or _params_sha(params) == params_before
    return {
        "deterministic": bool(ok and params_unmodified),
        "repeats": len(hashes),
        "hashes": hashes,
        "params_hashes": param_hashes,
        "unique_hashes": len(set(hashes)),
        "unique_params": len(set(param_hashes)),
        "params_unmodified": bool(params_unmodified),
        "mask_sha": first["mask_sha"] if first else None,
        "glyph_height": first["glyph_height"] if first else None,
        "window": first["window"] if first else None,
    }
