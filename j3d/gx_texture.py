"""GameCube/Wii GX texture formats: decoding and encoding with numpy.

All images are handled as (height, width, 4) uint8 RGBA arrays, row 0 = top.
"""

import numpy as np

from .common import J3DError

I4, I8, IA4, IA8, RGB565, RGB5A3, RGBA32 = 0, 1, 2, 3, 4, 5, 6
C4, C8, C14X2 = 8, 9, 10
CMPR = 14

FORMAT_NAMES = {
    I4: "I4", I8: "I8", IA4: "IA4", IA8: "IA8", RGB565: "RGB565",
    RGB5A3: "RGB5A3", RGBA32: "RGBA32", C4: "C4", C8: "C8",
    C14X2: "C14X2", CMPR: "CMPR",
}
NAME_TO_FORMAT = {v: k for k, v in FORMAT_NAMES.items()}

# format -> (block width, block height, bits per pixel)
BLOCK = {
    I4: (8, 8, 4), I8: (8, 4, 8), IA4: (8, 4, 8), IA8: (4, 4, 16),
    RGB565: (4, 4, 16), RGB5A3: (4, 4, 16), RGBA32: (4, 4, 32),
    C4: (8, 8, 4), C8: (8, 4, 8), C14X2: (4, 4, 16), CMPR: (8, 8, 4),
}

PAL_IA8, PAL_RGB565, PAL_RGB5A3 = 0, 1, 2


def level_size(fmt, w, h):
    bw, bh, bpp = BLOCK[fmt]
    bx = -(-w // bw)
    by = -(-h // bh)
    return bx * by * bw * bh * bpp // 8


def total_size(fmt, w, h, mips):
    size = 0
    for _ in range(max(1, mips)):
        size += level_size(fmt, w, h)
        w = max(1, w // 2)
        h = max(1, h // 2)
    return size


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _untile(a, by, bx):
    """(by, bx, bh, bw, ...) -> (by*bh, bx*bw, ...)"""
    a = np.moveaxis(a, 2, 1)  # (by, bh, bx, bw, ...)
    s = a.shape
    return a.reshape((s[0] * s[1], s[2] * s[3]) + s[4:])


def _tile(img, bh, bw):
    """(H, W, ...) -> (by, bx, bh, bw, ...) with H % bh == W % bw == 0."""
    H, W = img.shape[:2]
    by, bx = H // bh, W // bw
    a = img.reshape((by, bh, bx, bw) + img.shape[2:])
    return np.moveaxis(a, 1, 2)


def _expand5(v):
    return ((v << 3) | (v >> 2)).astype(np.uint8)


def _expand6(v):
    return ((v << 2) | (v >> 4)).astype(np.uint8)


def _rgb565_to_rgba(v):
    v = v.astype(np.uint16)
    r = _expand5((v >> 11) & 31)
    g = _expand6((v >> 5) & 63)
    b = _expand5(v & 31)
    a = np.full(v.shape, 255, np.uint8)
    return np.stack([r, g, b, a], -1)


def _rgb5a3_to_rgba(v):
    v = v.astype(np.uint16)
    opaque = (v & 0x8000) != 0
    r5 = _expand5((v >> 10) & 31)
    g5 = _expand5((v >> 5) & 31)
    b5 = _expand5(v & 31)
    a3 = ((v >> 12) & 7).astype(np.uint16)
    a = ((a3 << 5) | (a3 << 2) | (a3 >> 1)).astype(np.uint8)
    r4 = (((v >> 8) & 15) * 17).astype(np.uint8)
    g4 = (((v >> 4) & 15) * 17).astype(np.uint8)
    b4 = ((v & 15) * 17).astype(np.uint8)
    r = np.where(opaque, r5, r4)
    g = np.where(opaque, g5, g4)
    b = np.where(opaque, b5, b4)
    a = np.where(opaque, 255, a).astype(np.uint8)
    return np.stack([r, g, b, a], -1)


def decode_palette(pal_fmt, buf, count):
    raw = np.frombuffer(buf, ">u2", count=count)
    if pal_fmt == PAL_IA8:
        a = (raw >> 8).astype(np.uint8)
        i = (raw & 255).astype(np.uint8)
        return np.stack([i, i, i, a], -1)
    if pal_fmt == PAL_RGB565:
        return _rgb565_to_rgba(raw)
    if pal_fmt == PAL_RGB5A3:
        return _rgb5a3_to_rgba(raw)
    raise J3DError(f"Unsupported palette format {pal_fmt}")


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

def decode(fmt, buf, w, h, palette=None):
    """Decode one mip level to an (h, w, 4) uint8 RGBA array."""
    if fmt not in BLOCK:
        raise J3DError(f"Unsupported texture format {fmt}")
    bw, bh, _ = BLOCK[fmt]
    bx, by = -(-w // bw), -(-h // bh)
    size = level_size(fmt, w, h)
    raw = np.frombuffer(buf, np.uint8, count=size)

    if fmt == I4 or fmt == C4:
        b = raw.reshape(by, bx, bh, bw // 2)
        px = np.empty((by, bx, bh, bw), np.uint8)
        px[..., 0::2] = b >> 4
        px[..., 1::2] = b & 15
        px = _untile(px, by, bx)
        if fmt == I4:
            i = px * 17
            out = np.stack([i, i, i, i], -1)
        else:
            out = palette[px]
    elif fmt == I8 or fmt == C8:
        px = _untile(raw.reshape(by, bx, bh, bw), by, bx)
        if fmt == I8:
            out = np.stack([px, px, px, px], -1)
        else:
            out = palette[px]
    elif fmt == IA4:
        px = _untile(raw.reshape(by, bx, bh, bw), by, bx)
        a = (px >> 4) * 17
        i = (px & 15) * 17
        out = np.stack([i, i, i, a], -1)
    elif fmt == IA8:
        px = _untile(raw.reshape(by, bx, bh, bw, 2), by, bx)
        a, i = px[..., 0], px[..., 1]
        out = np.stack([i, i, i, a], -1)
    elif fmt == RGB565:
        v = np.frombuffer(buf, ">u2", count=size // 2).reshape(by, bx, bh, bw)
        out = _rgb565_to_rgba(_untile(v, by, bx))
    elif fmt == RGB5A3:
        v = np.frombuffer(buf, ">u2", count=size // 2).reshape(by, bx, bh, bw)
        out = _rgb5a3_to_rgba(_untile(v, by, bx))
    elif fmt == RGBA32:
        blk = raw.reshape(by, bx, 2, bh, bw, 2)
        ar = _untile(blk[:, :, 0], by, bx)
        gb = _untile(blk[:, :, 1], by, bx)
        out = np.stack([ar[..., 1], gb[..., 0], gb[..., 1], ar[..., 0]], -1)
    elif fmt == C14X2:
        v = np.frombuffer(buf, ">u2", count=size // 2).reshape(by, bx, bh, bw)
        idx = _untile(v, by, bx) & 0x3FFF
        out = palette[idx]
    elif fmt == CMPR:
        sub = raw.reshape(by, bx, 2, 2, 8)
        c0 = (sub[..., 0].astype(np.uint16) << 8) | sub[..., 1]
        c1 = (sub[..., 2].astype(np.uint16) << 8) | sub[..., 3]
        p0 = _rgb565_to_rgba(c0).astype(np.int32)
        p1 = _rgb565_to_rgba(c1).astype(np.int32)
        four = (c0 > c1)[..., None]
        p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
        p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
        p2[..., 3] = 255
        p3[..., 3] = np.where(four[..., 0], 255, 0)
        pal = np.stack([p0, p1, p2, p3], -2)  # (by,bx,2,2,4,4)
        rows = sub[..., 4:8]  # (by,bx,2,2,4)
        shifts = np.array([6, 4, 2, 0], np.uint8)
        idx = (rows[..., None] >> shifts) & 3  # (by,bx,2,2,4rows,4cols)
        out = np.take_along_axis(
            pal, idx.reshape(by, bx, 2, 2, 16, 1), axis=4
        ).reshape(by, bx, 2, 2, 4, 4, 4)
        # (by,bx,sy,sx,r,c,ch) -> (by,sy,r,bx,sx,c,ch)
        out = out.transpose(0, 2, 4, 1, 3, 5, 6).reshape(by * 8, bx * 8, 4)
        out = out.astype(np.uint8)
    else:  # pragma: no cover
        raise J3DError(f"Unsupported texture format {fmt}")

    return np.ascontiguousarray(out[:h, :w].astype(np.uint8))


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def _pad_image(rgba, bw, bh):
    h, w = rgba.shape[:2]
    ph, pw = -h % bh, -w % bw
    if ph or pw:
        rgba = np.pad(rgba, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return rgba


def _intensity(rgba):
    r = rgba[..., 0].astype(np.float32)
    g = rgba[..., 1].astype(np.float32)
    b = rgba[..., 2].astype(np.float32)
    return np.clip(np.rint(0.299 * r + 0.587 * g + 0.114 * b), 0, 255).astype(np.uint8)


def _q5(v):
    return ((v.astype(np.int32) * 31 + 127) // 255).astype(np.uint16)


def _q6(v):
    return ((v.astype(np.int32) * 63 + 127) // 255).astype(np.uint16)


def _q4(v):
    return ((v.astype(np.int32) * 15 + 127) // 255).astype(np.uint16)


def _q3(v):
    return ((v.astype(np.int32) * 7 + 127) // 255).astype(np.uint16)


def _pack565(rgb):
    return (_q5(rgb[..., 0]) << 11) | (_q6(rgb[..., 1]) << 5) | _q5(rgb[..., 2])


def _pack_rgb5a3(rgba):
    r, g, b, a = (rgba[..., i] for i in range(4))
    opaque = a >= 0xE0
    v_op = 0x8000 | (_q5(r) << 10) | (_q5(g) << 5) | _q5(b)
    v_tr = (_q3(a) << 12) | (_q4(r) << 8) | (_q4(g) << 4) | _q4(b)
    return np.where(opaque, v_op, v_tr).astype(np.uint16)


def _encode_cmpr(rgba):
    rgba = _pad_image(rgba, 8, 8)
    H, W = rgba.shape[:2]
    sub = rgba.reshape(H // 4, 4, W // 4, 4, 4).transpose(0, 2, 1, 3, 4)
    blocks = sub.reshape(-1, 16, 4).astype(np.int32)
    n = blocks.shape[0]
    rgb = blocks[..., :3]
    opaque = blocks[..., 3] >= 128
    has_trans = ~opaque.all(1)

    big = np.where(opaque[..., None], rgb, 255).min(1)
    small = np.where(opaque[..., None], rgb, 0).max(1)
    none_opaque = ~opaque.any(1)
    big[none_opaque] = 0
    small[none_opaque] = 0
    axis = small - big
    proj = ((rgb - big[:, None]) * axis[:, None]).sum(-1).astype(np.float64)
    imin = np.where(opaque, proj, np.inf).argmin(1)
    imax = np.where(opaque, proj, -np.inf).argmax(1)
    ar = np.arange(n)
    p_lo = rgb[ar, imin]
    p_hi = rgb[ar, imax]
    p_lo[none_opaque] = 0
    p_hi[none_opaque] = 0

    q_hi = _pack565(p_hi.astype(np.uint8)).astype(np.int32)
    q_lo = _pack565(p_lo.astype(np.uint8)).astype(np.int32)
    big_c = np.maximum(q_hi, q_lo)
    small_c = np.minimum(q_hi, q_lo)
    c0 = np.where(has_trans, small_c, big_c)
    c1 = np.where(has_trans, big_c, small_c)

    def expand(c):
        r = ((c >> 11) & 31)
        g = ((c >> 5) & 63)
        b = (c & 31)
        return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)

    e0, e1 = expand(c0), expand(c1)
    four = (c0 > c1)[:, None]
    e2 = np.where(four, (2 * e0 + e1) // 3, (e0 + e1) // 2)
    e3 = np.where(four, (e0 + 2 * e1) // 3, 0)
    pal = np.stack([e0, e1, e2, e3], 1)  # (n,4,3)
    dist = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)  # (n,16,4)
    tri = ~four[:, 0]
    dist[..., 3] = np.where(tri[:, None], np.iinfo(np.int32).max // 2, dist[..., 3])
    idx = dist.argmin(2)
    idx = np.where(~opaque, 3, idx)
    # in 3-colour mode transparent pixels use 3; in 4-colour mode there are none
    idx = idx.reshape(n, 4, 4).astype(np.uint8)
    row_bytes = (idx[..., 0] << 6) | (idx[..., 1] << 4) | (idx[..., 2] << 2) | idx[..., 3]
    out = np.empty((n, 8), np.uint8)
    out[:, 0] = (c0 >> 8) & 255
    out[:, 1] = c0 & 255
    out[:, 2] = (c1 >> 8) & 255
    out[:, 3] = c1 & 255
    out[:, 4:8] = row_bytes
    out = out.reshape(H // 4, W // 4, 8)
    # group 2x2 sub-blocks into 8x8 blocks
    out = out.reshape(H // 8, 2, W // 8, 2, 8).transpose(0, 2, 1, 3, 4)
    return out.tobytes()


def encode(fmt, rgba):
    """Encode one mip level (h, w, 4 uint8) to GX bytes."""
    if fmt not in BLOCK or fmt in (C4, C8, C14X2):
        raise J3DError(f"Cannot encode texture format {FORMAT_NAMES.get(fmt, fmt)}")
    if fmt == CMPR:
        return _encode_cmpr(rgba)
    bw, bh, _ = BLOCK[fmt]
    img = _pad_image(rgba, bw, bh)
    H, W = img.shape[:2]

    if fmt == I4:
        t = _tile(_intensity(img), bh, bw)
        q = ((t.astype(np.uint16) + 8) // 17).astype(np.uint8)
        q = np.minimum(q, 15)
        return ((q[..., 0::2] << 4) | q[..., 1::2]).astype(np.uint8).tobytes()
    if fmt == I8:
        return np.ascontiguousarray(_tile(_intensity(img), bh, bw)).tobytes()
    if fmt == IA4:
        i4 = np.minimum((_intensity(img).astype(np.uint16) + 8) // 17, 15)
        a4 = np.minimum((img[..., 3].astype(np.uint16) + 8) // 17, 15)
        v = ((a4 << 4) | i4).astype(np.uint8)
        return np.ascontiguousarray(_tile(v, bh, bw)).tobytes()
    if fmt == IA8:
        v = np.stack([img[..., 3], _intensity(img)], -1)
        return np.ascontiguousarray(_tile(v, bh, bw)).tobytes()
    if fmt == RGB565:
        v = _pack565(img[..., :3]).astype(">u2")
        return np.ascontiguousarray(_tile(v, bh, bw)).tobytes()
    if fmt == RGB5A3:
        v = _pack_rgb5a3(img).astype(">u2")
        return np.ascontiguousarray(_tile(v, bh, bw)).tobytes()
    if fmt == RGBA32:
        ar = np.stack([img[..., 3], img[..., 0]], -1)
        gb = np.stack([img[..., 1], img[..., 2]], -1)
        ta = _tile(ar, bh, bw)
        tg = _tile(gb, bh, bw)
        both = np.stack([ta, tg], 2)  # (by,bx,2,4,4,2)
        return np.ascontiguousarray(both).tobytes()
    raise J3DError("unreachable")  # pragma: no cover


def downsample(rgba):
    """2x2 box filter to the next mip level."""
    h, w = rgba.shape[:2]
    nh, nw = max(1, h // 2), max(1, w // 2)
    a = rgba.astype(np.float32)
    if h > 1 and w > 1:
        a = a[: nh * 2, : nw * 2]
        a = a.reshape(nh, 2, nw, 2, 4).mean(axis=(1, 3))
    elif h > 1:
        a = a[: nh * 2].reshape(nh, 2, w, 4).mean(axis=1)
    elif w > 1:
        a = a[:, : nw * 2].reshape(h, nw, 2, 4).mean(axis=2)
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def max_mip_count(w, h):
    n = 1
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        n += 1
    return n


def encode_with_mips(fmt, rgba, mip_count):
    mip_count = max(1, min(mip_count, max_mip_count(rgba.shape[1], rgba.shape[0])))
    out = bytearray()
    cur = rgba
    for _ in range(mip_count):
        out += encode(fmt, cur)
        cur = downsample(cur)
    return bytes(out), mip_count


def has_alpha(rgba, threshold=255):
    return bool((rgba[..., 3] < threshold).any())


def has_partial_alpha(rgba):
    a = rgba[..., 3]
    return bool(((a > 0) & (a < 255)).any())
