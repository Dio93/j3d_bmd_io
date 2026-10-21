"""Low-level helpers shared by the J3D reader and writer (no Blender dependency)."""

import struct

PAD_TEXT = b"This is padding data to alignment."


class J3DError(Exception):
    """Raised for malformed or unsupported J3D data."""


def u8(d, o):
    return d[o]


def u16(d, o):
    return struct.unpack_from(">H", d, o)[0]


def s16(d, o):
    return struct.unpack_from(">h", d, o)[0]


def u32(d, o):
    return struct.unpack_from(">I", d, o)[0]


def f32(d, o):
    return struct.unpack_from(">f", d, o)[0]


def align(n, a=32):
    return (n + a - 1) // a * a


def padding(length):
    """Nintendo-style padding text of exactly ``length`` bytes."""
    if length <= 0:
        return b""
    return (PAD_TEXT * (length // len(PAD_TEXT) + 1))[:length]


def pad_to(buf, a=32):
    """Pad a bytearray to a multiple of ``a`` bytes."""
    buf += padding(align(len(buf), a) - len(buf))
    return buf


def j3d_hash(raw):
    h = 0
    for c in raw:
        h = (h * 3 + c) & 0xFFFF
    return h


def _decode_name(raw):
    for enc in ("shift_jis", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def _encode_name(name):
    try:
        return name.encode("shift_jis")
    except UnicodeEncodeError:
        return name.encode("ascii", "replace")


def read_string_table(d, off):
    """Read a J3D name table located at absolute offset ``off``."""
    n = u16(d, off)
    names = []
    for i in range(n):
        _h, so = struct.unpack_from(">HH", d, off + 4 + 4 * i)
        s = off + so
        e = d.index(b"\0", s)
        names.append(_decode_name(bytes(d[s:e])))
    return names


def write_string_table(names):
    """Serialise a J3D name table (hash + offset entries followed by strings)."""
    head = bytearray(struct.pack(">HH", len(names), 0xFFFF))
    strings = bytearray()
    base = 4 + 4 * len(names)
    for nm in names:
        raw = _encode_name(nm)
        head += struct.pack(">HH", j3d_hash(raw), base + len(strings))
        strings += raw + b"\0"
    return bytes(head + strings)


def yaz0_decompress(data):
    """Decompress a Yaz0 stream (used by .szs archives; handy for raw BMD in SZS)."""
    if data[:4] != b"Yaz0":
        raise J3DError("Not a Yaz0 stream")
    size = u32(data, 4)
    out = bytearray()
    src = 16
    while len(out) < size:
        code = data[src]
        src += 1
        for bit in range(8):
            if len(out) >= size:
                break
            if code & (0x80 >> bit):
                out.append(data[src])
                src += 1
            else:
                b1, b2 = data[src], data[src + 1]
                src += 2
                dist = ((b1 & 0x0F) << 8) | b2
                n = b1 >> 4
                if n == 0:
                    n = data[src] + 0x12
                    src += 1
                else:
                    n += 2
                start = len(out) - (dist + 1)
                for i in range(n):
                    out.append(out[start + i])
    return bytes(out)
