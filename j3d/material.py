"""MAT3 chunk (materials) reading and writing.

A material is represented as a plain, JSON-serialisable ``dict`` with all table
indices resolved to values.  ``None`` stands for "index 0xFFFF" (unused slot).
Texture references are stored by *name* (``texNames``) so that textures can be
re-ordered / replaced when exporting.
"""

import copy
import struct

from .common import J3DError, align, pad_to, read_string_table, write_string_table, u16, u32

# --- table layout -----------------------------------------------------------
# slot in the MAT3 header offset list -> (name, element size)
TABLE_SLOTS = [
    ("init", 0x14C), ("remap", 2), ("names", 0), ("indirect", 0x138),
    ("cull", 4), ("matColor", 4), ("chanCount", 1), ("colorChan", 8),
    ("ambColor", 4), ("light", 0x34), ("texGenCount", 1), ("texCoord", 4),
    ("texCoord2", 4), ("texMtx", 0x64), ("postTexMtx", 0x64), ("texNo", 2),
    ("tevOrder", 4), ("tevColor", 8), ("tevKColor", 4), ("tevStageCount", 1),
    ("tevStage", 0x14), ("tevSwapMode", 4), ("tevSwapTable", 4), ("fog", 0x2C),
    ("alphaComp", 8), ("blend", 4), ("zMode", 4), ("zCompLoc", 1),
    ("dither", 1), ("nbtScale", 0x10),
]
SLOT_INDEX = {n: i for i, (n, _) in enumerate(TABLE_SLOTS)}
ELEM_SIZE = {n: s for n, s in TABLE_SLOTS}

# (field, offset in init entry, count, struct code, table)
INIT_FIELDS = [
    ("flags", 0x00, 1, "B", None),
    ("cull", 0x01, 1, "B", "cull"),
    ("colorChanCount", 0x02, 1, "B", "chanCount"),
    ("texGenCount", 0x03, 1, "B", "texGenCount"),
    ("tevStageCount", 0x04, 1, "B", "tevStageCount"),
    ("zCompLoc", 0x05, 1, "B", "zCompLoc"),
    ("zMode", 0x06, 1, "B", "zMode"),
    ("dither", 0x07, 1, "B", "dither"),
    ("matColor", 0x08, 2, "H", "matColor"),
    ("colorChan", 0x0C, 4, "H", "colorChan"),
    ("ambColor", 0x14, 2, "H", "ambColor"),
    ("light", 0x18, 8, "H", "light"),
    ("texCoord", 0x28, 8, "H", "texCoord"),
    ("texCoord2", 0x38, 8, "H", "texCoord2"),
    ("texMtx", 0x48, 10, "H", "texMtx"),
    ("postTexMtx", 0x5C, 20, "H", "postTexMtx"),
    ("texNo", 0x84, 8, "H", "texNo"),
    ("kColor", 0x94, 4, "H", "tevKColor"),
    ("kColorSel", 0x9C, 16, "B", None),
    ("kAlphaSel", 0xAC, 16, "B", None),
    ("tevOrder", 0xBC, 16, "H", "tevOrder"),
    ("tevColor", 0xDC, 4, "H", "tevColor"),
    ("tevStage", 0xE4, 16, "H", "tevStage"),
    ("tevSwapMode", 0x104, 16, "H", "tevSwapMode"),
    ("tevSwapTable", 0x124, 4, "H", "tevSwapTable"),
    ("unknown12", 0x12C, 12, "H", None),
    ("fog", 0x144, 1, "H", "fog"),
    ("alphaComp", 0x146, 1, "H", "alphaComp"),
    ("blend", 0x148, 1, "H", "blend"),
    ("nbtScale", 0x14A, 1, "H", "nbtScale"),
]
# index field name in dict for the single-valued ones that are stored as "table entry"
SCALAR_TABLE_FIELDS = {"cull", "colorChanCount", "texGenCount", "tevStageCount",
                       "zCompLoc", "zMode", "dither", "fog", "alphaComp", "blend", "nbtScale"}

# Canonical "indirect texturing disabled" block (0x138 bytes) as found in game files.
_DEFAULT_INDIRECT_HEX = None  # filled lazily by _default_indirect()


def default_indirect():
    b = bytearray()
    b += struct.pack(">4B", 0, 0, 0xFF, 0xFF)
    b += b"\xff" * 16
    for _ in range(3):
        b += struct.pack(">6fb3B", 0.5, 0, 0, 0, 0.5, 0, 1, 0xFF, 0xFF, 0xFF)
    b += b"\x00" * (0x138 - len(b))
    return bytes(b)


# --- (un)packers for individual table entries --------------------------------

def _un_list(fmt):
    n = struct.calcsize(">" + fmt)
    return (lambda b: list(struct.unpack(">" + fmt, b[:n]))), (lambda o: struct.pack(">" + fmt, *o))


def _un_int(fmt):
    return (lambda b: struct.unpack(">" + fmt, b)[0]), (lambda o: struct.pack(">" + fmt, o))


def _un_texmtx(b):
    return {
        "projection": b[0], "type": b[1],
        "center": list(struct.unpack_from(">3f", b, 4)),
        "scale": list(struct.unpack_from(">2f", b, 0x10)),
        "rotation": struct.unpack_from(">h", b, 0x18)[0],
        "translation": list(struct.unpack_from(">2f", b, 0x1C)),
        "matrix": list(struct.unpack_from(">16f", b, 0x24)),
    }


def _pk_texmtx(o):
    return (struct.pack(">BBH3f2fhH2f", o["projection"], o["type"], 0xFFFF, *o["center"],
                        *o["scale"], o["rotation"], 0xFFFF, *o["translation"])
            + struct.pack(">16f", *o["matrix"]))


def _un_stage(b):
    return {
        "u0": b[0], "colorIn": list(b[1:5]), "colorOp": b[5], "colorBias": b[6],
        "colorScale": b[7], "colorClamp": b[8], "colorReg": b[9],
        "alphaIn": list(b[10:14]), "alphaOp": b[14], "alphaBias": b[15],
        "alphaScale": b[16], "alphaClamp": b[17], "alphaReg": b[18], "u19": b[19],
    }


def _pk_stage(o):
    return bytes([o["u0"], *o["colorIn"], o["colorOp"], o["colorBias"], o["colorScale"],
                  o["colorClamp"], o["colorReg"], *o["alphaIn"], o["alphaOp"], o["alphaBias"],
                  o["alphaScale"], o["alphaClamp"], o["alphaReg"], o["u19"]])


def _un_fog(b):
    t, en, center = struct.unpack_from(">BBH", b, 0)
    start, end, near, far = struct.unpack_from(">4f", b, 4)
    return {"type": t, "enable": en, "center": center, "startZ": start, "endZ": end,
            "nearZ": near, "farZ": far, "color": list(b[0x14:0x18]),
            "adjTable": list(struct.unpack_from(">10H", b, 0x18))}


def _pk_fog(o):
    return (struct.pack(">BBH4f", o["type"], o["enable"], o["center"], o["startZ"], o["endZ"],
                        o["nearZ"], o["farZ"]) + bytes(o["color"])
            + struct.pack(">10H", *o["adjTable"]))


def _un_nbt(b):
    return {"enable": b[0], "scale": list(struct.unpack_from(">3f", b, 4))}


def _pk_nbt(o):
    return bytes([o["enable"], 0xFF, 0xFF, 0xFF]) + struct.pack(">3f", *o["scale"])


def _un_light(b):
    return b.hex()


def _pk_light(o):
    return bytes.fromhex(o)


_ENTRY = {
    "cull": _un_int("I"),
    "matColor": _un_list("4B"),
    "chanCount": _un_int("B"),
    "colorChan": _un_list("8B"),
    "ambColor": _un_list("4B"),
    "light": (_un_light, _pk_light),
    "texGenCount": _un_int("B"),
    "texCoord": _un_list("4B"),
    "texCoord2": _un_list("4B"),
    "texMtx": (_un_texmtx, _pk_texmtx),
    "postTexMtx": (_un_texmtx, _pk_texmtx),
    "texNo": _un_int("H"),
    "tevOrder": _un_list("4B"),
    "tevColor": _un_list("4h"),
    "tevKColor": _un_list("4B"),
    "tevStageCount": _un_int("B"),
    "tevStage": (_un_stage, _pk_stage),
    "tevSwapMode": _un_list("4B"),
    "tevSwapTable": _un_list("4B"),
    "fog": (_un_fog, _pk_fog),
    "alphaComp": _un_list("8B"),
    "blend": _un_list("4B"),
    "zMode": _un_list("4B"),
    "zCompLoc": _un_int("B"),
    "dither": _un_int("B"),
    "nbtScale": (_un_nbt, _pk_nbt),
}

_TABLE_OF_FIELD = {f: t for f, _o, _c, _s, t in INIT_FIELDS if t}


# --- reading ----------------------------------------------------------------

def parse_mat3(d, off, tex_names):
    """Parse a MAT3 chunk. Returns (materials, names).

    ``tex_names`` is the list of TEX1 texture names (used to resolve ``texNo``).
    """
    count = u16(d, off + 8)
    offs = [u32(d, off + 0x0C + 4 * i) for i in range(len(TABLE_SLOTS))]
    remap = [u16(d, off + offs[1] + 2 * i) for i in range(count)]
    names = read_string_table(d, off + offs[2]) if offs[2] else [f"material{i}" for i in range(count)]

    def entry_bytes(table, idx):
        base = offs[SLOT_INDEX[table]]
        size = ELEM_SIZE[table]
        o = off + base + size * idx
        return bytes(d[o:o + size])

    default_ind = default_indirect()
    materials = []
    for m in range(count):
        ie = off + offs[0] + 0x14C * remap[m]
        init = bytes(d[ie:ie + 0x14C])
        mat = {}
        for field, io, cnt, code, table in INIT_FIELDS:
            vals = list(struct.unpack_from(">" + code * cnt, init, io))
            if table is None:
                mat[field] = vals if cnt > 1 else vals[0]
                continue
            unpack = _ENTRY[table][0]
            out = []
            sentinel = 0xFFFF if code == "H" else 0xFF
            for v in vals:
                out.append(None if v == sentinel else unpack(entry_bytes(table, v)))
            mat[field] = out[0] if field in SCALAR_TABLE_FIELDS else out
        # resolve texture names
        tn = []
        for t in mat.pop("texNo"):
            tn.append(tex_names[t] if t is not None and t < len(tex_names) else None)
        mat["texNames"] = tn
        # indirect block
        if offs[3]:
            raw = entry_bytes("indirect", m)
            mat["indirect"] = None if raw == default_ind else raw.hex()
        else:
            mat["indirect"] = None
        materials.append(mat)
    return materials, names


# --- writing ----------------------------------------------------------------

class _Table:
    def __init__(self):
        self.items = []
        self.lookup = {}

    def add(self, raw):
        i = self.lookup.get(raw)
        if i is None:
            i = len(self.items)
            self.items.append(raw)
            self.lookup[raw] = i
        return i


def build_mat3(materials, names, tex_name_to_index):
    """Serialise materials into a complete MAT3 chunk (bytes)."""
    tables = {name: _Table() for name, _ in TABLE_SLOTS}
    inits = bytearray()
    indirect = bytearray()
    default_ind = default_indirect()

    for mat in materials:
        init = bytearray(0x14C)
        m = dict(mat)
        tex_idx = []
        for name in m.get("texNames", [None] * 8):
            if name is None:
                tex_idx.append(None)
            else:
                if name not in tex_name_to_index:
                    raise J3DError(f"Texture '{name}' referenced by a material is not in the texture list")
                tex_idx.append(tex_name_to_index[name])
        m["texNo"] = tex_idx

        for field, io, cnt, code, table in INIT_FIELDS:
            val = m.get(field)
            if table is None:
                vals = val if cnt > 1 else [val]
                vals = list(vals) + [0xFFFF if code == "H" else 0xFF] * (cnt - len(vals))
                struct.pack_into(">" + code * cnt, init, io, *vals)
                continue
            packer = _ENTRY[table][1]
            vals = [val] if field in SCALAR_TABLE_FIELDS else list(val)
            vals = vals + [None] * (cnt - len(vals))
            idxs = []
            for v in vals:
                if v is None:
                    idxs.append(0xFFFF if code == "H" else 0xFF)
                else:
                    idxs.append(tables[table].add(packer(v)))
            struct.pack_into(">" + code * cnt, init, io, *idxs)
        inits += init
        ind = m.get("indirect")
        indirect += bytes.fromhex(ind) if ind else default_ind

    count = len(materials)
    # --- assemble the chunk
    header_size = 0x0C + 4 * len(TABLE_SLOTS)
    body = bytearray()
    offsets = [0] * len(TABLE_SLOTS)

    def put(slot_name, raw, even_if_empty=False):
        if not raw and not even_if_empty:
            return
        while len(body) % 4:
            body.append(0)
        offsets[SLOT_INDEX[slot_name]] = header_size + len(body)
        body.extend(raw)

    put("init", bytes(inits))
    put("remap", struct.pack(f">{count}H", *range(count)))
    put("names", write_string_table(names))
    put("indirect", bytes(indirect))
    pending = []  # empty 'light' table: game files point it at the following table (offset != 0)
    for name, size in TABLE_SLOTS[4:]:
        raw = b"".join(tables[name].items)
        if not raw and name == "light":
            pending.append(name)
            continue
        put(name, raw)
        for p in pending:
            offsets[SLOT_INDEX[p]] = offsets[SLOT_INDEX[name]]
        pending = []

    chunk = bytearray(b"MAT3")
    chunk += struct.pack(">IHH", 0, count, 0xFFFF)
    chunk += struct.pack(f">{len(TABLE_SLOTS)}I", *offsets)
    chunk += body
    pad_to(chunk, 32)
    struct.pack_into(">I", chunk, 4, len(chunk))
    return bytes(chunk)


# --- defaults -----------------------------------------------------------------

def default_fog():
    return {"type": 2, "enable": 0, "center": 320, "startZ": 5000.0, "endZ": 25000.0,
            "nearZ": 1.0, "farZ": 500000.0, "color": [0x60, 0x60, 0x60, 0xFF],
            "adjTable": [0x0100, 0x0101, 0x0103, 0x0106, 0x0109, 0x010D, 0x0112, 0x0117, 0x011D, 0x0124]}


RENDER_OPAQUE, RENDER_CUTOUT, RENDER_TRANSLUCENT = "OPAQUE", "CUTOUT", "TRANSLUCENT"


def apply_render_mode(mat, mode, alpha_ref=128):
    """Set flags / alpha test / blend / z-mode fields for a high-level render mode."""
    if mode == RENDER_OPAQUE:
        mat["flags"] = 1
        mat["alphaComp"] = [7, 0, 1, 7, 0, 0xFF, 0xFF, 0xFF]
        mat["blend"] = [0, 1, 0, 3]
        mat["zMode"] = [1, 3, 1, 0xFF]
        mat["zCompLoc"] = 1
    elif mode == RENDER_CUTOUT:
        mat["flags"] = 1
        mat["alphaComp"] = [6, int(alpha_ref), 0, 3, 0xFF, 0xFF, 0xFF, 0xFF]
        mat["blend"] = [0, 1, 0, 3]
        mat["zMode"] = [1, 3, 1, 0xFF]
        mat["zCompLoc"] = 0
    elif mode == RENDER_TRANSLUCENT:
        mat["flags"] = 4
        mat["alphaComp"] = [7, 0, 1, 7, 0, 0xFF, 0xFF, 0xFF]
        mat["blend"] = [1, 4, 5, 3]
        mat["zMode"] = [1, 3, 0, 0xFF]
        mat["zCompLoc"] = 0
    else:
        raise ValueError(mode)
    return mat


def default_material(texture_name=None, use_vertex_colors=False, mode=RENDER_OPAQUE,
                     cull_back=True, alpha_ref=128):
    """Build a simple unlit material: texture * vertex colour (or white)."""
    src = 1 if use_vertex_colors else 0  # GX_SRC_VTX / GX_SRC_REG
    unlit = lambda s: [0, s, 0, 2, 1, 0, 0xFF, 0xFF]
    tex = texture_name is not None
    mat = {
        "cull": 2 if cull_back else 0,
        "colorChanCount": 1,
        "matColor": [[255, 255, 255, 255], [255, 255, 255, 255]],
        "colorChan": [unlit(src), unlit(src), unlit(0), unlit(0)],
        "ambColor": [[255, 255, 255, 255], [0, 0, 0, 255]],
        "light": [None] * 8,
        "texGenCount": 1 if tex else 0,
        "texCoord": ([[1, 4, 0x3C, 0xFF]] if tex else [None]) + [None] * 7,
        "texCoord2": [None] * 8,
        "texMtx": [None] * 10,
        "postTexMtx": [None] * 20,
        "texNames": [texture_name] + [None] * 7,
        "kColor": [[255, 255, 255, 255]] * 4,
        "kColorSel": [12] * 16,
        "kAlphaSel": [28] * 16,
        "tevOrder": [[0 if tex else 0xFF, 0 if tex else 0xFF, 4, 0xFF]] + [None] * 15,
        "tevColor": [[0, 0, 0, 255], [255, 255, 255, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
        "tevStageCount": 1,
        "tevStage": [{
            "u0": 0xFF,
            "colorIn": [15, 8, 10, 15] if tex else [15, 15, 15, 10],
            "colorOp": 0, "colorBias": 0, "colorScale": 0, "colorClamp": 1, "colorReg": 0,
            "alphaIn": [7, 4, 5, 7] if tex else [7, 7, 7, 5],
            "alphaOp": 0, "alphaBias": 0, "alphaScale": 0, "alphaClamp": 1, "alphaReg": 0,
            "u19": 0xFF,
        }] + [None] * 15,
        "tevSwapMode": [[0, 0, 0xFF, 0xFF]] + [None] * 15,
        "tevSwapTable": [[0, 1, 2, 3]] + [None] * 3,
        "unknown12": [0xFFFF] * 12,
        "fog": default_fog(),
        "dither": 1,
        "nbtScale": {"enable": 0, "scale": [1.0, 1.0, 1.0]},
        "indirect": None,
    }
    apply_render_mode(mat, mode, alpha_ref)
    return mat


def classify_alpha(mat):
    """Return (kind, reference) with kind in OPAQUE / CUTOUT / TRANSLUCENT, as the game draws it."""
    blend = mat.get("blend") or [0]
    if blend[0] == 1:
        return RENDER_TRANSLUCENT, None
    ac = mat.get("alphaComp")
    if ac:
        if ac[0] in (4, 6):      # GREATER / GEQUAL
            return RENDER_CUTOUT, ac[1]
        if ac[3] in (4, 6) and ac[0] == 7 and ac[2] != 1:
            return RENDER_CUTOUT, ac[4]
    return RENDER_OPAQUE, None


def required_uv_sets(mat):
    """Indices (0-7) of UV sets referenced by the material's texgens."""
    used = set()
    n = mat.get("texGenCount", 0)
    for tc in mat.get("texCoord", [])[:n]:
        if tc and 4 <= tc[1] <= 11:
            used.add(tc[1] - 4)
    return sorted(used)


def uses_vertex_colors(mat):
    return any(c and c[1] == 1 for c in mat.get("colorChan", [])[:2 * max(1, mat.get("colorChanCount", 1))])


def deep_copy(mat):
    return copy.deepcopy(mat)
