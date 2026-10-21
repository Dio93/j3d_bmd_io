"""J3D (BMD3 / BDL4) model parsing and geometry extraction (no Blender dependency)."""

import math
import struct

import numpy as np

from . import gx_texture as gx
from . import material as mat3
from .common import (J3DError, read_string_table, s16, u8, u16, u32, yaz0_decompress)

# GX vertex attributes
VA_PNMTXIDX, VA_POS, VA_NRM, VA_CLR0, VA_CLR1, VA_TEX0 = 0, 9, 10, 11, 12, 13
VA_NBT = 25

# index of an attribute in the VTX1 offset table
ATTR_SLOT = {VA_POS: 0, VA_NRM: 1, VA_NBT: 2, VA_CLR0: 3, VA_CLR1: 4}
ATTR_SLOT.update({VA_TEX0 + i: 5 + i for i in range(8)})

# GX primitive opcodes
PRIM_QUADS, PRIM_TRIS, PRIM_STRIP, PRIM_FAN = 0x80, 0x90, 0x98, 0xA0

HIER_END, HIER_OPEN, HIER_CLOSE, HIER_JOINT, HIER_MATERIAL, HIER_SHAPE = 0, 1, 2, 0x10, 0x11, 0x12

INF_MAYA = 2


# --------------------------------------------------------------------------
# data classes
# --------------------------------------------------------------------------

class Joint:
    def __init__(self, name="joint"):
        self.name = name
        self.kind = 1
        self.scale_compensate = 0
        self.scale = [1.0, 1.0, 1.0]
        self.rot = [0.0, 0.0, 0.0]  # radians, XYZ euler (R = Rz*Ry*Rx)
        self.trans = [0.0, 0.0, 0.0]
        self.radius = 0.0
        self.bbmin = [0.0, 0.0, 0.0]
        self.bbmax = [0.0, 0.0, 0.0]
        self.parent = -1


class Texture:
    def __init__(self, name="texture"):
        self.name = name       # name stored in the file (may repeat)
        self.key = name        # unique identifier used by materials / Blender images
        self.format = gx.CMPR
        self.transparency = 0
        self.wrap_s = 1
        self.wrap_t = 1
        self.min_filter = 1
        self.mag_filter = 1
        self.min_lod = 0       # raw s8 (units of 1/8)
        self.max_lod = 0       # raw s8
        self.mip_count = 1
        self.lod_bias = 0      # raw s16 (units of 1/100)
        self.flags10 = b"\x00\x00\x00\x00"
        self.rgba = None       # (h, w, 4) uint8
        # original encoded data (all mips) - written back verbatim if the image is unchanged
        self.raw_data = None
        self.raw_palette = None
        # True for placeholder entries without image data (data offset 0 in the header).  The game
        # fills these slots at runtime (e.g. texture-pattern animations); they must stay empty.
        self.external = False
        self.pal_enabled = 0
        self.pal_format = 0
        self.pal_count = 0


class Shape:
    def __init__(self):
        self.index = 0
        self.matrix_type = 0
        self.attribs = []      # [(attr, type)]
        self.groups = []       # [(matrix_table [drw idx], dl_offset, dl_size)]
        self.radius = 0.0
        self.bbmin = [0.0] * 3
        self.bbmax = [0.0] * 3


class MeshData:
    """Flattened, Blender-friendly geometry of one shape instance."""

    def __init__(self):
        self.name = ""
        self.shape_index = 0
        self.material_index = 0
        self.joint = 0
        self.verts = None            # (V,3) float32, model space
        self.vert_influence = None   # (V,) int32 -> influence_sets
        self.influence_sets = []     # list of tuples ((joint, weight), ...)
        self.faces = None            # (T,3) int32
        self.loop_normals = None     # (3T,3) float32 or None
        self.loop_colors = [None, None]  # (3T,4) float32 or None
        self.loop_uvs = [None] * 8   # (3T,2) float32 or None  (GX convention: v down)


class Bmd:
    def __init__(self):
        self.data = b""
        self.file_tag = b"bmd3"
        self.sub_version = b"SVR3" + b"\xff" * 12
        self.inf_flags = 0
        self.vertex_count = 0
        self.vtx_formats = []
        self.arrays = {}
        self.envelopes = []
        self.drw = []
        self.joints = []
        self.shapes = []
        self.materials = []
        self.material_names = []
        self.textures = []
        self.hierarchy = []
        self.shape_nodes = []  # (joint, material, shape)
        self.world = None

    @property
    def maya(self):
        return (self.inf_flags & 0x0F) == INF_MAYA


# --------------------------------------------------------------------------
# joint maths
# --------------------------------------------------------------------------

def euler_to_matrix(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_euler(R):
    """Inverse of :func:`euler_to_matrix` (R = Rz*Ry*Rx)."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    if abs(sy) < 0.999999:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        z = 0.0
    return x, y, z


def local_matrix(j):
    M = np.eye(4)
    M[:3, :3] = euler_to_matrix(*j.rot) * np.array(j.scale)[None, :]
    M[:3, 3] = j.trans
    return M


def compute_world_matrices(joints, maya=False):
    n = len(joints)
    world = [None] * n

    def get(i, depth=0):
        if world[i] is not None:
            return world[i]
        if depth > n:
            raise J3DError("Cycle in joint hierarchy")
        j = joints[i]
        L = local_matrix(j)
        if j.parent < 0:
            world[i] = L
        else:
            P = get(j.parent, depth + 1)
            if maya and j.scale_compensate:
                ps = np.array(joints[j.parent].scale, dtype=float)
                inv = np.eye(4)
                inv[0, 0], inv[1, 1], inv[2, 2] = 1 / ps[0], 1 / ps[1], 1 / ps[2]
                world[i] = P @ inv @ L
            else:
                world[i] = P @ L
        return world[i]

    for i in range(n):
        get(i)
    return np.array(world)


# --------------------------------------------------------------------------
# chunk parsers
# --------------------------------------------------------------------------

_NP_TYPES = {0: "u1", 1: "i1", 2: ">u2", 3: ">i2", 4: ">f4"}


def _decode_color_array(d, start, nbytes, ctype):
    if ctype == 5:  # RGBA8
        n = nbytes // 4
        a = np.frombuffer(d, np.uint8, n * 4, start).reshape(n, 4) / 255.0
    elif ctype == 2:  # RGBX8
        n = nbytes // 4
        a = np.frombuffer(d, np.uint8, n * 4, start).reshape(n, 4).astype(np.float32) / 255.0
        a[:, 3] = 1.0
    elif ctype == 1:  # RGB8
        n = nbytes // 3
        rgb = np.frombuffer(d, np.uint8, n * 3, start).reshape(n, 3) / 255.0
        a = np.concatenate([rgb, np.ones((n, 1))], 1)
    elif ctype == 0:  # RGB565
        n = nbytes // 2
        a = gx._rgb565_to_rgba(np.frombuffer(d, ">u2", n, start)) / 255.0
    elif ctype == 3:  # RGBA4
        n = nbytes // 2
        v = np.frombuffer(d, ">u2", n, start).astype(np.uint16)
        a = np.stack([(v >> 12) & 15, (v >> 8) & 15, (v >> 4) & 15, v & 15], -1) / 15.0
    elif ctype == 4:  # RGBA6
        n = nbytes // 3
        b = np.frombuffer(d, np.uint8, n * 3, start).reshape(n, 3).astype(np.uint32)
        v = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
        a = np.stack([(v >> 18) & 63, (v >> 12) & 63, (v >> 6) & 63, v & 63], -1) / 63.0
    else:
        raise J3DError(f"Unsupported colour format {ctype}")
    return a.astype(np.float32)


def _parse_vtx1(d, off):
    size = u32(d, off + 4)
    fmt_off = u32(d, off + 8)
    offs = [u32(d, off + 0x0C + 4 * i) for i in range(13)]
    formats = []
    p = off + fmt_off
    while True:
        a, c, t, fr = struct.unpack_from(">IIIB", d, p)
        p += 16
        if a == 0xFF:
            break
        formats.append((a, c, t, fr))
    nonzero = sorted(o for o in offs if o)
    arrays = {}
    for a, c, t, fr in formats:
        if a not in ATTR_SLOT:
            continue
        o = offs[ATTR_SLOT[a]]
        if not o:
            continue
        later = [x for x in nonzero if x > o]
        end = later[0] if later else size
        nbytes = end - o
        start = off + o
        if a in (VA_CLR0, VA_CLR1):
            arrays[a] = _decode_color_array(d, start, nbytes, t)
            continue
        if t not in _NP_TYPES:
            raise J3DError(f"Unsupported vertex component type {t}")
        dt = np.dtype(_NP_TYPES[t])
        if a == VA_POS:
            ncomp = 2 + c
        elif a == VA_NRM:
            ncomp = (3, 9, 9)[min(c, 2)]
        elif a == VA_NBT:
            continue
        else:
            ncomp = 1 + c
        n = nbytes // (dt.itemsize * ncomp)
        arr = np.frombuffer(d, dt, n * ncomp, start).reshape(n, ncomp).astype(np.float32)
        if t != 4 and fr:
            arr = arr / float(1 << fr)
        if a == VA_NRM:
            arr = arr[:, :3]
        elif a == VA_POS and ncomp == 2:
            arr = np.concatenate([arr, np.zeros((n, 1), np.float32)], 1)
        elif a >= VA_TEX0 and ncomp == 1:
            arr = np.concatenate([arr, np.zeros((n, 1), np.float32)], 1)
        arrays[a] = arr
    return formats, arrays


def _parse_evp1(d, off):
    n = u16(d, off + 8)
    if n == 0:
        return []
    o_cnt, o_idx, o_w, _o_m = (u32(d, off + 0x0C + 4 * i) for i in range(4))
    counts = list(d[off + o_cnt: off + o_cnt + n])
    total = sum(counts)
    idx = np.frombuffer(d, ">u2", total, off + o_idx)
    w = np.frombuffer(d, ">f4", total, off + o_w)
    envs, p = [], 0
    for c in counts:
        envs.append(tuple((int(idx[p + k]), float(w[p + k])) for k in range(c)))
        p += c
    return envs


def _parse_drw1(d, off):
    n = u16(d, off + 8)
    o_flag = u32(d, off + 0x0C)
    o_idx = u32(d, off + 0x10)
    return [(d[off + o_flag + i], u16(d, off + o_idx + 2 * i)) for i in range(n)]


def _parse_jnt1(d, off):
    n = u16(d, off + 8)
    o_data = u32(d, off + 0x0C)
    o_remap = u32(d, off + 0x10)
    o_names = u32(d, off + 0x14)
    names = read_string_table(d, off + o_names) if o_names else [f"joint{i}" for i in range(n)]
    joints = []
    for i in range(n):
        r = u16(d, off + o_remap + 2 * i)
        e = off + o_data + 0x40 * r
        j = Joint(names[i] if i < len(names) else f"joint{i}")
        j.kind = u16(d, e)
        j.scale_compensate = u8(d, e + 2)
        j.scale = list(struct.unpack_from(">3f", d, e + 4))
        raw = struct.unpack_from(">3h", d, e + 0x10)
        j.rot = [v * math.pi / 32768.0 for v in raw]
        j.trans = list(struct.unpack_from(">3f", d, e + 0x18))
        j.radius = struct.unpack_from(">f", d, e + 0x24)[0]
        j.bbmin = list(struct.unpack_from(">3f", d, e + 0x28))
        j.bbmax = list(struct.unpack_from(">3f", d, e + 0x34))
        joints.append(j)
    return joints


def _parse_shp1(d, off):
    n = u16(d, off + 8)
    o_shape, o_remap, _o_names, o_attr, o_mtxtab, o_prim, o_mtxgrp, o_pkt = (
        u32(d, off + 0x0C + 4 * i) for i in range(8))
    shapes = []
    for i in range(n):
        r = u16(d, off + o_remap + 2 * i) if o_remap else i
        e = off + o_shape + 0x28 * r
        s = Shape()
        s.index = i
        s.matrix_type = u8(d, e)
        grp_count = u16(d, e + 2)
        attr_off = u16(d, e + 4)
        first_grp = u16(d, e + 6)
        first_pkt = u16(d, e + 8)
        s.radius = struct.unpack_from(">f", d, e + 0x0C)[0]
        s.bbmin = list(struct.unpack_from(">3f", d, e + 0x10))
        s.bbmax = list(struct.unpack_from(">3f", d, e + 0x1C))
        p = off + o_attr + attr_off
        while True:
            a, t = struct.unpack_from(">II", d, p)
            p += 8
            if a == 0xFF:
                break
            s.attribs.append((a, t))
        for g in range(grp_count):
            ge = off + o_mtxgrp + 8 * (first_grp + g)
            cnt = u16(d, ge + 2)
            first = u32(d, ge + 4)
            table = [u16(d, off + o_mtxtab + 2 * (first + k)) for k in range(cnt)]
            pe = off + o_pkt + 8 * (first_pkt + g)
            psize, poff = struct.unpack_from(">II", d, pe)
            s.groups.append((table, off + o_prim + poff, psize))
        shapes.append(s)
    return shapes


def _parse_tex1(d, off):
    n = u16(d, off + 8)
    o_hdr = u32(d, off + 0x0C)
    o_names = u32(d, off + 0x10)
    names = read_string_table(d, off + o_names) if o_names else [f"texture{i}" for i in range(n)]
    textures = []
    for i in range(n):
        e = off + o_hdr + 0x20 * i
        t = Texture(names[i] if i < len(names) else f"texture{i}")
        t.format = u8(d, e)
        t.transparency = u8(d, e + 1)
        w, h = u16(d, e + 2), u16(d, e + 4)
        t.wrap_s, t.wrap_t = u8(d, e + 6), u8(d, e + 7)
        pal_fmt = u8(d, e + 9)
        pal_cnt = u16(d, e + 0x0A)
        pal_off = u32(d, e + 0x0C)
        t.flags10 = bytes(d[e + 0x10:e + 0x14])
        t.min_filter, t.mag_filter = u8(d, e + 0x14), u8(d, e + 0x15)
        t.min_lod = struct.unpack_from(">b", d, e + 0x16)[0]
        t.max_lod = struct.unpack_from(">b", d, e + 0x17)[0]
        t.mip_count = max(1, u8(d, e + 0x18))
        t.lod_bias = s16(d, e + 0x1A)
        data_off = u32(d, e + 0x1C)
        palette = None
        t.pal_enabled, t.pal_format, t.pal_count = u8(d, e + 8), pal_fmt, pal_cnt
        if data_off == 0:
            # No image data in the file: reading from the header itself would yield garbage.
            t.external = True
            t.rgba = np.full((h, w, 4), 128, np.uint8)
            textures.append(t)
            continue
        if t.format in (gx.C4, gx.C8, gx.C14X2) and pal_cnt:
            t.raw_palette = bytes(d[e + pal_off:e + pal_off + pal_cnt * 2])
            palette = gx.decode_palette(pal_fmt, t.raw_palette, pal_cnt)
        start = e + data_off
        size = gx.level_size(t.format, w, h)
        buf = bytes(d[start:start + size])
        if len(buf) < size:
            raise J3DError(f"Texture '{t.name}' data is truncated")
        total = gx.total_size(t.format, w, h, t.mip_count)
        raw = bytes(d[start:start + total])
        t.raw_data = raw if len(raw) == total else None
        t.rgba = gx.decode(t.format, buf, w, h, palette)
        textures.append(t)
    seen = {}
    for t in textures:  # names can repeat in game files: make unique keys
        n = seen.get(t.name, 0)
        seen[t.name] = n + 1
        if n:
            t.key = f"{t.name}#{n}"
    return textures


def _parse_hierarchy(d, off):
    entries = []
    p = off
    while True:
        t, i = struct.unpack_from(">HH", d, p)
        p += 4
        entries.append((t, i))
        if t == HIER_END:
            break
    return entries


def _walk_hierarchy(entries, joints):
    """Fill ``joint.parent`` and return [(joint, material, shape)] triples."""
    stack = []
    cur_joint, cur_mat = -1, -1
    last = None
    nodes = []
    for t, i in entries:
        if t == HIER_JOINT:
            if i < len(joints):
                joints[i].parent = cur_joint
            last = (HIER_JOINT, i)
        elif t == HIER_MATERIAL:
            cur_mat = i
            last = (HIER_MATERIAL, i)
        elif t == HIER_SHAPE:
            nodes.append((cur_joint, cur_mat, i))
            last = (HIER_SHAPE, i)
        elif t == HIER_OPEN:
            stack.append((cur_joint, cur_mat))
            if last and last[0] == HIER_JOINT:
                cur_joint = last[1]
        elif t == HIER_CLOSE:
            if stack:
                cur_joint, cur_mat = stack.pop()
        elif t == HIER_END:
            break
    return nodes


# --------------------------------------------------------------------------
# top level parse
# --------------------------------------------------------------------------

def parse(data):
    """Parse a BMD3/BDL4 file (optionally Yaz0 compressed)."""
    data = bytes(data)
    if data[:4] == b"Yaz0":
        data = yaz0_decompress(data)
    if data[:4] != b"J3D2" or data[4:8] not in (b"bmd3", b"bdl4"):
        raise J3DError("Not a J3D2 bmd3/bdl4 model file")
    bmd = Bmd()
    bmd.data = data
    bmd.file_tag = data[4:8]
    count = u32(data, 12)
    bmd.sub_version = data[16:32]

    chunks = {}
    off = 32
    for _ in range(count):
        if off + 8 > len(data):
            raise J3DError("Truncated file (chunk header)")
        tag = data[off:off + 4].decode("ascii", "replace")
        size = u32(data, off + 4)
        if size < 8:
            raise J3DError(f"Invalid chunk size for {tag}")
        chunks[tag] = off
        off += size

    for req in ("INF1", "VTX1", "DRW1", "JNT1", "SHP1", "MAT3"):
        if req not in chunks:
            raise J3DError(f"Missing chunk {req}")

    o = chunks["INF1"]
    bmd.inf_flags = u16(data, o + 8)
    bmd.vertex_count = u32(data, o + 0x10)
    bmd.hierarchy = _parse_hierarchy(data, o + u32(data, o + 0x14))

    bmd.vtx_formats, bmd.arrays = _parse_vtx1(data, chunks["VTX1"])
    bmd.envelopes = _parse_evp1(data, chunks["EVP1"]) if "EVP1" in chunks else []
    bmd.drw = _parse_drw1(data, chunks["DRW1"])
    bmd.joints = _parse_jnt1(data, chunks["JNT1"])
    bmd.shapes = _parse_shp1(data, chunks["SHP1"])
    bmd.textures = _parse_tex1(data, chunks["TEX1"]) if "TEX1" in chunks else []
    tex_names = [t.key for t in bmd.textures]
    bmd.materials, bmd.material_names = mat3.parse_mat3(data, chunks["MAT3"], tex_names)
    bmd.shape_nodes = _walk_hierarchy(bmd.hierarchy, bmd.joints)
    bmd.world = compute_world_matrices(bmd.joints, bmd.maya)
    return bmd


# --------------------------------------------------------------------------
# geometry extraction
# --------------------------------------------------------------------------

def _triangulate(prim, n, base):
    if prim == PRIM_TRIS:
        m = n - n % 3
        return (base + np.arange(m)).reshape(-1, 3)
    if prim == PRIM_STRIP:
        if n < 3:
            return np.zeros((0, 3), np.int64)
        i = np.arange(n - 2)
        odd = (i & 1) == 1
        a = np.where(odd, i + 1, i)
        b = np.where(odd, i, i + 1)
        return base + np.stack([a, b, i + 2], 1)
    if prim == PRIM_FAN:
        if n < 3:
            return np.zeros((0, 3), np.int64)
        i = np.arange(1, n - 1)
        return base + np.stack([np.zeros_like(i), i, i + 1], 1)
    if prim == PRIM_QUADS:
        m = n - n % 4
        q = (base + np.arange(m)).reshape(-1, 4)
        return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], 0)
    return np.zeros((0, 3), np.int64)


def _decode_shape(bmd, shape):
    """Return per-record attribute arrays + triangle list for one shape."""
    d = bmd.data
    fields = []
    for a, t in shape.attribs:
        if t == 0:
            continue
        if t == 1:
            if a > 8:
                raise J3DError("Direct (non-indexed) vertex data is not supported")
            fields.append((f"a{a}", "u1"))
        elif t == 2:
            fields.append((f"a{a}", "u1"))
        elif t == 3:
            fields.append((f"a{a}", ">u2"))
        else:
            raise J3DError(f"Unknown attribute type {t}")
    vdt = np.dtype(fields)
    recs, tris, drw_of_rec = [], [], []
    base = 0
    slot_state = [0] * 10  # GX position-matrix slots persist between packets
    for table, start, size in shape.groups:
        for k, v in enumerate(table[:10]):
            if v != 0xFFFF:
                slot_state[k] = v
        pos, end = start, start + size
        group_recs = []
        while pos < end:
            op = d[pos]
            pos += 1
            if op == 0:
                continue
            prim = op & 0xF8
            n = u16(d, pos)
            pos += 2
            arr = np.frombuffer(d, vdt, n, pos)
            pos += n * vdt.itemsize
            tris.append(_triangulate(prim, n, base))
            base += n
            group_recs.append(arr)
        if not group_recs:
            continue
        rec = np.concatenate(group_recs)
        recs.append(rec)
        if "a0" in vdt.names:
            slot = rec["a0"].astype(np.int64) // 3
        else:
            slot = np.zeros(len(rec), np.int64)
        tab = np.array(slot_state, np.int64)
        drw_of_rec.append(tab[np.minimum(slot, 9)])
    if not recs:
        return None
    rec = np.concatenate(recs)
    return rec, np.concatenate(tris, 0), np.concatenate(drw_of_rec)


def build_mesh(bmd, node, mesh_index=0):
    """Create a :class:`MeshData` for an INF1 (joint, material, shape) node."""
    joint, material, shape_idx = node
    shape = bmd.shapes[shape_idx]
    decoded = _decode_shape(bmd, shape)
    mesh = MeshData()
    mesh.shape_index = shape_idx
    mesh.material_index = material
    mesh.joint = max(joint, 0)
    if decoded is None:
        return None
    rec, tris, drw = decoded
    names = rec.dtype.names

    def col(attr):
        key = f"a{attr}"
        return rec[key].astype(np.int64) if key in names else None

    pos_i = col(VA_POS)
    if pos_i is None:
        return None
    pos_arr = bmd.arrays[VA_POS]
    pos_i = np.minimum(pos_i, len(pos_arr) - 1)

    # drop degenerate triangles (same vertex key more than once)
    ndrw = max(len(bmd.drw), 1)
    key = pos_i * ndrw + drw
    k0, k1, k2 = key[tris[:, 0]], key[tris[:, 1]], key[tris[:, 2]]
    good = (k0 != k1) & (k1 != k2) & (k0 != k2)
    tris = tris[good]
    if len(tris) == 0:
        return None
    loops = tris.reshape(-1)

    uniq, inverse = np.unique(key, return_inverse=True)
    u_pos_i = uniq // ndrw
    u_drw = uniq % ndrw
    verts = pos_arr[u_pos_i].astype(np.float64)

    drw_ids = np.unique(u_drw)
    inf_sets = []
    for k in drw_ids:
        weighted, idx = bmd.drw[int(k)]
        if weighted:
            inf_sets.append(tuple(bmd.envelopes[idx]))
        else:
            inf_sets.append(((int(idx), 1.0),))
    lookup = {int(k): n for n, k in enumerate(drw_ids)}
    vert_inf = np.array([lookup[int(k)] for k in u_drw], np.int32)

    W = bmd.world
    for k in drw_ids:
        weighted, idx = bmd.drw[int(k)]
        if weighted:
            continue
        m = u_drw == k
        M = W[idx]
        verts[m] = verts[m] @ M[:3, :3].T + M[:3, 3]

    mesh.verts = verts.astype(np.float32)
    mesh.vert_influence = vert_inf
    mesh.influence_sets = inf_sets
    mesh.faces = inverse[tris].astype(np.int32)

    nrm_i = col(VA_NRM)
    if nrm_i is not None and VA_NRM in bmd.arrays:
        na = bmd.arrays[VA_NRM]
        nrm = na[np.minimum(nrm_i[loops], len(na) - 1)].astype(np.float64)
        loop_drw = drw[loops]
        for k in drw_ids:
            weighted, idx = bmd.drw[int(k)]
            if weighted:
                continue
            m = loop_drw == k
            nrm[m] = nrm[m] @ W[idx][:3, :3].T
        length = np.linalg.norm(nrm, axis=1, keepdims=True)
        length[length == 0] = 1
        mesh.loop_normals = (nrm / length).astype(np.float32)

    for c, attr in enumerate((VA_CLR0, VA_CLR1)):
        ci = col(attr)
        if ci is not None and attr in bmd.arrays:
            ca = bmd.arrays[attr]
            mesh.loop_colors[c] = ca[np.minimum(ci[loops], len(ca) - 1)]
    for t in range(8):
        ti = col(VA_TEX0 + t)
        if ti is not None and (VA_TEX0 + t) in bmd.arrays:
            ta = bmd.arrays[VA_TEX0 + t]
            mesh.loop_uvs[t] = ta[np.minimum(ti[loops], len(ta) - 1)][:, :2]
    return mesh


def extract_meshes(bmd):
    meshes = []
    for n, node in enumerate(bmd.shape_nodes):
        m = build_mesh(bmd, node, n)
        if m is not None:
            meshes.append(m)
    return meshes
