"""Serialise an :class:`ExportModel` into a BMD3 file."""

import math
import struct

import numpy as np

from . import gx_texture as gx
from . import material as mat3
from .common import J3DError, align, pad_to, write_string_table
from .model import (HIER_CLOSE, HIER_END, HIER_JOINT, HIER_MATERIAL, HIER_OPEN, HIER_SHAPE,
                    VA_CLR0, VA_CLR1, VA_NRM, VA_POS, VA_TEX0, compute_world_matrices)

_INT_TYPES = {0: "u1", 1: "i1", 2: ">u2", 3: ">i2"}
MAX_PACKET_MATRICES = 10
MAX_PRIM_VERTS = 65532  # multiple of 3 below the u16 limit


class ExportMesh:
    def __init__(self):
        self.name = ""
        self.material = 0            # index into ExportModel.materials
        self.verts = None            # (V,3) model space
        self.vert_influence = None   # (V,) -> influence_sets
        self.influence_sets = []     # tuples of (joint, weight), normalised
        self.faces = None            # (T,3) indices into verts
        self.loop_normals = None     # (3T,3)
        self.loop_colors = [None, None]  # (3T,4) 0..1
        self.loop_uvs = [None] * 8   # (3T,2) GX convention
        self.sort_key = 0            # export order of shapes (original shape index)
        self.order = 0               # position in the scene graph (INF1) among shapes of a joint
        self.joint = None            # joint the shape hangs under (None = derive from skinning)


class ExportModel:
    def __init__(self):
        self.joints = []
        self.inf_flags = 0
        self.meshes = []
        self.materials = []
        self.material_names = []
        self.textures = []
        self.sub_version = b"SVR3" + b"\xff" * 12
        self.file_tag = b"bmd3"
        # {attr: (comp_count, comp_type, frac_bits)} of the imported file; None = write floats
        self.vtx_formats = None


def _build_strips(tri):
    """Greedy triangle stripper. ``tri`` = (T,3) vertex-record ids; returns lists of ids.

    Strips follow the GX convention used by the reader: triangle i is (v[i], v[i+1], v[i+2])
    for even i and (v[i+1], v[i], v[i+2]) for odd i, so every face keeps its orientation.
    """
    tri = [tuple(int(x) for x in t) for t in tri]
    n = len(tri)
    used = [False] * n
    edge = {}
    for f, (a, b, c) in enumerate(tri):
        for p, q in ((a, b), (b, c), (c, a)):
            edge.setdefault((p, q), []).append(f)

    def grow(f, rot):
        a, b, c = tri[f]
        strip = [(a, b, c), (b, c, a), (c, a, b)][rot]
        strip = list(strip)
        faces = [f]
        while True:
            t = len(strip) - 2
            p, q = (strip[-2], strip[-1]) if t % 2 == 0 else (strip[-1], strip[-2])
            found = None
            for g in edge.get((p, q), ()):
                if used[g] or g in faces:
                    continue
                x, y, z = tri[g]
                third = z if (x, y) == (p, q) else (x if (y, z) == (p, q) else y)
                found = (g, third)
                break
            if found is None:
                return strip, faces
            faces.append(found[0])
            strip.append(found[1])

    strips = []
    for f in range(n):
        if used[f]:
            continue
        best = max((grow(f, r) for r in range(3)), key=lambda sf: len(sf[1]))
        for g in best[1]:
            used[g] = True
        strips.append(best[0])
    return strips


def _chunk(tag, body):
    out = bytearray(tag)
    out += struct.pack(">I", 0)
    out += body
    pad_to(out, 32)
    struct.pack_into(">I", out, 4, len(out))
    return bytes(out)


def _s16_angle(rad):
    v = int(round(rad / math.pi * 32768.0))
    return ((v + 32768) % 65536) - 32768


# --------------------------------------------------------------------------

def _build_hierarchy(joints, nodes_by_joint):
    """Emit the Nintendo-style nested chain  J { M { S { M { S ... } } } }."""
    children = {i: [] for i in range(-1, len(joints))}
    for i, j in enumerate(joints):
        children[j.parent].append(i)
    out = []

    def chain(pairs, kids):
        toks = []
        if pairs:
            (m, s), rest = pairs[0], pairs[1:]
            toks += [(HIER_MATERIAL, m), (HIER_OPEN, 0), (HIER_SHAPE, s)]
            inner = chain(rest, kids)
            if inner:
                toks += [(HIER_OPEN, 0)] + inner + [(HIER_CLOSE, 0)]
            toks += [(HIER_CLOSE, 0)]
        else:
            for k in kids:
                toks += joint(k)
        return toks

    def joint(i):
        toks = [(HIER_JOINT, i)]
        inner = chain(nodes_by_joint.get(i, []), children[i])
        if inner:
            toks += [(HIER_OPEN, 0)] + inner + [(HIER_CLOSE, 0)]
        return toks

    for r in children[-1]:
        out += joint(r)
    out.append((HIER_END, 0))
    return out


def write(model):
    joints = model.joints
    if not joints:
        raise J3DError("Model has no joints")
    maya = (model.inf_flags & 0x0F) == 2
    world = compute_world_matrices(joints, maya)
    inv_world = np.array([np.linalg.inv(w) for w in world])
    nj = len(joints)

    # ---------------- skinning: DRW1 / EVP1 -----------------
    # The engine (J3DModelLoader::readDraw / J3DMtxBuffer) expects this exact DRW1 layout:
    #   [ rigid entries (flag 0, joint index) ]            -> draw matrices 0 .. R-1
    #   [ one weighted entry per envelope, in order ]       -> draw matrices R .. R+E-1
    #   [ E filler entries (the same envelopes again) ]     -> never referenced
    # because it computes  drawMtxNum = DRW1.count - EVP1.count  and fills the weighted draw
    # matrices as  view * envelope[k]  for k = 0..E-1.  Shapes address draw matrices by position.
    def classify(infl):
        infl = tuple(infl)
        if len(infl) == 1 or max(w for _, w in infl) > 0.9999:
            return 0, max(infl, key=lambda x: x[1])[0]
        return 1, infl

    rigid_joints, envelopes, env_lookup = set(), [], {}
    for mesh in model.meshes:
        for infl in mesh.influence_sets:
            kind, val = classify(infl)
            if kind == 0:
                rigid_joints.add(val)
            elif val not in env_lookup:
                env_lookup[val] = len(envelopes)
                envelopes.append(val)
    rigid = sorted(rigid_joints)
    rigid_pos = {j: i for i, j in enumerate(rigid)}
    ne = len(envelopes)
    drws = ([(0, j) for j in rigid] + [(1, k) for k in range(ne)] + [(1, k) for k in range(ne)])

    def drw_index(infl):
        kind, val = classify(infl)
        return rigid_pos[val] if kind == 0 else len(rigid) + env_lookup[val]

    prepared = []
    for mesh in model.meshes:
        set_drw = np.array([drw_index(s) for s in mesh.influence_sets], np.int64)
        vdrw = set_drw[mesh.vert_influence]
        prepared.append(vdrw)

    # --------------- attribute data per mesh (stored space) ---------------
    pos_chunks, nrm_chunks = [], []
    col_chunks = [[], []]
    uv_chunks = [[] for _ in range(8)]
    per_mesh = []
    for mesh, vdrw in zip(model.meshes, prepared):
        verts = mesh.verts.astype(np.float64).copy()
        stored = verts.copy()
        for k in np.unique(vdrw):
            weighted, idx = drws[int(k)]
            if weighted:
                continue
            m = vdrw == k
            M = inv_world[idx]
            stored[m] = verts[m] @ M[:3, :3].T + M[:3, 3]
        faces = mesh.faces.astype(np.int64)
        loops = faces.reshape(-1)
        info = {"vdrw": vdrw, "stored": stored, "loops": loops}
        if mesh.loop_normals is not None:
            n = mesh.loop_normals.astype(np.float64).copy()
            ld = vdrw[loops]
            for k in np.unique(ld):
                weighted, idx = drws[int(k)]
                if weighted:
                    continue
                m = ld == k
                n[m] = n[m] @ inv_world[idx][:3, :3].T
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            ln[ln == 0] = 1
            info["normals"] = np.round(n / ln, 4).astype(np.float32)
        per_mesh.append(info)

    def quantize(attr, arr):
        """Encode an array like the imported file did (e.g. s16 / 14 frac bits) if it fits."""
        fmt = (model.vtx_formats or {}).get(attr)
        default_cnt = 0 if attr == VA_NRM else 1
        if fmt and fmt[1] in _INT_TYPES:
            dt = _INT_TYPES[fmt[1]]
            info = np.iinfo(np.dtype(dt))
            q = np.rint(np.asarray(arr, np.float64) * float(1 << fmt[2]))
            if len(q) == 0 or (q.min() >= info.min - 1 and q.max() <= info.max + 1):
                q = np.clip(q, info.min, info.max)  # at most one LSB of clamping (e.g. UV 4.0)
                return q.astype(np.int64), (default_cnt, fmt[1], fmt[2]), dt
        return np.asarray(arr, np.float32), (default_cnt, 4, 0), ">f4"

    def uniq_attr(attr, chunks):
        allr, fmt, dt = quantize(attr, np.concatenate(chunks))
        u, inv = np.unique(allr, axis=0, return_inverse=True)
        return u, inv.reshape(-1), fmt, dt

    # positions are indexed per *vertex*; the DL indexes per loop through faces
    pos_u, pos_inv, pos_fmt, pos_dt = uniq_attr(VA_POS, [i["stored"] for i in per_mesh])
    offs = 0
    for info in per_mesh:
        nv = len(info["stored"])
        info["pos_idx"] = pos_inv[offs:offs + nv]
        offs += nv

    def collect(attr, key_fn):
        sel = [(i, key_fn(m, i)) for i, m in enumerate(model.meshes)]
        sel = [(i, a) for i, a in sel if a is not None]
        if not sel:
            return None
        u, inv, fmt, dt = uniq_attr(attr, [a for _, a in sel])
        res, o = {}, 0
        for i, a in sel:
            res[i] = inv[o:o + len(a)]
            o += len(a)
        return u, res, fmt, dt

    nrm = collect(VA_NRM, lambda m, i: per_mesh[i].get("normals"))
    cols = []
    for c in range(2):
        sel = [(i, m.loop_colors[c]) for i, m in enumerate(model.meshes) if m.loop_colors[c] is not None]
        if not sel:
            cols.append(None)
            continue
        allc = np.concatenate([np.round(np.clip(a, 0, 1) * 255).astype(np.uint8) for _, a in sel])
        u, inv = np.unique(allc, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        res, o = {}, 0
        for i, a in sel:
            res[i] = inv[o:o + len(a)]
            o += len(a)
        cols.append((u, res))
    uvs = [collect(VA_TEX0 + t, lambda m, i, t=t: m.loop_uvs[t]) for t in range(8)]

    # ------------------------- SHP1 ----------------------------
    attr_lists, attr_lookup = bytearray(), {}
    matrix_table = []
    prim_data = bytearray()
    group_entries = bytearray()
    packet_entries = bytearray()
    shape_entries = bytearray()
    total_groups = 0
    shape_bounds = []

    for mi, (mesh, info) in enumerate(zip(model.meshes, per_mesh)):
        vdrw = info["vdrw"]
        faces = mesh.faces.astype(np.int64)
        loops = info["loops"]
        face_draw = vdrw[faces]  # (T,3)
        uniq_draw = np.unique(vdrw)
        multi = len(uniq_draw) > 1

        attribs = []
        if multi:
            attribs.append((0, 1))
        attribs.append((VA_POS, 3))
        fields = []
        if multi:
            fields.append(("pn", "u1"))
        fields.append(("pos", ">u2"))
        have_n = nrm is not None and mi in nrm[1]
        if have_n:
            attribs.append((VA_NRM, 3))
            fields.append(("nrm", ">u2"))
        for c, attr in ((0, VA_CLR0), (1, VA_CLR1)):
            if cols[c] is not None and mi in cols[c][1]:
                attribs.append((attr, 3))
                fields.append((f"c{c}", ">u2"))
        for t in range(8):
            if uvs[t] is not None and mi in uvs[t][1]:
                attribs.append((VA_TEX0 + t, 3))
                fields.append((f"t{t}", ">u2"))
        for a, t in attribs:
            if a > VA_POS and a > 8 and t != 3:
                raise J3DError("internal: bad attribute type")
        vdt = np.dtype(fields)

        raw_attr = b"".join(struct.pack(">II", a, t) for a, t in attribs) + struct.pack(">II", 0xFF, 0)
        if raw_attr not in attr_lookup:
            attr_lookup[raw_attr] = len(attr_lists)
            attr_lists += raw_attr
        attr_off = attr_lookup[raw_attr]

        # greedy packet splitting by distinct draw matrices
        # cluster faces that share draw matrices so that packets stay few and small
        face_sorted = np.sort(face_draw, axis=1)
        face_order = np.lexsort(face_sorted.T[::-1])
        packets, cur, cur_set = [], [], set()
        for f in face_order.tolist():
            fs = set(face_draw[f].tolist())
            if len(cur_set | fs) > MAX_PACKET_MATRICES:
                packets.append((cur, cur_set))
                cur, cur_set = [], set()
            cur.append(f)
            cur_set |= fs
        if cur:
            packets.append((cur, cur_set))

        first_group = len(group_entries) // 8
        first_packet = len(packet_entries) // 8
        for face_ids, dset in packets:
            order = sorted(dset)
            slot_of = {d: s for s, d in enumerate(order)}
            fl = np.array(face_ids, np.int64)
            lp = (fl[:, None] * 3 + np.arange(3)).reshape(-1)  # loop ids
            vert_ids = faces[fl].reshape(-1)
            rec = np.zeros(len(lp), vdt)
            if multi:
                lut = np.zeros(int(vdrw.max()) + 1, np.uint8)
                for d, s in slot_of.items():
                    lut[d] = s * 3
                rec["pn"] = lut[vdrw[vert_ids]]
            rec["pos"] = info["pos_idx"][vert_ids]
            if have_n:
                rec["nrm"] = nrm[1][mi][lp]
            for c in range(2):
                if cols[c] is not None and mi in cols[c][1]:
                    rec[f"c{c}"] = cols[c][1][mi][lp]
            for t in range(8):
                if uvs[t] is not None and mi in uvs[t][1]:
                    rec[f"t{t}"] = uvs[t][1][mi][lp]
            rec = np.ascontiguousarray(rec)
            raw = rec.view(np.uint8).reshape(len(rec), vdt.itemsize)
            _u, first, rid = np.unique(raw, axis=0, return_index=True, return_inverse=True)
            urec = rec[first]
            dl = bytearray()
            for strip in _build_strips(rid.reshape(-1, 3)):
                dl += bytes([0x98]) + struct.pack(">H", len(strip)) + urec[strip].tobytes()
            dl += b"\0" * (align(len(dl), 32) - len(dl))
            packet_entries += struct.pack(">II", len(dl), len(prim_data))
            prim_data += dl
            first_idx = len(matrix_table)
            matrix_table.extend(order)
            group_entries += struct.pack(">HHI", order[0], len(order), first_idx)
            total_groups += 1

        # game files store shape bounds in model space; the radius is measured from the origin
        st = mesh.verts.astype(np.float64)
        bmin, bmax = st.min(0), st.max(0)
        radius = float(np.linalg.norm(st, axis=1).max())
        shape_entries += struct.pack(
            ">BBHHHHH f3f3f", 3 if multi else 0, 0xFF, len(packets), attr_off, first_group,
            first_packet, 0xFFFF, radius, *bmin, *bmax)
        # NOTE: struct above: B pad(0xFF) -> keep 0xFF like original files
        shape_bounds.append((bmin, bmax))

    nshapes = len(model.meshes)
    # fix the format: BBHHHHH is 12 bytes, floats 4+12+12 -> 0x28
    assert len(shape_entries) == 0x28 * nshapes, len(shape_entries)

    def table_bytes(vals):
        b = struct.pack(f">{len(vals)}H", *vals)
        return b

    shp = bytearray(b"SHP1")
    shp += struct.pack(">IHH", 0, nshapes, 0xFFFF)
    header_end = 0x2C
    shp += b"\0" * (header_end - len(shp))
    sections = [
        ("shape", bytes(shape_entries)),
        ("remap", table_bytes(list(range(nshapes)))),
        ("attr", bytes(attr_lists)),
        ("mtxtab", table_bytes(matrix_table)),
        ("prim", bytes(prim_data)),
        ("group", bytes(group_entries)),
        ("packet", bytes(packet_entries)),
    ]
    sec_off = {}
    for name, raw in sections:
        while len(shp) % 4:
            shp.append(0)
        if name in ("prim",):
            while len(shp) % 32:
                shp.append(0)
        sec_off[name] = len(shp)
        shp += raw
    struct.pack_into(">IIIIIIII", shp, 0x0C, sec_off["shape"], sec_off["remap"], 0,
                     sec_off["attr"], sec_off["mtxtab"], sec_off["prim"], sec_off["group"],
                     sec_off["packet"])
    shp_chunk = _chunk(b"SHP1", bytes(shp[8:]))

    # ------------------------- VTX1 ----------------------------
    arrays = []  # (slot, attr, cnt, type, frac, bytes)
    arrays.append((0, VA_POS, pos_fmt[0], pos_fmt[1], pos_fmt[2], pos_u.astype(pos_dt).tobytes()))
    if nrm is not None:
        arrays.append((1, VA_NRM, nrm[2][0], nrm[2][1], nrm[2][2], nrm[0].astype(nrm[3]).tobytes()))
    for c in range(2):
        if cols[c] is not None:
            arrays.append((3 + c, VA_CLR0 + c, 1, 5, 0, cols[c][0].astype(np.uint8).tobytes()))
    for t in range(8):
        if uvs[t] is not None:
            u, _res, fmt, dt = uvs[t]
            arrays.append((5 + t, VA_TEX0 + t, fmt[0], fmt[1], fmt[2], u.astype(dt).tobytes()))
    fmt_bytes = b"".join(struct.pack(">IIIB3x", a, c, ty, fr) for _, a, c, ty, fr, _ in arrays)
    fmt_bytes += struct.pack(">IIIB3x", 0xFF, 0, 0, 0)
    body = bytearray(fmt_bytes)          # begins at chunk offset 0x40
    slot_offsets = [0] * 13
    for slot, a, c, ty, fr, raw in arrays:
        body += b"\0" * (align(0x40 + len(body), 32) - 0x40 - len(body))
        slot_offsets[slot] = 0x40 + len(body)
        body += raw
    chunk = bytearray(b"VTX1") + struct.pack(">I", 0) + struct.pack(">I", 0x40)
    chunk += struct.pack(">13I", *slot_offsets)
    chunk += body
    pad_to(chunk, 32)
    struct.pack_into(">I", chunk, 4, len(chunk))
    vtx_chunk = bytes(chunk)

    # ------------------------- EVP1 ----------------------------
    evp = bytearray(b"EVP1") + struct.pack(">IHH", 0, len(envelopes), 0xFFFF)
    if envelopes:
        counts = bytes(len(e) for e in envelopes)
        idx = struct.pack(f">{sum(len(e) for e in envelopes)}H", *[j for e in envelopes for j, _ in e])
        wts = struct.pack(f">{sum(len(e) for e in envelopes)}f", *[w for e in envelopes for _, w in e])
        mats = b"".join(struct.pack(">12f", *inv_world[j][:3, :4].reshape(-1)) for j in range(nj))
        o = 0x1C
        o_cnt = o
        o += len(counts)
        o = align(o, 4)
        o_idx = o
        o += len(idx)
        o = align(o, 4)
        o_w = o
        o += len(wts)
        o_m = o
        evp += struct.pack(">4I", o_cnt, o_idx, o_w, o_m)
        evp += counts
        evp += b"\0" * (o_idx - len(evp))
        evp += idx
        evp += b"\0" * (o_w - len(evp))
        evp += wts + mats
    else:
        evp += struct.pack(">4I", 0, 0, 0, 0)
    evp_chunk = _chunk(b"EVP1", bytes(evp[8:]))

    # ------------------------- DRW1 ----------------------------
    n = len(drws)
    o_flag = 0x14
    o_idx = align(o_flag + n, 2)
    drw = bytearray(b"DRW1") + struct.pack(">IHHII", 0, n, 0xFFFF, o_flag, o_idx)
    drw += bytes(w for w, _ in drws)
    drw += b"\0" * (o_idx - len(drw))
    drw += struct.pack(f">{n}H", *[i for _, i in drws])
    drw_chunk = _chunk(b"DRW1", bytes(drw[8:]))

    # ------------------------- JNT1 ----------------------------
    # Like the original files: only the root joint carries a bounding box, in model space.
    allv = np.concatenate([m.verts for m in model.meshes]).astype(np.float64)
    roots = [i for i, j in enumerate(joints) if j.parent < 0]
    jmin = np.full((nj, 3), np.inf)
    jmax = np.full((nj, 3), -np.inf)
    jrad = np.zeros(nj)
    jmin[roots[0]], jmax[roots[0]] = allv.min(0), allv.max(0)
    jrad[roots[0]] = float(np.linalg.norm(allv, axis=1).max())
    jdata = bytearray()
    for i, j in enumerate(joints):
        lo = jmin[i] if np.isfinite(jmin[i]).all() else np.zeros(3)
        hi = jmax[i] if np.isfinite(jmax[i]).all() else np.zeros(3)
        rad = float(jrad[i])
        jdata += struct.pack(">HBB3f3hH3ff3f3f", j.kind, j.scale_compensate, 0, *j.scale,
                             *[_s16_angle(a) for a in j.rot], 0xFFFF, *j.trans, rad, *lo, *hi)
    names = write_string_table([j.name for j in joints])
    o_data = 0x18
    o_remap = o_data + len(jdata)
    o_names = align(o_remap + 2 * nj, 4)
    jnt = bytearray(b"JNT1") + struct.pack(">IHHIII", 0, nj, 0xFFFF, o_data, o_remap, o_names)
    jnt += jdata + struct.pack(f">{nj}H", *range(nj))
    jnt += b"\0" * (o_names - len(jnt)) + names
    jnt_chunk = _chunk(b"JNT1", bytes(jnt[8:]))

    # ------------------------- TEX1 ----------------------------
    tex_chunk, tex_index = _build_tex1(model.textures)

    # ------------------------- MAT3 ----------------------------
    mat_chunk = mat3.build_mat3(model.materials, model.material_names, tex_index)

    # ------------------------- INF1 ----------------------------
    by_joint = {}
    for mi, mesh in enumerate(model.meshes):
        if mesh.joint is not None and 0 <= mesh.joint < nj:
            j = mesh.joint
        else:
            dom = ({s[0][0] for s in mesh.influence_sets}
                   if all(len(s) == 1 for s in mesh.influence_sets) else set())
            j = next(iter(dom)) if len(dom) == 1 else roots[0]
        by_joint.setdefault(j, []).append((mesh.order, mesh.material, mi))
    by_joint = {j: [(m, s) for _o, m, s in sorted(v)] for j, v in by_joint.items()}
    hier = _build_hierarchy(joints, by_joint)
    inf = bytearray(b"INF1") + struct.pack(">IHHIII", 0, model.inf_flags, 0xFFFF,
                                           total_groups, len(pos_u), 0x18)
    for t, i in hier:
        inf += struct.pack(">HH", t, i)
    inf_chunk = _chunk(b"INF1", bytes(inf[8:]))

    chunks = [inf_chunk, vtx_chunk, evp_chunk, drw_chunk, jnt_chunk, shp_chunk, mat_chunk, tex_chunk]
    total = 32 + sum(len(c) for c in chunks)
    out = bytearray(b"J3D2bmd3")
    out += struct.pack(">II", total, len(chunks)) + model.sub_version
    for c in chunks:
        out += c
    validate_engine_layout(bytes(out))
    return bytes(out)


def _build_tex1(textures):
    n = len(textures)
    names = write_string_table([t.name for t in textures])
    hdr_off = 0x20
    name_off = hdr_off + 0x20 * n
    data_start = align(name_off + len(names), 32)
    blobs = []
    headers = bytearray()
    pos = data_start
    for i, t in enumerate(textures):
        hpos = hdr_off + 0x20 * i
        pal_enabled, pal_fmt, pal_cnt, pal_pos = 0, 0, 0, 0
        external = bool(getattr(t, "external", False))
        if external:                       # placeholder: header only, data offset stays 0
            fmt = t.format
            h, w = t.rgba.shape[:2]
            data, mips = b"", max(1, t.mip_count)
            pal_enabled, pal_fmt, pal_cnt = t.pal_enabled, t.pal_format, t.pal_count
        elif t.raw_data is not None:
            fmt = t.format
            h, w = t.rgba.shape[:2]
            data, mips = t.raw_data, t.mip_count
            pal_enabled, pal_fmt, pal_cnt = t.pal_enabled, t.pal_format, t.pal_count
        else:
            if t.rgba is None:
                raise J3DError(f"Texture '{t.name}' has no pixel data")
            fmt = t.format
            h, w = t.rgba.shape[:2]
            data, mips = gx.encode_with_mips(fmt, t.rgba, t.mip_count)
        max_lod = (mips - 1) * 8
        if t.raw_data is not None and t.raw_palette:
            pal_pos = align(pos + len(data), 32)
        headers += struct.pack(">BBHHBBBBHI4BBBbbBBhI", fmt, t.transparency, w, h, t.wrap_s,
                               t.wrap_t, pal_enabled, pal_fmt, pal_cnt,
                               (pal_pos - hpos) if pal_pos else 0, *t.flags10, t.min_filter,
                               t.mag_filter, min(max(t.min_lod, -128), 127),
                               min(max(t.max_lod if (t.raw_data is not None or external) else max_lod, -128), 127),
                               mips, 0, t.lod_bias, 0 if external else pos - hpos)
        if external:
            continue
        blobs.append((pos, data))
        end = pos + len(data)
        if pal_pos:
            blobs.append((pal_pos, t.raw_palette))
            end = pal_pos + len(t.raw_palette)
        pos = align(end, 32)
    body = bytearray(b"TEX1") + struct.pack(">IHHII", 0, n, 0xFFFF, hdr_off, name_off)
    body += b"\0" * (hdr_off - len(body))
    body += headers
    body += names
    body += b"\0" * (data_start - len(body))
    for p, data in blobs:
        body += b"\0" * (p - len(body))
        body += data
    chunk = _chunk(b"TEX1", bytes(body[8:]))
    return chunk, {t.key: i for i, t in enumerate(textures)}


def validate_engine_layout(data):
    """Check the invariants the game's loader relies on; raises J3DError when violated."""
    from . import model as _m
    b = _m.parse(data)
    ne = len(b.envelopes)
    entries = len(b.drw) - ne                     # J3DModelLoader::readDraw: mEntryNum
    if entries <= 0:
        raise J3DError("DRW1 has no draw matrices")
    first_weighted = next((i for i in range(entries) if b.drw[i][0] == 1), entries)
    if any(b.drw[i][0] == 0 for i in range(first_weighted, len(b.drw))):
        raise J3DError("DRW1: rigid entries must precede all weighted entries")
    if entries - first_weighted != ne:
        raise J3DError(f"DRW1: expected {ne} weighted draw matrices, found {entries - first_weighted}")
    for k in range(ne):
        if b.drw[first_weighted + k] != (1, k):
            raise J3DError("DRW1: weighted entries must list the envelopes in order")
    for s in b.shapes:
        for table, _start, _size in s.groups:
            for v in table:
                if v != 0xFFFF and v >= entries:
                    raise J3DError(f"Shape {s.index} references draw matrix {v} >= {entries}")
    for i in range(first_weighted):
        if b.drw[i][1] >= len(b.joints):
            raise J3DError("DRW1 references a missing joint")
