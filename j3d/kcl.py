"""Reader / writer for the collision format used by Twilight Princess stages (``room.kcl``).

The file is a big-endian "KCL" with prisms:

* a table of positions (vertex 1 of every triangle),
* a table of unit normals (face normals and three edge normals per triangle),
* a prism table (``height`` + indices into the two tables + attribute index),
* an octree of 16-bit prism-index lists used for every spatial query.

The attribute index of a prism refers to an entry of the companion ``room.plc`` file (20 bytes per
entry).  This module never edits the PLC: attributes are plain integers.

Layout facts (derived from ``dBgWKCol`` in the Dusklight source):

* prism index 0 is a dummy entry; real prisms are numbered from 1,
* triangle vertices are rebuilt as ``v1 = P``, ``v3 = P + (N x A) * h / ((N x A) . C)``,
  ``v2 = P + (B x N) * h / ((B x N) . C)``,
* octree entries are signed: bit 31 set = leaf (offset of a u16 list, relative to the table that
  holds the entry), otherwise the offset of a child table of 8 entries (also relative),
* lists are read with a pre-increment: the stored offset points at the u16 *before* the first
  index and the list ends at the first 0.
"""

import struct

import numpy as np

from .common import J3DError

HEADER_SIZE = 0x38
MIN_LEAF_SHIFT = 2        # smallest leaf the original files use: 4 x 4 x 4 units
MAX_LIST = 16             # split leaves that contain more triangles than this
MAX_PRISMS = 12000        # dBgWKCol::create asserts (block - prism) / 0xC <= 0x4000 in debug builds
EDGE_MARGIN = 2.0         # triangles are put into every cell within this distance


class KclData:
    """Triangle soup plus the header values needed to write the file again."""

    def __init__(self):
        self.tris = np.zeros((0, 3, 3), np.float64)   # (T, 3, 3): v1, v2, v3 per triangle
        self.attrs = np.zeros(0, np.int32)             # attribute index per triangle
        self.thickness = 1.0
        self.area_min = (0.0, 0.0, 0.0)
        self.area_size = (1, 1, 1)                     # powers of two
        self.block_shift = 13
        self.skipped = 0                               # degenerate prisms dropped while reading


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def parse(data):
    """Parse a KCL file into a :class:`KclData`."""
    if len(data) < HEADER_SIZE:
        raise J3DError("KCL file is too small")
    o_pos, o_nrm, o_pri, o_blk = struct.unpack_from(">4I", data, 0)
    if not (HEADER_SIZE <= o_pos <= o_nrm <= o_pri <= o_blk <= len(data)):
        raise J3DError("KCL header offsets are inconsistent (is this a Twilight Princess KCL?)")
    thickness, = struct.unpack_from(">f", data, 0x10)
    amin = struct.unpack_from(">3f", data, 0x14)
    mx, my, mz, bshift, _xs, _xys = struct.unpack_from(">6I", data, 0x20)
    size = tuple(((~m) & 0xFFFFFFFF) + 1 for m in (mx, my, mz))

    n_pos = (o_nrm - o_pos) // 12
    n_nrm = (o_pri + 16 - o_nrm) // 12              # the last normals may overlap dummy prism 0
    n_prism = (o_blk - o_pri) // 16 - 1             # without the dummy entry
    pos = np.frombuffer(data, ">f4", n_pos * 3, o_pos).reshape(-1, 3).astype(np.float64)
    nrm = np.frombuffer(data, ">f4", n_nrm * 3, o_nrm).reshape(-1, 3).astype(np.float64)
    rec = np.frombuffer(data, np.dtype([("h", ">f4"), ("p", ">u2"), ("f", ">u2"), ("a", ">u2"),
                                         ("b", ">u2"), ("c", ">u2"), ("att", ">u2")]),
                        n_prism, o_pri + 16)
    P = pos[rec["p"]]
    N, A, B, C = (nrm[rec[k]] for k in ("f", "a", "b", "c"))
    h = rec["h"].astype(np.float64)
    cA, cB = np.cross(N, A), np.cross(B, N)
    dA, dB = np.einsum("ij,ij->i", cA, C), np.einsum("ij,ij->i", cB, C)
    ok = (np.abs(dA) > 1e-9) & (np.abs(dB) > 1e-9)
    with np.errstate(all="ignore"):
        v3 = P + cA * (h / dA)[:, None]
        v2 = P + cB * (h / dB)[:, None]
    tris = np.stack([P, v2, v3], axis=1)
    ok &= np.isfinite(tris).all(axis=(1, 2))
    out = KclData()
    out.tris = tris[ok]
    out.attrs = rec["att"][ok].astype(np.int32)
    out.skipped = int((~ok).sum())
    out.thickness = float(thickness)
    out.area_min = tuple(float(v) for v in amin)
    out.area_size = size
    out.block_shift = int(bshift)
    return out


def read_plc_codes(data):
    """Return the PLC entries as a list of 5-tuples of u32 (``code0`` .. ``code4``)."""
    if data[:4] != b"SPLC":
        raise J3DError("Not a PLC file (missing 'SPLC' magic)")
    size, num = struct.unpack_from(">HH", data, 4)
    if size != 0x14:
        raise J3DError(f"Unexpected PLC entry size {size:#x}")
    return [struct.unpack_from(">5I", data, 8 + 20 * i) for i in range(num)]


# --------------------------------------------------------------------------
# triangle geometry
# --------------------------------------------------------------------------

def _prism_normals(tris):
    """Face normal, three edge normals and the prism height for every triangle."""
    v1, v2, v3 = tris[:, 0], tris[:, 1], tris[:, 2]
    e2, e3, e23 = v2 - v1, v3 - v1, v3 - v2
    N = _unit(np.cross(e2, e3))
    # edge normals lie in the triangle plane, perpendicular to their edge
    A = _unit(np.cross(e3, N))                       # edge v1-v3, pointing away from v2
    A = np.where((np.einsum("ij,ij->i", A, e2) > 0)[:, None], -A, A)
    B = _unit(np.cross(e2, N))                       # edge v1-v2, pointing away from v3
    B = np.where((np.einsum("ij,ij->i", B, e3) > 0)[:, None], -B, B)
    C = _unit(np.cross(e23, N))                      # edge v2-v3, pointing away from v1
    C = np.where((np.einsum("ij,ij->i", C, -e2) > 0)[:, None], -C, C)
    h = np.einsum("ij,ij->i", C, e2)
    return N, A, B, C, h


def triangle_areas(tris):
    return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)


# --------------------------------------------------------------------------
# octree
# --------------------------------------------------------------------------

def _tri_box_overlap(tris, centre, half):
    """Vectorised triangle / axis-aligned box overlap test (separating axis theorem)."""
    v = tris - centre                                  # (T, 3, 3)
    edges = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]], axis=1)  # (T,3,3)
    sep = np.zeros(len(tris), bool)
    # 3 box axes
    sep |= (v.min(axis=1) > half).any(axis=1) | (v.max(axis=1) < -half).any(axis=1)
    # triangle plane
    n = np.cross(edges[:, 0], edges[:, 1])
    d = np.einsum("ij,ij->i", n, v[:, 0])
    r = (np.abs(n) * half).sum(axis=1)
    sep |= np.abs(d) > r
    # 9 cross-product axes
    axes_box = np.eye(3)
    for e in range(3):
        for a in range(3):
            ax = np.cross(edges[:, e], axes_box[a])    # (T,3)
            p = np.einsum("tvj,tj->tv", v, ax)
            r = (np.abs(ax) * half).sum(axis=1)
            sep |= (p.min(axis=1) > r) | (p.max(axis=1) < -r)
    return ~sep


class _Leaf:
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items


class _Node:
    __slots__ = ("children",)

    def __init__(self, children):
        self.children = children


def _build_cell(tris, tri_min, tri_max, idx, origin, shift):
    """Recursively build the octree below one cell. ``idx`` are candidate triangle indices."""
    size = 1 << shift
    centre = origin + size / 2.0
    half = np.full(3, size / 2.0 + EDGE_MARGIN)
    if len(idx):
        idx = idx[_tri_box_overlap(tris[idx], centre, half)]
    if len(idx) <= MAX_LIST or shift <= MIN_LEAF_SHIFT:
        return _Leaf(tuple(int(i) + 1 for i in idx))       # prism indices start at 1
    kids = []
    cs = shift - 1
    for c in range(8):
        o = origin + np.array([(c & 1), (c >> 1) & 1, (c >> 2) & 1], np.float64) * (1 << cs)
        # cheap AABB pre-filter before the exact test
        lo, hi = o - EDGE_MARGIN, o + (1 << cs) + EDGE_MARGIN
        sel = idx[(tri_max[idx] >= lo).all(axis=1) & (tri_min[idx] <= hi).all(axis=1)]
        kids.append(_build_cell(tris, tri_min, tri_max, sel, o, cs))
    return _Node(kids)


def _choose_area(tris, template):
    """Area origin, power-of-two sizes and root block shift."""
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    pad = EDGE_MARGIN + 1.0
    if template is not None:
        tmin = np.array(template.area_min, np.float64)
        tsize = np.array(template.area_size, np.float64)
        if (lo - pad >= tmin).all() and (hi + pad < tmin + tsize).all():
            return tmin, [int(s) for s in tsize], int(template.block_shift)
    amin = np.floor(lo - pad)
    extent = hi + pad - amin
    size = [1 << max(int(np.ceil(np.log2(max(e, 2.0)))), 4) for e in extent]
    shift = max(2, min(13, int(np.log2(min(size)))))
    return amin, size, shift


def write(tris, attrs, template=None):
    """Build a KCL file from triangles ``(T,3,3)`` and attribute indices ``(T,)``.

    ``template`` (a :class:`KclData` read from the original file) keeps the original area and
    block size when the new geometry still fits into them.
    """
    tris = np.asarray(tris, np.float64).reshape(-1, 3, 3)
    attrs = np.asarray(attrs, np.int64)
    if len(tris) != len(attrs):
        raise J3DError("Triangle and attribute counts differ")
    keep = triangle_areas(tris) > 1e-6
    tris, attrs = tris[keep], attrs[keep]
    n = len(tris)
    if n == 0:
        raise J3DError("No usable triangles for the collision mesh")
    if n > MAX_PRISMS:
        raise J3DError(f"Collision mesh has {n} triangles; the game supports at most about "
                       f"{MAX_PRISMS}. Decimate it or split the mesh.")
    if attrs.min() < 0 or attrs.max() > 0xFFFF:
        raise J3DError("Attribute indices must be in 0..65535")
    # float32 positions: use the rounded values everywhere so the octree matches what the game reads
    tris = tris.astype(np.float32).astype(np.float64)

    N, A, B, C, H = _prism_normals(tris)

    # --- tables -----------------------------------------------------------------------
    def dedupe(vecs, decimals=6):
        table, lookup, out = [], {}, np.empty(len(vecs), np.int64)
        keys = vecs.astype(np.float32).astype(np.float64)
        if decimals is not None:
            keys = np.round(keys, decimals)
        for i, k in enumerate(map(tuple, keys)):
            j = lookup.get(k)
            if j is None:
                j = lookup[k] = len(table)
                table.append(vecs[i])
            out[i] = j
        return np.asarray(table, np.float64).reshape(-1, 3), out

    pos_t, pos_i = dedupe(tris[:, 0], 4)
    nrm_all = np.concatenate([N, A, B, C])
    nrm_t, nrm_i = dedupe(nrm_all, None)         # exact float32 match only: keeps vertices precise
    fi, ai, bi, ci = nrm_i[:n], nrm_i[n:2 * n], nrm_i[2 * n:3 * n], nrm_i[3 * n:]
    if len(pos_t) > 0xFFFF or len(nrm_t) > 0xFFFF:
        raise J3DError("Too many unique positions/normals for 16-bit indices; simplify the mesh")

    # --- octree -----------------------------------------------------------------------
    amin, size, bshift = _choose_area(tris, template)
    rel = tris - amin
    tri_min, tri_max = rel.min(axis=1), rel.max(axis=1)
    nb = [max(1, s >> bshift) for s in size]
    xs = int(np.log2(nb[0]))
    xys = xs + int(np.log2(nb[1]))
    roots = []
    for z in range(nb[2]):
        for y in range(nb[1]):
            for x in range(nb[0]):
                org = np.array([x, y, z], np.float64) * (1 << bshift)
                lo, hi = org - EDGE_MARGIN, org + (1 << bshift) + EDGE_MARGIN
                cand = np.nonzero((tri_max >= lo).all(axis=1) & (tri_min <= hi).all(axis=1))[0]
                roots.append(_build_cell(rel, tri_min, tri_max, cand, org, bshift))
    # roots were generated in (z, y, x) order which is exactly the table index order

    # lay out all tables (root table first, then child tables breadth first)
    tables = []              # (entries list, position)
    order = [roots]
    queue = [roots]
    pos = 0
    table_pos = {}
    while queue:
        nodes = queue.pop(0)
        table_pos[id(nodes)] = pos
        pos += 4 * len(nodes)
        for nd in nodes:
            if isinstance(nd, _Node):
                queue.append(nd.children)
    lists_start = (pos + 3) & ~3
    stream = bytearray(struct.pack(">HH", 0, 0))      # empty list at offset 0 (p[0]=0, p[1]=0)
    list_ptr = {(): 0}

    def list_pointer(items):
        p = list_ptr.get(items)
        if p is None:
            p = len(stream) - 2                          # the 0 that ends the previous list
            stream.extend(struct.pack(f">{len(items) + 1}H", *items, 0))
            list_ptr[items] = p
        return p

    body = bytearray(lists_start)
    queue = [roots]
    while queue:
        nodes = queue.pop(0)
        base = table_pos[id(nodes)]
        for i, nd in enumerate(nodes):
            if isinstance(nd, _Leaf):
                val = 0x80000000 | (lists_start + list_pointer(nd.items) - base)
            else:
                val = table_pos[id(nd.children)] + 0  # filled in below once positions are known
                queue.append(nd.children)
                val = val - base
            struct.pack_into(">I", body, base + 4 * i, val)
    body += stream
    if len(body) % 4:
        body += b"\0" * (4 - len(body) % 4)

    # --- assemble file ----------------------------------------------------------------
    o_pos = HEADER_SIZE
    o_nrm = o_pos + 12 * len(pos_t)
    o_pri = o_nrm + 12 * len(nrm_t)
    o_blk = o_pri + 16 * (n + 1)
    thickness = template.thickness if template is not None else 1.0
    out = bytearray(struct.pack(">4If3f6I", o_pos, o_nrm, o_pri, o_blk, thickness, *amin.tolist(),
                                *(((~(s - 1)) & 0xFFFFFFFF) for s in size), bshift, xs, xys))
    out += pos_t.astype(">f4").tobytes()
    out += nrm_t.astype(">f4").tobytes()
    out += b"\0" * 16                                  # prism 0 is a dummy
    rec = np.zeros(n, np.dtype([("h", ">f4"), ("p", ">u2"), ("f", ">u2"), ("a", ">u2"),
                                ("b", ">u2"), ("c", ">u2"), ("att", ">u2")]))
    rec["h"], rec["p"], rec["f"] = H, pos_i, fi
    rec["a"], rec["b"], rec["c"], rec["att"] = ai, bi, ci, attrs
    out += rec.tobytes()
    out += body
    # the octree offsets above are relative to their own table, so nothing else to patch
    verify(bytes(out), tris, attrs)
    return bytes(out)


# --------------------------------------------------------------------------
# verification / queries (also used by the tests)
# --------------------------------------------------------------------------

def _tables(data):
    o_blk = struct.unpack_from(">I", data, 0x0C)[0]
    bshift, xs, xys = struct.unpack_from(">3I", data, 0x2C)
    return o_blk, bshift, xs, xys


def cell_lists(data):
    """Yield ``(origin, shift, [prism indices])`` for every leaf of the octree."""
    o_blk, bshift, xs, xys = _tables(data)
    mx, my, mz = struct.unpack_from(">3I", data, 0x20)
    nb = [(((~m) & 0xFFFFFFFF) >> bshift) + 1 for m in (mx, my, mz)]

    def lst(p):
        out, p = [], p + 2
        while True:
            v = struct.unpack_from(">H", data, p)[0]
            if v == 0:
                return out
            out.append(v)
            p += 2

    def walk(base, entry, shift, org):
        e = struct.unpack_from(">I", data, entry)[0]
        if e & 0x80000000:
            yield org, shift, lst(base + (e & 0x7FFFFFFF))
            return
        nb_ = base + e
        for c in range(8):
            cs = shift - 1
            o = (org[0] + ((c & 1) << cs), org[1] + (((c >> 1) & 1) << cs), org[2] + (((c >> 2) & 1) << cs))
            yield from walk(nb_, nb_ + 4 * c, cs, o)

    for z in range(nb[2]):
        for y in range(nb[1]):
            for x in range(nb[0]):
                i = (z << xys) | (y << xs) | x
                yield from walk(o_blk, o_blk + 4 * i, bshift, (x << bshift, y << bshift, z << bshift))


def verify(data, tris, attrs, sample=400):
    """Sanity checks: re-parse the geometry and make sure every triangle sits in its cells."""
    back = parse(data)
    if len(back.tris) != len(tris):
        raise J3DError(f"KCL self-check failed: {len(back.tris)} triangles read back, {len(tris)} written")
    err = float(np.abs(back.tris - tris).max())
    scale = max(1.0, float(np.abs(tris).max()))
    if err > 1e-3 * scale:
        raise J3DError(f"KCL self-check failed: triangle vertices deviate by {err:.4f}")
    if not np.array_equal(back.attrs, np.asarray(attrs)):
        raise J3DError("KCL self-check failed: attribute indices differ")
    amin = np.array(back.area_min)
    rel = tris - amin
    seen = np.zeros(len(tris) + 1, bool)
    for _org, _shift, items in cell_lists(data):
        for i in items:
            seen[i] = True
    if not seen[1:].all():
        raise J3DError("KCL self-check failed: some triangles are not referenced by the octree")
    # random-point test: every triangle must be listed in the leaf containing its centroid
    rng = np.random.default_rng(1)
    pick = rng.choice(len(tris), min(sample, len(tris)), replace=False)
    leaves = list(cell_lists(data))
    for t in pick:
        c = rel[t].mean(axis=0)
        for org, shift, items in leaves:
            if all(org[k] <= c[k] < org[k] + (1 << shift) for k in range(3)):
                if int(t) + 1 not in items:
                    raise J3DError("KCL self-check failed: triangle missing from the cell of its centroid")
                break
