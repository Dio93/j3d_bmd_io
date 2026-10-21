"""J3D joint animations (.bck, chunk ANK1): parsing, sampling and writing.

No Blender dependency.  Rotations are handled in radians (XYZ Euler, R = Rz*Ry*Rx,
the same convention as the JNT1 joint data); all tangents are "value units per frame".
"""

import math
import struct

import numpy as np

from .common import J3DError, align, pad_to, u32

LOOP_ONCE, LOOP_ONCE_RESET, LOOP_LOOP, LOOP_MIRROR_ONCE, LOOP_MIRROR_LOOP = range(5)
LOOP_NAMES = ["ONCE", "ONCE_RESET", "LOOP", "MIRROR_ONCE", "MIRROR_LOOP"]

# Track order inside one joint entry
TRACK_NAMES = ["scaleX", "rotX", "transX", "scaleY", "rotY", "transY", "scaleZ", "rotZ", "transZ"]
KIND_SCALE, KIND_ROT, KIND_TRANS = 0, 1, 2

ROT_UNIT = math.pi / 32768.0  # radians per raw rotation unit (before the 2**angle_mult factor)


class Track:
    """Animation curve of one component. ``keys`` = [(time, value, tangent_in, tangent_out)]."""

    def __init__(self, keys=None):
        self.keys = keys if keys else [(0.0, 0.0, 0.0, 0.0)]

    @property
    def is_constant(self):
        return len(self.keys) == 1


class Bck:
    def __init__(self):
        self.loop_mode = LOOP_LOOP
        self.angle_mult = 0
        self.duration = 1
        self.tracks = []  # tracks[joint][9]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_bck(data):
    data = bytes(data)
    if data[:3] != b"J3D" or data[4:8] != b"bck1":
        raise J3DError("Not a J3D bck1 animation file")
    chunks = u32(data, 12)
    off = 32
    ank = None
    for _ in range(chunks):
        tag = data[off:off + 4]
        size = u32(data, off + 4)
        if tag == b"ANK1":
            ank = off
        off += size
    if ank is None:
        raise J3DError("Missing ANK1 chunk")

    loop, mult, duration, njoints, nscale, nrot, ntrans = struct.unpack_from(">BbHHHHH", data, ank + 8)
    o_joint, o_scale, o_rot, o_trans = struct.unpack_from(">4I", data, ank + 0x14)
    scale = np.frombuffer(data, ">f4", nscale, ank + o_scale).astype(np.float64)
    rot = np.frombuffer(data, ">i2", nrot, ank + o_rot).astype(np.float64)
    trans = np.frombuffer(data, ">f4", ntrans, ank + o_trans).astype(np.float64)
    rot_unit = ROT_UNIT * (2.0 ** mult)

    bck = Bck()
    bck.loop_mode, bck.angle_mult, bck.duration = loop, mult, duration
    for j in range(njoints):
        e = ank + o_joint + j * 0x36
        raw = struct.unpack_from(">27H", data, e)
        joint_tracks = []
        for k in range(9):
            count, index, ttype = raw[k * 3:k * 3 + 3]
            kind = k % 3
            arr = (scale, rot, trans)[kind]
            unit = rot_unit if kind == KIND_ROT else 1.0
            if count == 1:
                joint_tracks.append(Track([(0.0, arr[index] * unit, 0.0, 0.0)]))
                continue
            n = 4 if ttype else 3
            keys = []
            for i in range(count):
                b = index + i * n
                t = arr[b]
                v = arr[b + 1] * unit
                tin = arr[b + 2] * unit
                tout = arr[b + 3] * unit if ttype else tin
                keys.append((float(t), float(v), float(tin), float(tout)))
            joint_tracks.append(Track(keys))
        bck.tracks.append(joint_tracks)
    return bck


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def sample_track(track, frames):
    """Evaluate a track (cubic Hermite, clamped outside the key range) at ``frames``."""
    frames = np.asarray(frames, dtype=np.float64)
    keys = track.keys
    if len(keys) == 1:
        return np.full(frames.shape, keys[0][1])
    t = np.array([k[0] for k in keys])
    v = np.array([k[1] for k in keys])
    tin = np.array([k[2] for k in keys])
    tout = np.array([k[3] for k in keys])
    seg = np.clip(np.searchsorted(t, frames, side="right") - 1, 0, len(keys) - 2)
    t0, t1 = t[seg], t[seg + 1]
    dt = np.maximum(t1 - t0, 1e-9)
    s = np.clip((frames - t0) / dt, 0.0, 1.0)
    s2, s3 = s * s, s * s * s
    out = ((2 * s3 - 3 * s2 + 1) * v[seg] + (-2 * s3 + 3 * s2) * v[seg + 1]
           + (s3 - 2 * s2 + s) * dt * tout[seg] + (s3 - s2) * dt * tin[seg + 1])
    out = np.where(frames <= t[0], v[0], out)
    out = np.where(frames >= t[-1], v[-1], out)
    return out


def sample(bck, frames):
    """Return an array (joints, 9, frames) of sampled values (rotations in radians)."""
    frames = np.asarray(frames, dtype=np.float64)
    out = np.zeros((len(bck.tracks), 9, len(frames)))
    for j, tracks in enumerate(bck.tracks):
        for k, tr in enumerate(tracks):
            out[j, k] = sample_track(tr, frames)
    return out


# --------------------------------------------------------------------------
# curve fitting (used when exporting)
# --------------------------------------------------------------------------

def fit_track(values, tolerance, const_tolerance=None):
    """Build a :class:`Track` from per-frame samples, keeping as few keys as the tolerance allows."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if const_tolerance is None:
        const_tolerance = tolerance * 0.5
    if n < 2 or (v.max() - v.min()) <= const_tolerance:
        return Track([(0.0, float(v[0]), 0.0, 0.0)])
    tg = np.gradient(v)
    keys = [0, n - 1]
    while True:
        err = np.zeros(n)
        for a, b in zip(keys[:-1], keys[1:]):
            if b - a < 2:
                continue
            f = np.arange(a + 1, b)
            s = (f - a) / (b - a)
            dt = b - a
            h = ((2 * s**3 - 3 * s**2 + 1) * v[a] + (-2 * s**3 + 3 * s**2) * v[b]
                 + (s**3 - 2 * s**2 + s) * dt * tg[a] + (s**3 - s**2) * dt * tg[b])
            err[f] = np.abs(h - v[f])
        i = int(err.argmax())
        if err[i] <= tolerance:
            break
        keys.append(i)
        keys.sort()
    return Track([(float(i), float(v[i]), float(tg[i]), float(tg[i])) for i in keys])


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _choose_angle_mult(bck):
    """Smallest non-negative exponent so that every rotation value fits in an s16."""
    hi, lo = 0.0, 0.0
    for joint_tracks in bck.tracks:
        for k in (1, 4, 7):
            for _t, v, tin, tout in joint_tracks[k].keys:
                hi = max(hi, v, tin, tout)
                lo = min(lo, v, tin, tout)
    mult = 0
    while mult < 7:
        unit = ROT_UNIT * (2.0 ** mult)
        if hi / unit <= 32767.4 and lo / unit >= -32768.4:
            break
        mult += 1
    # keep the original precision setting of a parsed file when it still fits
    return max(mult, min(max(bck.angle_mult, 0), 7))


def write_bck(bck):
    """Serialise ``bck`` into a complete J3D1 bck1 file (bytes)."""
    mult = _choose_angle_mult(bck)
    unit = ROT_UNIT * (2.0 ** mult)
    arrays = ([], [], [])
    const_lookup = ({}, {}, {})
    entries = bytearray()

    def encode(kind, x):
        if kind == KIND_ROT:
            return int(max(-32768, min(32767, round(x / unit))))
        return float(np.float32(x))

    for joint_tracks in bck.tracks:
        for k, tr in enumerate(joint_tracks):
            kind = k % 3
            arr = arrays[kind]
            if len(tr.keys) == 1:
                val = encode(kind, tr.keys[0][1])
                idx = const_lookup[kind].get(val)
                if idx is None:
                    idx = len(arr)
                    const_lookup[kind][val] = idx
                    arr.append(val)
                entries += struct.pack(">HHH", 1, idx, 0)
                continue
            double = any(abs(a - b) > 1e-12 for _, _, a, b in tr.keys)
            idx = len(arr)
            for t, v, tin, tout in tr.keys:
                arr.append(int(round(t)) if kind == KIND_ROT else float(t))
                arr.append(encode(kind, v))
                arr.append(encode(kind, tin))
                if double:
                    arr.append(encode(kind, tout))
            entries += struct.pack(">HHH", len(tr.keys), idx, 1 if double else 0)

    o_joint = 0x40
    o_scale = align(o_joint + len(entries), 32)
    scale_b = struct.pack(f">{len(arrays[0])}f", *arrays[0])
    o_rot = align(o_scale + len(scale_b), 32)
    rot_b = struct.pack(f">{len(arrays[1])}h", *arrays[1])
    o_trans = align(o_rot + len(rot_b), 32)
    trans_b = struct.pack(f">{len(arrays[2])}f", *arrays[2])

    chunk = bytearray(b"ANK1")
    chunk += struct.pack(">IBbHHHHH", 0, bck.loop_mode, mult, bck.duration, len(bck.tracks),
                         len(arrays[0]), len(arrays[1]), len(arrays[2]))
    chunk += struct.pack(">4I", o_joint, o_scale, o_rot, o_trans)
    chunk += b"\0" * (o_joint - len(chunk))
    chunk += entries
    for off, raw in ((o_scale, scale_b), (o_rot, rot_b), (o_trans, trans_b)):
        chunk += b"\0" * (off - len(chunk))
        chunk += raw
    pad_to(chunk, 32)
    struct.pack_into(">I", chunk, 4, len(chunk))
    return b"J3D1bck1" + struct.pack(">II", 32 + len(chunk), 1) + b"\xff" * 16 + bytes(chunk)
