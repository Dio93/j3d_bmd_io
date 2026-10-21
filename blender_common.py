"""Shared helpers for the Blender import / export operators."""

import math

import numpy as np
import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import PropertyGroup

from .j3d import gx_texture as gx
from .j3d import material as mat3

# --- coordinate systems -------------------------------------------------------
# J3D: +Y up, +Z forward.  Blender: +Z up, -Y forward.   p_blender = C @ p_j3d
C = np.array([[1, 0, 0, 0],
              [0, 0, -1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1]], dtype=np.float64)
C_INV = C.T.copy()

# Optional bone-frame correction: Blender bones point along their local +Y while
# Maya/J3D skeletons usually run along the joint's +X axis.  v_joint = R_FIX @ v_bone
R_FIX = np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64)
R_FIX_INV = R_FIX.T.copy()

BONE_AXIS_ITEMS = [
    ("X", "X-forward (nicer bones)", "Bone Y axis follows the joint X axis, as in Maya skeletons"),
    ("NONE", "Joint axes (exact)", "Bone axes are identical to the J3D joint axes"),
]

FORMAT_ITEMS = [("AUTO", "Auto", "Keep the imported format, or choose CMPR / RGB5A3 automatically")] + [
    (n, n, "") for n in ("CMPR", "RGB5A3", "RGBA32", "RGB565", "IA8", "IA4", "I8", "I4")
]

RENDER_ITEMS = [
    ("AUTO", "Auto (keep imported)", "Keep imported material settings; derive them for new materials"),
    (mat3.RENDER_OPAQUE, "Opaque", "No transparency"),
    (mat3.RENDER_CUTOUT, "Cutout (alpha test)", "Pixels below the alpha reference are discarded"),
    (mat3.RENDER_TRANSLUCENT, "Translucent (alpha blend)", "Alpha blended, no depth write"),
]


class BMDMaterialProps(PropertyGroup):
    render_mode: EnumProperty(name="Render Mode", items=RENDER_ITEMS, default="AUTO",
                              description="How the game renders this material")
    alpha_ref: IntProperty(name="Alpha Reference", default=128, min=0, max=255,
                           description="Alpha test threshold for the Cutout mode")


class BMDImageProps(PropertyGroup):
    format: EnumProperty(name="GX Format", items=FORMAT_ITEMS, default="AUTO",
                         description="Texture format used when exporting this image")
    mip_count: IntProperty(name="Mip Levels", default=0, min=0, max=12,
                           description="Number of mip levels (0 = automatic / keep imported)")
    external: BoolProperty(name="Runtime Texture", default=False,
                           description="Placeholder without image data: the game supplies the picture at "
                                       "runtime (data offset 0 in the file). The pixels shown in Blender "
                                       "are meaningless; untick to export this image as a normal texture")


def axis_fix(arm_obj):
    """Bone frame correction that was used when the armature was imported."""
    return R_FIX if arm_obj.get("bmd_bone_axis", "X") == "X" else np.eye(4)


def j3d_local_to_blender(L, s, fix, is_root=False):
    """J3D joint-local matrix -> Blender bone-local (rest-relative) matrix.

    ``s`` = Blender units per J3D unit, ``fix`` = bone frame correction (R_FIX or identity).
    For child bones the parent's bone frame cancels the left ``fix``; roots are relative to
    the armature, which has no such frame, so they keep it out.
    """
    M = C @ L @ C_INV @ fix
    if not is_root:
        M = fix.T @ M
    M[:3, 3] *= s
    return M


def blender_local_to_j3d(M, s, fix, is_root=False):
    """Inverse of :func:`j3d_local_to_blender`."""
    M = M.copy()
    M[:3, 3] /= s
    if not is_root:
        M = fix @ M
    return C_INV @ M @ fix.T @ C


def trs_matrix(scale, euler, trans):
    """4x4 matrix T * Rz*Ry*Rx * S (J3D joint convention)."""
    from .j3d.model import euler_to_matrix
    M = np.eye(4)
    M[:3, :3] = euler_to_matrix(*euler) * np.asarray(scale, dtype=np.float64)[None, :]
    M[:3, 3] = trans
    return M


def matrix_to_quat(R):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        k = math.sqrt(t + 1.0) * 2
        q = (0.25 * k, (R[2, 1] - R[1, 2]) / k, (R[0, 2] - R[2, 0]) / k, (R[1, 0] - R[0, 1]) / k)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        k = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / k, 0.25 * k, (R[0, 1] + R[1, 0]) / k, (R[0, 2] + R[2, 0]) / k)
    elif R[1, 1] > R[2, 2]:
        k = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / k, (R[0, 1] + R[1, 0]) / k, 0.25 * k, (R[1, 2] + R[2, 1]) / k)
    else:
        k = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / k, (R[0, 2] + R[2, 0]) / k, (R[1, 2] + R[2, 1]) / k, 0.25 * k)
    q = np.array(q)
    return q / np.linalg.norm(q)


def decompose(M):
    """4x4 matrix -> (location, quaternion wxyz, scale)."""
    loc = M[:3, 3].copy()
    scale = np.linalg.norm(M[:3, :3], axis=0)
    scale[scale == 0] = 1.0
    R = M[:3, :3] / scale[None, :]
    return loc, matrix_to_quat(R), scale


def to_np(m):
    return np.array([list(r) for r in m], dtype=np.float64)


def orthonormalize(M):
    """Remove scale from the 3x3 part of a 4x4 matrix (column normalisation)."""
    R = M.copy()
    for i in range(3):
        n = np.linalg.norm(R[:3, i])
        if n > 1e-12:
            R[:3, i] /= n
    return R


def point_to_blender(p, scale):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1) * scale


def point_from_blender(p, scale):
    p = np.asarray(p, dtype=np.float64) * scale  # export scale = J3D units per Blender unit
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)


def vec_from_blender(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def texture_key(image):
    """Unique key of an image inside the exported file."""
    key = image.get("bmd_key")
    if key:
        return str(key)
    return clean_texture_name(image)


def clean_texture_name(image):
    name = image.get("bmd_name")
    if name:
        return str(name)
    name = image.name
    low = name.lower()
    for ext in (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".dds", ".webp"):
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def register_classes():
    bpy.utils.register_class(BMDMaterialProps)
    bpy.utils.register_class(BMDImageProps)
    from bpy.props import PointerProperty
    bpy.types.Material.bmd = PointerProperty(type=BMDMaterialProps)
    bpy.types.Image.bmd = PointerProperty(type=BMDImageProps)


def unregister_classes():
    del bpy.types.Image.bmd
    del bpy.types.Material.bmd
    bpy.utils.unregister_class(BMDImageProps)
    bpy.utils.unregister_class(BMDMaterialProps)
