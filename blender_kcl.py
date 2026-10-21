"""Blender glue for Twilight Princess collision files (``room.kcl`` + ``room.plc``).

Every collision attribute becomes a material called ``KCL_<n>`` (``n`` = attribute index, i.e. the
row of the companion ``room.plc``).  The material of a face *is* its attribute, so painting faces
with a material changes their collision type.  New faces without a ``KCL_<n>`` material get the
export option "Default Attribute".
"""

import colorsys
import os
import re

import bpy
import numpy as np

from . import blender_common as bc
from .j3d import kcl
from .j3d.common import J3DError

_NAME = re.compile(r"^KCL_(\d+)")


class KclExportError(Exception):
    pass


def _colour(index):
    r, g, b = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.55, 0.85)
    return r, g, b, 1.0


_THROUGH_BITS = [(14, "object"), (15, "camera"), (16, "link"), (17, "arrow"), (18, "hookshot-stick"),
                 (19, "boomerang"), (20, "rope"), (21, "horse-no-entry"), (22, "shadow"), (23, "bomb")]


def _describe(code):
    """Readable summary of a PLC entry (field layout from sBgPc / dBgPc in the Dusklight source)."""
    c0, c1, c2, c3, c4 = code
    ignored = [n for b, n in _THROUGH_BITS if c0 >> b & 1]
    ignored += [n for b, n in ((24, "iron-ball"), (25, "attack")) if c1 >> b & 1]
    return {
        "ignored_by": ", ".join(ignored) or "-",   # these query types pass through the polygon
        "exit": c0 & 0x3F,
        "ground_code": c1 >> 19 & 0x1F,
        "wall_code": c1 >> 8 & 0xF,
        "att0": c1 >> 12 & 0xF,
        "att1": c1 >> 16 & 0x7,
        "special": c0 >> 24 & 0xF,
        "room_cam": c2 >> 8 & 0xFF,
        "room_path": c2 >> 16 & 0xFF,
        "room_info": c4 & 0xFF,
    }


def _plc_path(kcl_path, plc_path):
    if plc_path and os.path.isfile(plc_path):
        return plc_path
    cand = os.path.join(os.path.dirname(kcl_path), os.path.splitext(os.path.basename(kcl_path))[0] + ".plc")
    return cand if os.path.isfile(cand) else ""


def import_kcl(context, filepath, scale=0.01, plc_file="", report=None):
    with open(filepath, "rb") as f:
        data = kcl.parse(f.read())
    codes = []
    plc = _plc_path(filepath, plc_file)
    if plc:
        try:
            with open(plc, "rb") as f:
                codes = kcl.read_plc_codes(f.read())
        except J3DError as exc:
            if report:
                report({"WARNING"}, f"PLC ignored: {exc}")

    # J3D (+Y up) -> Blender (+Z up); the rotation keeps the winding order and the normals
    pts = data.tris.reshape(-1, 3) @ bc.C[:3, :3].T * scale
    keys = np.round(pts / max(scale, 1e-9) * 100.0).astype(np.int64)      # weld at 0.01 J3D units
    _u, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    verts = pts[first]
    faces = inverse.reshape(-1, 3)
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces, attrs = faces[good], data.attrs[good]          # welding can collapse sliver triangles

    base = os.path.splitext(os.path.basename(filepath))[0]
    mesh = bpy.data.meshes.new(base)
    mesh.from_pydata(verts.astype(np.float32).tolist(), [], faces.tolist())

    used = sorted(set(int(a) for a in attrs))
    slot_of = {a: i for i, a in enumerate(used)}
    for a in used:
        mat = bpy.data.materials.new(f"KCL_{a}")
        mat.diffuse_color = _colour(a)
        mat["kcl_attribute"] = a
        if a < len(codes):
            mat["plc_codes"] = " ".join(f"{c:08X}" for c in codes[a])
            for k, v in _describe(codes[a]).items():
                mat[f"plc_{k}"] = v
        mesh.materials.append(mat)
    mesh.polygons.foreach_set("material_index", np.array([slot_of[int(a)] for a in attrs], np.int32))
    mesh.update(calc_edges=True)
    mesh.validate()

    obj = bpy.data.objects.new(base, mesh)
    context.collection.objects.link(obj)
    for o in list(context.view_layer.objects.selected):
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    obj.display_type = "SOLID"
    obj["kcl_source"] = os.path.abspath(filepath)
    obj["kcl_scale"] = float(scale)
    obj["kcl_area_min"] = list(data.area_min)
    obj["kcl_area_size"] = list(data.area_size)
    obj["kcl_block_shift"] = int(data.block_shift)
    obj["kcl_thickness"] = float(data.thickness)
    if plc:
        obj["kcl_plc"] = os.path.abspath(plc)
    if report:
        extra = f", {data.skipped} degenerate prisms dropped" if data.skipped else ""
        report({"INFO"}, f"Imported {len(faces)} collision triangles with {len(used)} attributes{extra}")
    return obj


def _attribute_of(obj, mesh, poly_materials, default_attr):
    """Attribute index per polygon from the KCL_<n> material of its slot."""
    slot_attr = []
    for slot in obj.material_slots:
        m = slot.material
        match = _NAME.match(m.name) if m else None
        if match:
            slot_attr.append(int(match.group(1)))
        elif m is not None and "kcl_attribute" in m:
            slot_attr.append(int(m["kcl_attribute"]))
        else:
            slot_attr.append(default_attr)
    out = np.full(len(poly_materials), default_attr, np.int64)
    for i, s in enumerate(poly_materials):
        if 0 <= s < len(slot_attr):
            out[i] = slot_attr[s]
    return out


def export_kcl(context, filepath, scale=100.0, use_selection=True, apply_modifiers=True,
               default_attr=0, keep_area=True, report=None):
    objs = [o for o in (context.selected_objects if use_selection else context.scene.objects)
            if o.type == "MESH"]
    if not objs:
        raise KclExportError("No mesh objects to export (select the collision mesh or untick 'Selected Only')")
    depsgraph = context.evaluated_depsgraph_get()
    tris, attrs = [], []
    template = None
    for obj in objs:
        eval_obj = obj.evaluated_get(depsgraph) if apply_modifiers else obj
        mesh = eval_obj.to_mesh() if apply_modifiers else obj.data
        try:
            mesh.calc_loop_triangles()
            nt = len(mesh.loop_triangles)
            if nt == 0:
                continue
            vidx = np.empty(nt * 3, np.int32)
            mesh.loop_triangles.foreach_get("vertices", vidx)
            pidx = np.empty(nt, np.int32)
            mesh.loop_triangles.foreach_get("polygon_index", pidx)
            co = np.empty(len(mesh.vertices) * 3, np.float32)
            mesh.vertices.foreach_get("co", co)
            mat = np.array(obj.matrix_world, np.float64)
            co = co.reshape(-1, 3).astype(np.float64) @ mat[:3, :3].T + mat[:3, 3]
            co = co @ bc.C_INV[:3, :3].T * scale
            tris.append(co[vidx].reshape(-1, 3, 3))
            polymat = np.empty(len(mesh.polygons), np.int32)
            mesh.polygons.foreach_get("material_index", polymat)
            attrs.append(_attribute_of(obj, mesh, polymat, default_attr)[pidx])
        finally:
            if apply_modifiers:
                eval_obj.to_mesh_clear()
        if template is None and keep_area and "kcl_area_min" in obj:
            template = kcl.KclData()
            template.area_min = tuple(obj["kcl_area_min"])
            template.area_size = tuple(int(v) for v in obj["kcl_area_size"])
            template.block_shift = int(obj["kcl_block_shift"])
            template.thickness = float(obj.get("kcl_thickness", 1.0))
    if not tris:
        raise KclExportError("The selected meshes contain no faces")
    tris = np.concatenate(tris)
    attrs = np.concatenate(attrs)
    # the PLC is never rewritten: warn about attributes it does not contain
    plc = ""
    for obj in objs:
        if "kcl_plc" in obj:
            plc = obj["kcl_plc"]
            break
    if plc and os.path.isfile(plc):
        try:
            with open(plc, "rb") as f:
                count = len(kcl.read_plc_codes(f.read()))
            over = sorted({int(a) for a in attrs if a >= count})
            if over and report:
                report({"WARNING"}, f"Attributes {over[:8]} do not exist in {os.path.basename(plc)} "
                                    f"({count} entries); the game would read garbage for them")
        except J3DError:
            pass
    blob = kcl.write(tris, attrs, template)
    with open(filepath, "wb") as f:
        f.write(blob)
    if report:
        report({"INFO"}, f"Wrote {len(tris)} collision triangles ({len(blob)} bytes) to "
                         f"{os.path.basename(filepath)}. Keep using the original room.plc.")
