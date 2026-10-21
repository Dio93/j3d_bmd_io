"""Collect Blender data into an :class:`ExportModel` and write it as BMD3."""

import base64
import json
import math
import os
import zlib

import bpy
import numpy as np

from . import blender_common as bc
from .j3d import gx_texture as gx
from .j3d import material as mat3
from .j3d import model as j3d
from .j3d import writer


class ExportError(Exception):
    pass


def export_bmd(context, filepath, scale=100.0, use_selection=True, apply_modifiers=False,
               default_format="AUTO", source_file="", allow_missing_textures=False,
               keep_vertex_formats=True, report=None):
    scene_objs = context.selected_objects if use_selection else [
        o for o in context.scene.objects if o.visible_get()]
    mesh_objs = [o for o in scene_objs if o.type == "MESH"]
    arm_obj = next((o for o in scene_objs if o.type == "ARMATURE"), None)
    if arm_obj is None:
        for o in mesh_objs:
            for m in o.modifiers:
                if m.type == "ARMATURE" and m.object:
                    arm_obj = m.object
                    break
            if arm_obj:
                break
    if not mesh_objs:
        raise ExportError("No mesh objects to export (select the meshes, or disable 'Selected Only')")

    mesh_objs.sort(key=lambda o: (o.get("bmd_shape_index", 1e9), o.get("bmd_order", 1e9), o.name))
    model = writer.ExportModel()
    bone_index = _build_joints(model, arm_obj, scale)

    mats = _MaterialCollector(model, default_format)
    if arm_obj is not None and "bmd_textures" in arm_obj:
        mats.preload_original_textures(arm_obj, source_file, allow_missing_textures, report)
    for obj in mesh_objs:
        _export_mesh_object(model, obj, arm_obj, bone_index, mats, scale, apply_modifiers, context)
    mats.finish()
    # keep the original shape order: the game addresses shapes and materials by index
    model.meshes.sort(key=lambda m: (m.sort_key, m.order))

    if arm_obj is not None:
        model.inf_flags = int(arm_obj.get("bmd_inf_flags", 0))
        if keep_vertex_formats and arm_obj.get("bmd_vtx_formats"):
            model.vtx_formats = {int(k): tuple(v) for k, v in json.loads(arm_obj["bmd_vtx_formats"]).items()}
        sv = arm_obj.get("bmd_sub_version")
        if sv:
            model.sub_version = bytes.fromhex(sv)
    data = writer.write(model)
    with open(filepath, "wb") as f:
        f.write(data)
    if report:
        report({"INFO"}, f"Exported {len(model.meshes)} meshes, {len(model.joints)} joints, "
                         f"{len(model.materials)} materials, {len(model.textures)} textures "
                         f"({len(data) // 1024} KiB)")
    return model


# --------------------------------------------------------------------------
# skeleton
# --------------------------------------------------------------------------

def ordered_bones(arm):
    """Bones in J3D joint order: imported bones keep ``bmd_index``, new bones follow depth-first."""
    bones = list(arm.bones)
    known = sorted([b for b in bones if "bmd_index" in b], key=lambda b: b["bmd_index"])
    new = []

    def dfs(b):
        if "bmd_index" not in b:
            new.append(b)
        for c in b.children:
            dfs(c)

    for r in [b for b in bones if b.parent is None]:
        dfs(r)
    return known + [b for b in new if b not in known]


def _build_joints(model, arm_obj, scale):
    """Fill model.joints; returns {bone name: joint index}."""
    if arm_obj is None:
        j = j3d.Joint("root")
        model.joints.append(j)
        return {}
    arm = arm_obj.data
    bones = list(arm.bones)
    if not bones:
        model.joints.append(j3d.Joint("root"))
        return {}

    ordered = ordered_bones(arm)
    index = {b.name: i for i, b in enumerate(ordered)}

    axis_fix = bc.R_FIX if arm_obj.get("bmd_bone_axis", "X") == "X" else np.eye(4)
    fix_inv = axis_fix.T
    world = {}
    for b in ordered:
        Mb = bc.to_np(b.matrix_local)
        Wb = Mb @ fix_inv
        W = bc.C_INV @ Wb @ bc.C
        W[:3, 3] *= scale
        world[b.name] = bc.orthonormalize(W)

    own_same = {}
    for b in ordered:
        stored, rest = b.get("bmd_local"), b.get("bmd_rest")
        cur = bc.to_np(b.matrix_local).reshape(-1)
        own_same[b.name] = bool(stored is not None and rest is not None and len(rest) == 16 and
                                np.allclose(cur, np.array(rest), atol=1e-5))
    reuse_memo = {}

    def can_reuse(b):
        if b.name not in reuse_memo:
            reuse_memo[b.name] = own_same[b.name] and (b.parent is None or can_reuse(b.parent))
        return reuse_memo[b.name]

    for b in ordered:
        j = j3d.Joint(b.name)
        j.parent = index[b.parent.name] if b.parent else -1
        j.kind = int(b.get("bmd_kind", 1))
        j.scale_compensate = int(b.get("bmd_scale_comp", 0))
        if can_reuse(b):
            s = [float(x) for x in b["bmd_local"]]
            j.trans, j.rot, j.scale = s[0:3], s[3:6], s[6:9]
        else:
            P = world[b.parent.name] if b.parent else np.eye(4)
            L = np.linalg.inv(P) @ world[b.name]
            j.trans = [float(x) for x in L[:3, 3]]
            j.rot = list(j3d.matrix_to_euler(L[:3, :3]))
            j.scale = [1.0, 1.0, 1.0]
        model.joints.append(j)

    roots = [i for i, j in enumerate(model.joints) if j.parent < 0]
    if len(roots) > 1:
        root = j3d.Joint("root")
        model.joints.append(root)
        for r in roots:
            model.joints[r].parent = len(model.joints) - 1
    return index


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def _srgb_from_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _read_color(me, name_hint):
    attrs = me.color_attributes
    if len(attrs) == 0:
        return None
    attr = attrs.get(name_hint)
    if attr is None and name_hint == "Col0":
        attr = attrs.active_color or attrs[0]
    if attr is None:
        return None
    n = len(attr.data)
    arr = np.empty(n * 4, np.float32)
    attr.data.foreach_get("color", arr)
    arr = arr.reshape(n, 4)
    if attr.data_type in ("BYTE_COLOR",):
        arr = arr.copy()
        arr[:, :3] = _srgb_from_linear(arr[:, :3])
    return arr, attr.domain


def _eval_mesh(obj, apply_modifiers, context):
    if not apply_modifiers:
        return obj.data, None
    disabled = []
    for m in obj.modifiers:
        if m.type == "ARMATURE" and m.show_viewport:
            m.show_viewport = False
            disabled.append(m)
    dg = context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    return me, (ev, disabled)


def _export_mesh_object(model, obj, arm_obj, bone_index, mats, scale, apply_modifiers, context):
    me, cleanup = _eval_mesh(obj, apply_modifiers, context)
    try:
        _export_mesh_data(model, obj, me, arm_obj, bone_index, mats, scale)
    finally:
        if cleanup:
            ev, disabled = cleanup
            ev.to_mesh_clear()
            for m in disabled:
                m.show_viewport = True


def _export_mesh_data(model, obj, me, arm_obj, bone_index, mats, scale):
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    if nt == 0:
        return
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)
    tri_verts = np.empty(nt * 3, np.int32)
    tri_loops = np.empty(nt * 3, np.int32)
    tri_mat = np.empty(nt, np.int32)
    me.loop_triangles.foreach_get("vertices", tri_verts)
    me.loop_triangles.foreach_get("loops", tri_loops)
    me.loop_triangles.foreach_get("material_index", tri_mat)
    tri_verts = tri_verts.reshape(nt, 3)
    tri_loops = tri_loops.reshape(nt, 3)

    # object -> armature space -> J3D space
    M = bc.to_np(obj.matrix_world)
    if arm_obj is not None:
        M = np.linalg.inv(bc.to_np(arm_obj.matrix_world)) @ M
    # Blender and GX use opposite front-face winding; a mirrored object flips it once more
    if np.linalg.det(M[:3, :3]) >= 0:
        tri_verts = tri_verts[:, [0, 2, 1]]
        tri_loops = tri_loops[:, [0, 2, 1]]
    pts = co @ M[:3, :3].T + M[:3, 3]
    verts = bc.point_from_blender(pts, scale)

    normals = None
    try:
        n = np.empty(len(me.loops) * 3, np.float32)
        me.corner_normals.foreach_get("vector", n)
        n = n.reshape(-1, 3).astype(np.float64)
        N = np.linalg.inv(M[:3, :3]).T
        n = bc.vec_from_blender(n @ N.T)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1
        normals = (n / ln).astype(np.float32)
    except Exception as exc:
        print(f"[BMD] could not read normals of {obj.name}: {exc}")

    # UV layers -> GX texcoord sets
    uv_layers = {}
    for i, layer in enumerate(me.uv_layers):
        t = None
        if layer.name.startswith("UV") and layer.name[2:].isdigit():
            t = int(layer.name[2:])
        if t is None or t > 7 or t in uv_layers:
            t = next((k for k in range(8) if k not in uv_layers), None)
        if t is None:
            break
        a = np.empty(len(layer.data) * 2, np.float32)
        layer.data.foreach_get("uv", a)
        a = a.reshape(-1, 2).copy()
        a[:, 1] = 1.0 - a[:, 1]
        uv_layers[t] = a

    colors = [None, None]
    loop_vert = None
    for c in range(2):
        res = _read_color(me, f"Col{c}")
        if res is None:
            continue
        arr, domain = res
        if domain == "POINT":
            if loop_vert is None:
                loop_vert = np.empty(len(me.loops), np.int32)
                me.loops.foreach_get("vertex_index", loop_vert)
            arr = arr[loop_vert]
        colors[c] = arr

    # skin weights
    group_joint = {}
    for vg in obj.vertex_groups:
        if vg.name in bone_index:
            group_joint[vg.index] = bone_index[vg.name]
    root = 0
    parent_joint = None
    if obj.parent_type == "BONE" and obj.parent_bone in bone_index:
        parent_joint = bone_index[obj.parent_bone]
    infl_sets, infl_lookup = [], {}
    vert_inf = np.zeros(nv, np.int32)
    for i, v in enumerate(me.vertices):
        ws = {}
        for g in v.groups:
            j = group_joint.get(g.group)
            if j is not None and g.weight > 1e-4:
                ws[j] = ws.get(j, 0.0) + g.weight
        if not ws:
            ws = {parent_joint if parent_joint is not None else root: 1.0}
        if len(ws) > 4:
            ws = dict(sorted(ws.items(), key=lambda kv: -kv[1])[:4])
        tot = sum(ws.values())
        key = tuple(sorted((j, round(w / tot, 5)) for j, w in ws.items()))
        k = infl_lookup.get(key)
        if k is None:
            k = len(infl_sets)
            infl_lookup[key] = k
            infl_sets.append(key)
        vert_inf[i] = k

    slots = obj.material_slots
    for mi in np.unique(tri_mat):
        sel = np.nonzero(tri_mat == mi)[0]
        mat = slots[mi].material if mi < len(slots) else None
        faces_old = tri_verts[sel]
        used, inv = np.unique(faces_old, return_inverse=True)
        em = writer.ExportMesh()
        em.name = obj.name
        em.verts = verts[used].astype(np.float32)
        em.vert_influence = vert_inf[used]
        em.influence_sets = infl_sets
        em.faces = inv.reshape(-1, 3).astype(np.int32)
        lp = tri_loops[sel].reshape(-1)
        if normals is not None:
            em.loop_normals = normals[lp]
        for c in range(2):
            if colors[c] is not None:
                em.loop_colors[c] = colors[c][lp]
        for t, a in uv_layers.items():
            em.loop_uvs[t] = a[lp]
        em.order = int(obj.get("bmd_order", 1e6 + len(model.meshes)))
        em.sort_key = int(obj.get("bmd_shape_index", 1e6 + len(model.meshes)))
        if "bmd_joint" in obj:
            em.joint = int(obj["bmd_joint"])
        em.material = mats.index_for(mat, em)
        model.meshes.append(em)


# --------------------------------------------------------------------------
# materials & textures
# --------------------------------------------------------------------------

class _MaterialCollector:
    def __init__(self, model, default_format):
        self.model = model
        self.default_format = default_format
        self.mat_index = {}
        self.images = {}   # tex name -> (Texture)
        self.pending = []  # (ExportMesh)
        self.sort_keys = {}  # material index -> original index
        self._source = None

    def index_for(self, bl_mat, mesh):
        key = bl_mat.name if bl_mat else None
        if key not in self.mat_index:
            self.mat_index[key] = len(self.model.materials)
            d, name = self._build(bl_mat, mesh)
            self.model.materials.append(d)
            self.model.material_names.append(name)
            n = self.mat_index[key]
            self.sort_keys[n] = int(bl_mat.get("bmd_index", 1e6 + n)) if bl_mat else 1e6 + n
        idx = self.mat_index[key]
        self.pending.append((mesh, idx))
        return idx

    def _source_textures(self, path):
        if self._source is None:
            self._source = {}
            if path and os.path.isfile(path):
                try:
                    with open(path, "rb") as f:
                        self._source = {t.key: t for t in j3d.parse(f.read()).textures}
                except Exception as exc:
                    print(f"[BMD] cannot read original model {path}: {exc}")
        return self._source

    def preload_original_textures(self, arm_obj, source_file, allow_missing, report):
        """Add every texture of the imported model in its original order.

        Texture indices are referenced by game code and animations (e.g. eye frames), so a
        missing or re-ordered texture shifts them and crashes the game.  Textures whose image was
        lost (Blender purges unused images) are restored from the original BMD if available.
        """
        names = list(arm_obj["bmd_textures"])
        keys = list(arm_obj["bmd_texture_keys"]) if "bmd_texture_keys" in arm_obj else names
        source = source_file or arm_obj.get("bmd_source", "")
        missing, restored = [], []
        for name, key in zip(names, keys):
            img = bpy.data.images.get(name)
            if img is not None:
                self._texture_for(img, "REPEAT")
                continue
            tex = self._source_textures(source).get(key)
            if tex is not None:
                self.images[key] = tex
                self.model.textures.append(tex)
                restored.append(key)
            else:
                missing.append(key)
        if restored and report:
            report({"WARNING"}, f"{len(restored)} textures were missing in Blender and were restored "
                                f"from the original model: {', '.join(restored)}")
        if missing and not allow_missing:
            raise ExportError(
                f"{len(missing)} textures of the imported model no longer exist in Blender "
                f"({', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}). Exporting without them "
                "would shift the texture indices and crash the game. Type the path of the file you "
                "imported into the 'Original BMD' export option so they can be restored (or enable "
                "'Allow Missing Textures' if you removed them on purpose).")

    def _texture_for(self, image, wrap_ext):
        name = bc.clean_texture_name(image)
        key = bc.texture_key(image)
        if key in self.images:
            return key
        t = j3d.Texture(name)
        t.key = key
        w, h = image.size
        if w == 0 or h == 0:
            raise ExportError(f"Image '{image.name}' has no pixel data")
        buf = np.empty(w * h * 4, np.float32)
        image.pixels.foreach_get(buf)
        t.rgba = np.clip(np.rint(buf.reshape(h, w, 4)[::-1] * 255.0), 0, 255).astype(np.uint8)
        props = image.bmd
        orig = image.get("bmd_orig_format")
        external = bool(props.external)
        reused = False if external else self._try_reuse_raw(image, t, w, h)
        if external:
            # Placeholder slot: header only.  Keep the original format / LOD settings.
            t.external = True
            t.format = gx.NAME_TO_FORMAT.get(orig, gx.CMPR)
            t.max_lod = int(image.get("bmd_max_lod", 0))
        elif not reused:
            want = props.format if props.format != "AUTO" else (
                self.default_format if self.default_format != "AUTO" else None)
            if want is None:
                if orig in gx.NAME_TO_FORMAT and gx.NAME_TO_FORMAT[orig] not in (gx.C4, gx.C8, gx.C14X2):
                    fmt = gx.NAME_TO_FORMAT[orig]
                else:
                    fmt = gx.RGB5A3 if gx.has_partial_alpha(t.rgba) else gx.CMPR
            else:
                fmt = gx.NAME_TO_FORMAT[want]
            t.format = fmt
        has_a = gx.has_alpha(t.rgba)
        t.transparency = int(image.get("bmd_transparency", 2 if has_a else 0))
        if reused:
            mips = t.mip_count
        else:
            mips = props.mip_count
            if mips <= 0:
                mips = 1
                if "bmd_min_filter" not in image and min(w, h) >= 16:
                    mips = min(gx.max_mip_count(w, h), 4)
            t.mip_count = mips
        ext_wrap = {"EXTEND": 0, "CLIP": 0, "REPEAT": 1, "MIRROR": 2}.get(wrap_ext, 1)
        t.wrap_s = int(image.get("bmd_wrap_s", ext_wrap))
        t.wrap_t = int(image.get("bmd_wrap_t", ext_wrap))
        t.min_filter = int(image.get("bmd_min_filter", 5 if mips > 1 else 1))
        t.mag_filter = int(image.get("bmd_mag_filter", 1))
        t.min_lod = int(image.get("bmd_min_lod", 0))
        t.lod_bias = int(image.get("bmd_lod_bias", 0))
        fl = image.get("bmd_flags10")
        t.flags10 = bytes.fromhex(fl) if fl else bytes([1 if mips > 1 else 0, 0, 0, 0])
        if mips > 1:
            t.flags10 = bytes([1]) + t.flags10[1:]
        self.images[key] = t
        self.model.textures.append(t)
        return key

    def _try_reuse_raw(self, image, t, w, h):
        """Use the original encoded bytes when the image still looks like the imported one."""
        raw = image.get("bmd_raw")
        if not raw:
            return False
        props = image.bmd
        orig = image.get("bmd_orig_format")
        mips0 = int(image.get("bmd_raw_mips", 0))
        if props.format not in ("AUTO", orig) or props.mip_count not in (0, mips0):
            return False
        try:
            data = zlib.decompress(base64.b64decode(raw))
            rfmt = int(image["bmd_raw_fmt"])
            rw, rh = (int(v) for v in image["bmd_raw_size"])
            if (rw, rh) != (w, h):
                return False
            pal_raw, pal = None, None
            if image.get("bmd_raw_pal"):
                pal_raw = zlib.decompress(base64.b64decode(image["bmd_raw_pal"]))
                pal = gx.decode_palette(int(image["bmd_pal_fmt"]), pal_raw, int(image["bmd_pal_count"]))
            ref = gx.decode(rfmt, data, rw, rh, pal).astype(np.int16)
            cur = t.rgba.astype(np.int16)
            diff = np.abs(ref - cur)
            visible = ((ref[..., 3] > 0) | (cur[..., 3] > 0))[..., None]
            worst = max(int(diff[..., 3].max()), int(np.where(visible, diff[..., :3], 0).max()))
            if worst > 1:
                return False
        except Exception as exc:
            print(f"[BMD] cannot reuse original texture data of {image.name}: {exc}")
            return False
        t.raw_data, t.raw_palette = data, pal_raw
        t.format, t.mip_count = rfmt, mips0
        t.max_lod = int(image.get("bmd_max_lod", (mips0 - 1) * 8))
        t.pal_enabled = int(image.get("bmd_pal_en", 0))
        t.pal_format = int(image.get("bmd_pal_fmt", 0))
        t.pal_count = int(image.get("bmd_pal_count", 0))
        return True

    def _image_nodes(self, bl_mat):
        """{slot: (image, extension)} from nodes named bmd_tex<N>, else first image node."""
        out = {}
        if bl_mat is None or bl_mat.node_tree is None:
            return out
        for node in bl_mat.node_tree.nodes:
            if node.type != "TEX_IMAGE":
                continue
            if node.name.startswith("bmd_tex") and node.name[7:].isdigit():
                out[int(node.name[7:])] = (node.image, getattr(node, "extension", "REPEAT"))
        if not out:
            for node in bl_mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    out[0] = (node.image, getattr(node, "extension", "REPEAT"))
                    break
        return out

    def _build(self, bl_mat, mesh):
        nodes = self._image_nodes(bl_mat)
        stored = bl_mat.get("bmd_material") if bl_mat else None
        mode = bl_mat.bmd.render_mode if bl_mat else "AUTO"
        alpha_ref = bl_mat.bmd.alpha_ref if bl_mat else 128
        cull_back = bool(bl_mat.use_backface_culling) if bl_mat else True
        name = (bl_mat.get("bmd_name") or bl_mat.name) if bl_mat else "default"

        if stored:
            d = json.loads(stored)
            tex_names = list(d.get("texNames", [None] * 8))
            for slot, (img, ext) in nodes.items():
                tex_names[slot] = self._texture_for(img, ext) if img else None
            # slots that previously had a texture but now have no node keep their name only
            # if the texture still exists in the scene
            for slot in range(8):
                if tex_names[slot] and tex_names[slot] not in self.images:
                    base = tex_names[slot]
                    img = bpy.data.images.get(base)
                    if img is not None:
                        tex_names[slot] = self._texture_for(img, "REPEAT")
                    else:
                        tex_names[slot] = None
            d["texNames"] = tex_names
            if (d.get("cull") == 2) != cull_back:
                d["cull"] = 2 if cull_back else 0
            if mode != "AUTO":
                mat3.apply_render_mode(d, mode, alpha_ref)
            else:
                kind, ref = mat3.classify_alpha(d)
                if kind == mat3.RENDER_CUTOUT and ref is not None and int(ref) != int(alpha_ref):
                    ac = d["alphaComp"]
                    if ac[0] in (4, 6):
                        ac[1] = int(alpha_ref)
                    else:
                        ac[4] = int(alpha_ref)
            return d, name

        # brand-new material: build a simple unlit texture * vertex colour material
        tex_name = None
        has_alpha = False
        if 0 in nodes and nodes[0][0] is not None:
            tex_name = self._texture_for(*nodes[0])
            has_alpha = gx.has_alpha(self.images[tex_name].rgba)
        elif nodes:
            slot = min(nodes)
            if nodes[slot][0] is not None:
                tex_name = self._texture_for(*nodes[slot])
                has_alpha = gx.has_alpha(self.images[tex_name].rgba)
        if mode == "AUTO":
            blended = False
            try:
                blended = bl_mat is not None and bl_mat.surface_render_method == "BLENDED"
            except Exception:
                pass
            if blended:
                mode = mat3.RENDER_TRANSLUCENT
            elif has_alpha:
                mode = mat3.RENDER_CUTOUT
            else:
                mode = mat3.RENDER_OPAQUE
        d = mat3.default_material(tex_name, use_vertex_colors=mesh.loop_colors[0] is not None,
                                  mode=mode, cull_back=cull_back, alpha_ref=alpha_ref)
        return d, name

    def finish(self):
        """Make meshes consistent with what their materials sample."""
        for mesh, idx in self.pending:
            d = self.model.materials[idx]
            if mat3.uses_vertex_colors(d) and mesh.loop_colors[0] is None:
                mesh.loop_colors[0] = np.ones((len(mesh.faces) * 3, 4), np.float32)
            for t in mat3.required_uv_sets(d):
                if mesh.loop_uvs[t] is None:
                    mesh.loop_uvs[t] = np.zeros((len(mesh.faces) * 3, 2), np.float32)
        # keep the original material order (animations and game code address materials by index)
        order = sorted(range(len(self.model.materials)), key=lambda i: (self.sort_keys.get(i, 1e9), i))
        remap = {old: new for new, old in enumerate(order)}
        self.model.materials[:] = [self.model.materials[i] for i in order]
        self.model.material_names[:] = [self.model.material_names[i] for i in order]
        for mesh, _idx in self.pending:
            mesh.material = remap[mesh.material]
        self.pending = [(m, remap[i]) for m, i in self.pending]
        # unique material names (J3D tables are by index but names should be distinct)
        seen = {}
        names = self.model.material_names
        for i, n in enumerate(names):
            if n in seen:
                names[i] = f"{n}_{i}"
            seen[n] = True
