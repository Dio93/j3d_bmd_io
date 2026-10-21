"""Create Blender data from a parsed J3D model."""

import base64
import json
import os
import zlib

import bpy
import numpy as np
from mathutils import Matrix

from . import blender_common as bc
from .j3d import gx_texture as gx
from .j3d import material as mat3
from .j3d import model as j3d


def import_bmd(context, filepath, scale=0.01, bone_axis="X", import_textures=True, report=None):
    with open(filepath, "rb") as f:
        raw = f.read()
    bmd = j3d.parse(raw)
    base = os.path.splitext(os.path.basename(filepath))[0]

    coll = bpy.data.collections.new(base)
    context.scene.collection.children.link(coll)

    images = _create_images(bmd, base) if import_textures else {}
    materials = [_create_material(bmd, i, m, images) for i, m in enumerate(bmd.materials)]
    arm_obj = _create_armature(context, coll, bmd, base, scale, bone_axis)
    bone_names = [b for b in arm_obj["bmd_bone_names"]]

    meshes = j3d.extract_meshes(bmd)
    for k, md in enumerate(meshes):
        _create_mesh_object(coll, arm_obj, bone_names, bmd, md, materials, base, scale, k)

    # keep the raw model data that is not editable in Blender
    # texture order matters in-game (texture animations switch textures by index)
    arm_obj["bmd_textures"] = [images[t.key].name for t in bmd.textures if t.key in images]
    arm_obj["bmd_texture_keys"] = [t.key for t in bmd.textures if t.key in images]
    arm_obj["bmd_source"] = os.path.abspath(filepath)
    arm_obj["bmd_vtx_formats"] = json.dumps({str(a): [c, t, f] for a, c, t, f in bmd.vtx_formats})
    arm_obj["bmd_inf_flags"] = int(bmd.inf_flags)
    arm_obj["bmd_sub_version"] = bmd.sub_version.hex()
    arm_obj["bmd_scale"] = float(scale)
    arm_obj["bmd_bone_axis"] = bone_axis
    if report:
        report({"INFO"}, f"Imported {base}: {len(bmd.joints)} joints, {len(meshes)} meshes, "
                         f"{len(bmd.materials)} materials, {len(bmd.textures)} textures")
    return arm_obj


# --------------------------------------------------------------------------
# textures & materials
# --------------------------------------------------------------------------

def _create_images(bmd, base):
    images = {}
    for t in bmd.textures:
        h, w = t.rgba.shape[:2]
        img = bpy.data.images.new(t.key, w, h, alpha=True)
        px = (t.rgba[::-1].astype(np.float32) / 255.0).reshape(-1)
        img.pixels.foreach_set(px)
        try:
            img.alpha_mode = "STRAIGHT"
        except Exception:
            pass
        img["bmd_name"] = t.name
        img["bmd_key"] = t.key
        img["bmd_transparency"] = int(t.transparency)
        img["bmd_wrap_s"] = int(t.wrap_s)
        img["bmd_wrap_t"] = int(t.wrap_t)
        img["bmd_min_filter"] = int(t.min_filter)
        img["bmd_mag_filter"] = int(t.mag_filter)
        img["bmd_min_lod"] = int(t.min_lod)
        img["bmd_max_lod"] = int(t.max_lod)
        img["bmd_lod_bias"] = int(t.lod_bias)
        img["bmd_flags10"] = t.flags10.hex()
        img["bmd_orig_format"] = gx.FORMAT_NAMES.get(t.format, str(t.format))
        if t.raw_data is not None:  # original bytes: re-used verbatim by the exporter if untouched
            img["bmd_raw"] = base64.b64encode(zlib.compress(t.raw_data, 6)).decode("ascii")
            img["bmd_raw_fmt"] = int(t.format)
            img["bmd_raw_size"] = [int(w), int(h)]
            img["bmd_raw_mips"] = int(t.mip_count)
            if t.raw_palette:
                img["bmd_raw_pal"] = base64.b64encode(zlib.compress(t.raw_palette, 6)).decode("ascii")
                img["bmd_pal_fmt"] = int(t.pal_format)
                img["bmd_pal_count"] = int(t.pal_count)
            img["bmd_pal_en"] = int(t.pal_enabled)
        fmt_name = gx.FORMAT_NAMES.get(t.format, "AUTO")
        try:
            img.bmd.format = fmt_name if fmt_name in {i[0] for i in bc.FORMAT_ITEMS} else "AUTO"
            img.bmd.mip_count = int(t.mip_count)
            img.bmd.external = bool(t.external)
        except Exception:
            pass
        try:
            img.pack()
        except Exception:
            pass
        # Textures such as the eye frames are not used by any material.  Without a fake user
        # Blender deletes them on save, which would shift every texture index in the export.
        img.use_fake_user = True
        images[t.key] = img
    return images


def _get_json_material(bmd_mat):
    return json.dumps(bmd_mat, separators=(",", ":"))


def _create_material(bmd, index, m, images):
    name = bmd.material_names[index] if index < len(bmd.material_names) else f"material{index}"
    mat = bpy.data.materials.new(name)
    mat["bmd_name"] = name
    mat["bmd_index"] = index
    mat["bmd_material"] = _get_json_material(m)
    mat.use_backface_culling = (m.get("cull") == 2)
    mat.bmd.render_mode = "AUTO"
    alpha_ref = m["alphaComp"][1] if m.get("alphaComp") else 128
    mat.bmd.alpha_ref = int(alpha_ref)
    try:
        _build_nodes(mat, m, images)
    except Exception as exc:  # node API differences must never break the import
        print(f"[BMD] could not build nodes for {name}: {exc}")
    return mat


_WRAP_EXT = {0: "EXTEND", 1: "REPEAT", 2: "MIRROR"}


def _wrap_axis(nodes, links, src, mode, x, y):
    """Per-axis UV wrapping: 0 clamp, 1 repeat, 2 mirror. Returns the output socket."""
    n = nodes.new("ShaderNodeMath")
    n.location = (x, y)
    if mode == 1:
        n.operation = "FRACT"
    elif mode == 2:
        n.operation = "PINGPONG"
        n.inputs[1].default_value = 1.0
    else:
        n.operation = "ADD"
        n.inputs[1].default_value = 0.0
        n.use_clamp = True
    links.new(src, n.inputs[0])
    return n.outputs[0]


def _build_nodes(mat, m, images):
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    nt.nodes.clear()
    nodes, links = nt.nodes, nt.links
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (1100, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (800, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    for sock, val in (("Roughness", 1.0), ("Specular IOR Level", 0.0), ("Specular", 0.0)):
        if sock in bsdf.inputs:
            bsdf.inputs[sock].default_value = val

    # texture slots -> image nodes
    tex_names = m.get("texNames", [None] * 8)
    slot_nodes = {}
    for slot, tname in enumerate(tex_names):
        if tname is None or tname not in images:
            continue
        node = nodes.new("ShaderNodeTexImage")
        node.name = f"bmd_tex{slot}"
        node.label = f"BMD tex{slot}: {tname}"
        node.image = images[tname]
        node.location = (-100, 300 - 280 * slot)
        img = node.image
        node.interpolation = "Closest" if img.get("bmd_mag_filter", 1) == 0 else "Linear"
        slot_nodes[slot] = node

    # which slot does TEV stage 0 sample?
    main_slot, main_coord = None, 0
    order = m.get("tevOrder", [None])[0]
    if order and order[1] < 8 and order[1] in slot_nodes:
        main_slot, main_coord = order[1], order[0]
    elif slot_nodes:
        main_slot = min(slot_nodes)

    if main_slot is not None:
        node = slot_nodes[main_slot]
        img = node.image
        ws, wt = int(img.get("bmd_wrap_s", 1)), int(img.get("bmd_wrap_t", 1))
        tc = m.get("texCoord", [None] * 8)
        uv_name = "UV0"
        if main_coord < 8 and tc[main_coord] and 4 <= tc[main_coord][1] <= 11:
            uv_name = f"UV{tc[main_coord][1] - 4}"
        uvn = nodes.new("ShaderNodeUVMap")
        uvn.uv_map = uv_name
        uvn.location = (-700, 300)
        use_chain = ws != wt
        if not use_chain:
            try:
                node.extension = _WRAP_EXT.get(ws, "REPEAT")
            except Exception:
                use_chain = True
        if not use_chain:
            links.new(uvn.outputs["UV"], node.inputs["Vector"])
        else:
            # the image node has a single extension mode, GX has one per axis
            sep = nodes.new("ShaderNodeSeparateXYZ")
            sep.location = (-520, 300)
            comb = nodes.new("ShaderNodeCombineXYZ")
            comb.location = (-250, 300)
            links.new(uvn.outputs["UV"], sep.inputs["Vector"])
            xs = _wrap_axis(nodes, links, sep.outputs["X"], ws, -400, 400)
            ys = _wrap_axis(nodes, links, sep.outputs["Y"], wt, -400, 200)
            links.new(xs, comb.inputs["X"])
            links.new(ys, comb.inputs["Y"])
            links.new(comb.outputs["Vector"], node.inputs["Vector"])
            node.extension = "EXTEND"
    # remaining slots: keep their own wrap mode (only the first axis is representable)
    for slot, node in slot_nodes.items():
        if slot != main_slot:
            try:
                node.extension = _WRAP_EXT.get(int(node.image.get("bmd_wrap_s", 1)), "REPEAT")
            except Exception:
                pass

    color_out = alpha_out = None
    if main_slot is not None:
        color_out = slot_nodes[main_slot].outputs["Color"]
        alpha_out = slot_nodes[main_slot].outputs["Alpha"]

    vc = None
    if mat3.uses_vertex_colors(m):
        vc = nodes.new("ShaderNodeVertexColor")
        vc.layer_name = "Col0"
        vc.location = (-100, -450)
        if color_out is None:
            color_out = vc.outputs["Color"]
        else:
            try:
                mix = nodes.new("ShaderNodeMix")
                mix.data_type = "RGBA"
                mix.blend_type = "MULTIPLY"
                mix.inputs[0].default_value = 1.0
                mix.location = (300, 100)
                links.new(color_out, mix.inputs[6])
                links.new(vc.outputs["Color"], mix.inputs[7])
                color_out = mix.outputs[2]
            except Exception:
                pass
    if color_out is not None:
        links.new(color_out, bsdf.inputs["Base Color"])

    # alpha: tex alpha (TEXA) and/or vertex alpha (RASA) of TEV stage 0, as the game draws it
    kind, ref = mat3.classify_alpha(m)
    if kind != mat3.RENDER_OPAQUE:
        stage = (m.get("tevStage") or [None])[0]
        a_in = stage["alphaIn"] if stage else []
        sources = []
        if alpha_out is not None and (4 in a_in or not a_in):
            sources.append(alpha_out)
        if vc is not None and 5 in a_in:
            sources.append(vc.outputs["Alpha"])
        if not sources and alpha_out is not None:
            sources.append(alpha_out)
        if sources:
            a = sources[0]
            for extra in sources[1:]:
                mul = nodes.new("ShaderNodeMath")
                mul.operation = "MULTIPLY"
                mul.location = (450, -300)
                links.new(a, mul.inputs[0])
                links.new(extra, mul.inputs[1])
                a = mul.outputs[0]
            if kind == mat3.RENDER_CUTOUT:
                gt = nodes.new("ShaderNodeMath")
                gt.operation = "GREATER_THAN"
                gt.inputs[1].default_value = max(0.0, (float(ref) - 0.5) / 255.0)
                gt.location = (600, -300)
                links.new(a, gt.inputs[0])
                a = gt.outputs[0]
            links.new(a, bsdf.inputs["Alpha"])
            try:
                mat.surface_render_method = "BLENDED" if kind == mat3.RENDER_TRANSLUCENT else "DITHERED"
            except Exception:
                pass


# --------------------------------------------------------------------------
# armature
# --------------------------------------------------------------------------

def _unique_names(joints):
    seen, out = set(), []
    for i, j in enumerate(joints):
        n = j.name or f"joint{i}"
        if n in seen:
            n = f"{n}.{i}"
        seen.add(n)
        out.append(n)
    return out


def _create_armature(context, coll, bmd, base, scale, bone_axis):
    joints = bmd.joints
    names = _unique_names(joints)
    arm = bpy.data.armatures.new(base)
    obj = bpy.data.objects.new(base, arm)
    coll.objects.link(obj)
    obj.show_in_front = True

    fix = bc.R_FIX if bone_axis == "X" else np.eye(4)
    rest = []  # bone matrices in Blender armature space
    for i, j in enumerate(joints):
        W = bc.C @ bmd.world[i] @ bc.C_INV
        W[:3, 3] *= scale
        rest.append(bc.orthonormalize(W) @ fix)

    heads = [r[:3, 3] for r in rest]
    children = {i: [] for i in range(len(joints))}
    for i, j in enumerate(joints):
        if j.parent >= 0:
            children[j.parent].append(i)
    default_len = 5.0 * scale
    lengths = [0.0] * len(joints)
    for i in range(len(joints)):
        d = [np.linalg.norm(heads[c] - heads[i]) for c in children[i]]
        d = [x for x in d if x > 1e-6]
        lengths[i] = min(d) if d else 0.0
    for i in range(len(joints)):
        if lengths[i] <= 0:
            p = joints[i].parent
            lengths[i] = max(lengths[p] * 0.5, default_len * 0.2) if p >= 0 and lengths[p] > 0 else default_len
        lengths[i] = max(lengths[i], default_len * 0.2)

    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    for o in view_layer.objects:
        o.select_set(False)
    view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = []
    for i, j in enumerate(joints):
        eb = arm.edit_bones.new(names[i])
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, float(lengths[i]), 0.0)
        eb.matrix = Matrix(rest[i].tolist())
        ebones.append(eb)
    for i, j in enumerate(joints):
        if j.parent >= 0:
            ebones[i].parent = ebones[j.parent]
            ebones[i].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    view_layer.objects.active = prev_active

    for i, j in enumerate(joints):
        b = arm.bones[names[i]]
        b["bmd_index"] = i
        b["bmd_kind"] = int(j.kind)
        b["bmd_scale_comp"] = int(j.scale_compensate)
        if bmd.maya and j.scale_compensate and j.parent >= 0:
            # Maya "segment scale compensate": the parent's scale is not inherited
            try:
                b.inherit_scale = "NONE"
            except Exception:
                pass
        b["bmd_local"] = [float(x) for x in (*j.trans, *j.rot, *j.scale)]
        b["bmd_rest"] = [float(x) for x in bc.to_np(b.matrix_local).reshape(-1)]
    obj["bmd_bone_names"] = names
    return obj


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def _create_mesh_object(coll, arm_obj, bone_names, bmd, md, materials, base, scale, k):
    shape_name = f"{base}_shape{md.shape_index:02d}"
    verts = bc.point_to_blender(md.verts, scale)
    # GX front faces are wound the opposite way to Blender's: flip every triangle and
    # permute the per-corner attributes to match.
    faces = md.faces[:, [0, 2, 1]]
    perm = (np.arange(len(faces))[:, None] * 3 + np.array([0, 2, 1])).reshape(-1)
    md.loop_normals = None if md.loop_normals is None else md.loop_normals[perm]
    md.loop_colors = [None if c is None else c[perm] for c in md.loop_colors]
    md.loop_uvs = [None if u is None else u[perm] for u in md.loop_uvs]
    me = bpy.data.meshes.new(shape_name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    n_loops = len(faces) * 3

    # smooth shading so custom normals take effect
    try:
        me.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    except Exception:
        pass

    for t in range(8):
        uv = md.loop_uvs[t]
        if uv is None:
            continue
        layer = me.uv_layers.new(name=f"UV{t}")
        a = uv.astype(np.float32).copy()
        a[:, 1] = 1.0 - a[:, 1]
        layer.data.foreach_set("uv", a.reshape(-1))
    for c in range(2):
        col = md.loop_colors[c]
        if col is None:
            continue
        attr = me.color_attributes.new(name=f"Col{c}", type="FLOAT_COLOR", domain="CORNER")
        attr.data.foreach_set("color", col.astype(np.float32).reshape(-1))

    if md.loop_normals is not None:
        n = bc.point_to_blender(md.loop_normals, 1.0)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1
        try:
            me.normals_split_custom_set((n / ln).tolist())
        except Exception as exc:
            print(f"[BMD] custom normals failed for {shape_name}: {exc}")

    mat_idx = md.material_index
    if 0 <= mat_idx < len(materials):
        me.materials.append(materials[mat_idx])

    obj = bpy.data.objects.new(shape_name, me)
    coll.objects.link(obj)
    obj["bmd_shape_index"] = int(md.shape_index)
    obj["bmd_joint"] = int(md.joint)
    obj["bmd_order"] = int(k)
    obj.parent = arm_obj
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj

    groups = {}
    for set_idx, infl in enumerate(md.influence_sets):
        verts_idx = np.nonzero(md.vert_influence == set_idx)[0].tolist()
        if not verts_idx:
            continue
        for joint, weight in infl:
            if joint >= len(bone_names):
                continue
            vg = groups.get(joint)
            if vg is None:
                vg = obj.vertex_groups.new(name=bone_names[joint])
                groups[joint] = vg
            vg.add(verts_idx, float(weight), "REPLACE")
    return obj
