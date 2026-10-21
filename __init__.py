"""J3D BMD Import/Export for Blender (Zelda: Twilight Princess / Dusklight)."""

import os

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty,
                       StringProperty)
from bpy.types import Operator, OperatorFileListElement, Panel
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import blender_common as bc


def blender_anim_items():
    from .blender_anim import LOOP_ITEMS
    return LOOP_ITEMS


class IMPORT_OT_bmd(Operator, ImportHelper):
    """Import a J3D BMD3/BDL4 model"""
    bl_idname = "import_scene.bmd"
    bl_label = "Import J3D BMD"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".bmd"
    filter_glob: StringProperty(default="*.bmd;*.bdl", options={"HIDDEN"})
    files: CollectionProperty(type=OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    scale: FloatProperty(name="Scale", default=0.01, min=1e-6, soft_max=10.0, precision=4,
                         description="Blender units per J3D unit (use the inverse value when exporting)")
    bone_axis: EnumProperty(name="Bone Orientation", items=bc.BONE_AXIS_ITEMS, default="X")
    import_textures: BoolProperty(name="Import Textures", default=True)

    def execute(self, context):
        from . import blender_import
        from .j3d.common import J3DError
        paths = []
        if self.files:
            paths = [os.path.join(self.directory, f.name) for f in self.files]
        if not paths:
            paths = [self.filepath]
        for p in paths:
            try:
                blender_import.import_bmd(context, p, self.scale, self.bone_axis,
                                          self.import_textures, self.report)
            except J3DError as exc:
                self.report({"ERROR"}, f"{os.path.basename(p)}: {exc}")
                return {"CANCELLED"}
        return {"FINISHED"}


class EXPORT_OT_bmd(Operator, ExportHelper):
    """Export meshes (and their armature) as a J3D BMD3 model"""
    bl_idname = "export_scene.bmd"
    bl_label = "Export J3D BMD"
    bl_options = {"REGISTER", "PRESET"}

    filename_ext = ".bmd"
    filter_glob: StringProperty(default="*.bmd", options={"HIDDEN"})

    scale: FloatProperty(name="Scale", default=100.0, min=1e-6, soft_max=1000.0, precision=3,
                         description="J3D units per Blender unit (inverse of the import scale)")
    use_selection: BoolProperty(name="Selected Only", default=True,
                                description="Export only selected objects (select the armature and meshes)")
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=False,
                                  description="Apply modifiers (except armature) before exporting")
    default_format: EnumProperty(name="Texture Format", items=bc.FORMAT_ITEMS, default="AUTO",
                                 description="Format for textures without their own GX Format setting")
    # Plain text field on purpose: a FILE_PATH property adds a browse button, and Blender cannot
    # open a file selector from inside the (already open) export file selector.
    source_file: StringProperty(name="Original BMD", default="",
                                description="Path of the file this model was imported from. Only needed "
                                            "when the path remembered at import no longer exists; used "
                                            "to restore textures that are gone from Blender")
    keep_vertex_formats: BoolProperty(name="Keep Original Vertex Formats", default=True,
                                      description="Encode normals / UVs / positions like the imported "
                                                  "file (e.g. 16-bit) instead of 32-bit floats")
    allow_missing_textures: BoolProperty(name="Allow Missing Textures", default=False,
                                         description="Export even if textures of the imported model "
                                                     "are gone (shifts texture indices!)")

    def execute(self, context):
        from . import blender_export
        from .j3d.common import J3DError
        try:
            blender_export.export_bmd(context, self.filepath, self.scale, self.use_selection,
                                      self.apply_modifiers, self.default_format, self.source_file,
                                      self.allow_missing_textures, self.keep_vertex_formats,
                                      self.report)
        except (blender_export.ExportError, J3DError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class IMPORT_OT_kcl(Operator, ImportHelper):
    """Import a Twilight Princess collision mesh (room.kcl, optionally with room.plc)"""
    bl_idname = "import_scene.kcl"
    bl_label = "Import TP Collision (.kcl)"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".kcl"
    filter_glob: StringProperty(default="*.kcl", options={"HIDDEN"})

    scale: FloatProperty(name="Scale", default=0.01, min=1e-6, soft_max=10.0, precision=4,
                         description="Blender units per game unit (same as the BMD import)")
    plc_file: StringProperty(name="PLC File", default="",
                             description="Attribute table (room.plc). Empty = the .plc next to the "
                                         "file. Only used to describe the attributes")

    def execute(self, context):
        from . import blender_kcl
        from .j3d.common import J3DError
        try:
            blender_kcl.import_kcl(context, self.filepath, self.scale, self.plc_file, self.report)
        except J3DError as exc:
            self.report({"ERROR"}, f"{os.path.basename(self.filepath)}: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class EXPORT_OT_kcl(Operator, ExportHelper):
    """Export meshes as a Twilight Princess collision file (room.kcl). Keep the original room.plc"""
    bl_idname = "export_scene.kcl"
    bl_label = "Export TP Collision (.kcl)"
    bl_options = {"REGISTER", "PRESET"}

    filename_ext = ".kcl"
    filter_glob: StringProperty(default="*.kcl", options={"HIDDEN"})

    scale: FloatProperty(name="Scale", default=100.0, min=1e-6, soft_max=1000.0, precision=3,
                         description="Game units per Blender unit (inverse of the import scale)")
    use_selection: BoolProperty(name="Selected Only", default=True)
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=True)
    default_attr: IntProperty(name="Default Attribute", default=0, min=0, max=65535,
                              description="Attribute (row of room.plc) for faces that have no "
                                          "KCL_<n> material")
    keep_area: BoolProperty(name="Keep Original Area", default=True,
                            description="Reuse the area and block size of the imported file when the "
                                        "new geometry still fits into it")

    def execute(self, context):
        from . import blender_kcl
        from .j3d.common import J3DError
        try:
            blender_kcl.export_kcl(context, self.filepath, self.scale, self.use_selection,
                                   self.apply_modifiers, self.default_attr, self.keep_area, self.report)
        except (blender_kcl.KclExportError, J3DError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class IMPORT_OT_bck(Operator, ImportHelper):
    """Import a J3D joint animation (.bck) onto the active armature"""
    bl_idname = "import_anim.bck"
    bl_label = "Import J3D Animation"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".bck"
    filter_glob: StringProperty(default="*.bck", options={"HIDDEN"})
    files: CollectionProperty(type=OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    use_model_scale: BoolProperty(name="Use Model Scale", default=True,
                                  description="Use the scale the armature was imported with")
    scale: FloatProperty(name="Scale", default=0.01, min=1e-6, soft_max=10.0, precision=4,
                         description="Blender units per J3D unit (only if Use Model Scale is off)")
    set_scene_range: BoolProperty(name="Set Scene Range", default=True,
                                  description="Set the scene frame range to the animation length")

    def execute(self, context):
        from . import blender_anim
        from .j3d.common import J3DError
        paths = [os.path.join(self.directory, f.name) for f in self.files] if self.files else []
        if not paths:
            paths = [self.filepath]
        for p in paths:
            try:
                blender_anim.import_bck(context, p, self.scale, self.use_model_scale,
                                        self.set_scene_range, self.report)
            except (blender_anim.AnimError, J3DError) as exc:
                self.report({"ERROR"}, f"{os.path.basename(p)}: {exc}")
                return {"CANCELLED"}
        return {"FINISHED"}


class EXPORT_OT_bck(Operator, ExportHelper):
    """Export the active action of the active armature as a J3D .bck animation"""
    bl_idname = "export_anim.bck"
    bl_label = "Export J3D Animation"
    bl_options = {"REGISTER", "PRESET"}

    filename_ext = ".bck"
    filter_glob: StringProperty(default="*.bck", options={"HIDDEN"})

    use_model_scale: BoolProperty(name="Use Model Scale", default=True,
                                  description="Use the scale the armature was imported with")
    scale: FloatProperty(name="Scale", default=100.0, min=1e-6, soft_max=1000.0, precision=3,
                         description="J3D units per Blender unit (only if Use Model Scale is off)")
    loop_mode: EnumProperty(name="Loop Mode", items=blender_anim_items(), default="AUTO")
    frame_source: EnumProperty(
        name="Frame Range",
        items=[("ACTION", "Action Range", "Frame range of the keyframes in the action"),
               ("SCENE", "Scene Range", "Scene start / end frame")],
        default="ACTION")

    def execute(self, context):
        from . import blender_anim
        from .j3d.common import J3DError
        try:
            blender_anim.export_bck(context, self.filepath, self.scale, self.use_model_scale,
                                    self.loop_mode, self.frame_source, self.report)
        except (blender_anim.AnimError, J3DError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class MATERIAL_PT_bmd(Panel):
    bl_label = "J3D (BMD) Material"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        mat = context.material
        layout.prop(mat.bmd, "render_mode")
        if mat.bmd.render_mode in {"AUTO", "CUTOUT"}:
            layout.prop(mat.bmd, "alpha_ref")
        layout.prop(mat, "use_backface_culling")
        if "bmd_material" in mat:
            layout.label(text="Imported TEV data kept in 'bmd_material'", icon="INFO")
        if mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.name.startswith("bmd_tex") and node.image:
                    row = layout.row()
                    row.label(text=node.name[4:])
                    row.prop(node.image.bmd, "format", text="")


class IMAGE_PT_bmd(Panel):
    bl_label = "J3D (BMD) Texture"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Image"

    @classmethod
    def poll(cls, context):
        return context.space_data and getattr(context.space_data, "image", None)

    def draw(self, context):
        img = context.space_data.image
        self.layout.prop(img.bmd, "format")
        self.layout.prop(img.bmd, "mip_count")
        self.layout.prop(img.bmd, "external")


def menu_import(self, context):
    self.layout.operator(IMPORT_OT_bmd.bl_idname, text="J3D Model (.bmd/.bdl)")
    self.layout.operator(IMPORT_OT_bck.bl_idname, text="J3D Animation (.bck)")
    self.layout.operator(IMPORT_OT_kcl.bl_idname, text="TP Collision (.kcl)")


def menu_export(self, context):
    self.layout.operator(EXPORT_OT_bmd.bl_idname, text="J3D Model (.bmd)")
    self.layout.operator(EXPORT_OT_bck.bl_idname, text="J3D Animation (.bck)")
    self.layout.operator(EXPORT_OT_kcl.bl_idname, text="TP Collision (.kcl)")


_classes = (IMPORT_OT_bmd, EXPORT_OT_bmd, IMPORT_OT_bck, EXPORT_OT_bck, IMPORT_OT_kcl, EXPORT_OT_kcl,
            MATERIAL_PT_bmd, IMAGE_PT_bmd)
_file_handlers = []


def _make_file_handlers():
    """Drag & drop support (Blender 4.1+); silently skipped if unavailable."""
    try:
        from bpy_extras.io_utils import poll_file_object_drop

        class BMD_FH_import(bpy.types.FileHandler):
            bl_idname = "BMD_FH_import"
            bl_label = "J3D BMD"
            bl_import_operator = "import_scene.bmd"
            bl_file_extensions = ".bmd;.bdl"

            @classmethod
            def poll_drop(cls, context):
                return poll_file_object_drop(context)

        class BCK_FH_import(bpy.types.FileHandler):
            bl_idname = "BCK_FH_import"
            bl_label = "J3D BCK"
            bl_import_operator = "import_anim.bck"
            bl_file_extensions = ".bck"

            @classmethod
            def poll_drop(cls, context):
                return poll_file_object_drop(context)

        class KCL_FH_import(bpy.types.FileHandler):
            bl_idname = "KCL_FH_import"
            bl_label = "TP Collision KCL"
            bl_import_operator = "import_scene.kcl"
            bl_file_extensions = ".kcl"

            @classmethod
            def poll_drop(cls, context):
                return poll_file_object_drop(context)

        return [BMD_FH_import, BCK_FH_import, KCL_FH_import]
    except Exception:
        return []


def register():
    global _file_handlers
    bc.register_classes()
    for c in _classes:
        bpy.utils.register_class(c)
    _file_handlers = _make_file_handlers()
    for fh in _file_handlers:
        bpy.utils.register_class(fh)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
    global _file_handlers
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    for fh in _file_handlers:
        bpy.utils.unregister_class(fh)
    _file_handlers = []
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
    bc.unregister_classes()
