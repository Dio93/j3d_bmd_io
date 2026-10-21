# J3D BMD / BCK Import & Export for Blender (Zelda: Twilight Princess / Dusklight)

Imports and exports Nintendo J3D models (`.bmd`, `.bdl` read-only) and joint animations (`.bck`).

Verified against `model.bmd` (static map, 22 shapes, 25 textures), `uri.bmd` (29 joints, envelope
skinning, Maya matrix mode) and `uri_wait_a.bck`. Both models survive parse -> write -> parse with
identical geometry, skin weights, normals, UVs and materials; unchanged textures are written back
byte-for-byte.

## AI disclosure and disclaimer
The source code of this add-on was written with the help of an AI model (Claude by Anthropic), working
from the requirements, test files and feedback of me. The maintainer directed the
project and tested the results in-game, but the code has not been independently audited.

The add-on is provided "as is", without warranty of any kind (see the GPL-3.0-or-later license).
It reads and writes file formats that were analysed from publicly available information and the
open-source Dusklight code. Always keep backups of your original game files; exported files can
still crash the game or behave unexpectedly, especially collision (`.kcl`) edits.

This is an unofficial fan tool. It is not affiliated with or endorsed by Nintendo, Anthropic, or the
Dusklight / Twilit Realm project. No game assets are included.

## Installation (Blender 4.2 - 5.x)
Download this code as ZIP.

In Blender:

Edit -> Preferences -> Get Extensions -> drop-down (top right) -> *Install from Disk...* and pick
the ZIP. The operators appear under File -> Import / Export:
*J3D Model (.bmd)* and *J3D Animation (.bck)*. Files can also be dragged into the viewport.

## Models
1. **Import** with the default scale `0.01` (100 game units = 1 m). You get an armature (one bone per
   joint), one mesh object per shape (vertex groups + Armature modifier), materials and images.
2. Edit meshes / textures. Append new bones rather than re-ordering existing ones.
3. **Export** with the armature and meshes selected (*Selected Only*) and **Scale = 100** (inverse of
   the import scale).

### What is preserved
* Joint order, hierarchy and parameters of unchanged bones, INF1 flags (Maya mode on Uri). Animations
  address joints by index, so the order matters.
* **All** textures in their original order with wrap / filter / LOD settings, including textures no
  material uses (e.g. eye textures swapped by texture animations). Duplicate names are allowed.
* The complete material block (TEV stages, colour channels, texture matrices, fog, blend ...) is kept
  as JSON in the material custom property `bmd_material`.

#### Index stability (important for the game)
The game and its animations address textures, shapes and materials **by index**, so the exporter keeps
them stable: shapes are written in their original shape-index order, materials in their original
order, every texture of the imported model is written in its original position (even those no
material uses, such as the eye frames), and shapes stay attached to their original joint.
Blender deletes images that have no user when a file is saved, so imported images get a *Fake User*.
If a texture is still missing at export time, it is restored from the original BMD (the path is
remembered at import, or set *Original BMD* in the export options); otherwise the export stops with
an error instead of silently writing a shifted texture list that crashes the game.

### Skinning layout (DRW1 / EVP1)
The game's loader computes the number of draw matrices as `DRW1 entries - envelopes` and fills the
weighted ones as `view * envelope[k]`. The exporter therefore writes DRW1 exactly like the original
tools: all rigid entries first, then one weighted entry per envelope in order, then the same
envelopes once more as filler. Every export is checked against these rules before it is written
(`validate_engine_layout`); an export that would crash the game raises an error instead.

### Output format
To stay as close as possible to what the game's own tools produce, the exporter writes normals / UVs /
positions in the encoding of the imported file (e.g. 16-bit normals with 14 fraction bits; switch off
with *Keep Original Vertex Formats*), uses triangle **strips** only, stores shape bounds in model
space (radius measured from the origin) and puts a bounding box on the root joint only.

## Textures and HD texture packs
The importer stores the original encoded texture bytes (all mips, palettes) on each image. On export,
images that still match the imported pixels are written back **byte-identically**, so texture packs
that match textures by content keep working. Only edited images, or images whose format / mip setting
you changed, are re-encoded (CMPR, RGB5A3, RGBA32, RGB565, IA8, IA4, I8, I4). Palette formats are
decoded on import and re-encoded as CMPR / RGB5A3 only if you change the image.

### Runtime textures (placeholders)

Some texture slots in stage and character models contain no image data at all: their data
offset in the TEX1 header is 0 and the game fills the slot at runtime (for example texture
pattern animations). The importer detects them, creates a grey placeholder image and ticks
**Runtime Texture** in the image panel. The exporter writes them back as header-only entries
(data offset 0). Never untick the option for such an image unless you really want to ship a
fixed texture there - otherwise the game shows the embedded picture instead of the runtime
one (in older versions of this add-on this showed up as green/black checkerboard patterns,
because header bytes were read as pixels).

### Materials
* Material tab -> *J3D (BMD) Material*: **Render Mode** (Auto keeps imported values; Opaque / Cutout /
  Translucent), Alpha Reference, Backface Culling.
* Alpha test and blending are detected from the material's `alphaComp` / `blend` data and wired into
  the shader (alpha test via "Greater Than", blending as *Blended*).
* GX wraps per axis (Clamp / Repeat / Mirror). If the two axes differ, the importer builds a
  Separate XYZ -> Math -> Combine XYZ chain in front of the image node.
* To swap a texture, set the image of node `bmd_tex0` (`bmd_tex1` ...). Per-image format and mip count:
  Image Editor -> sidebar -> *Image* -> *J3D (BMD) Texture*.
* New materials (without `bmd_material`) become simple unlit "texture x vertex colour" materials.
* UV layers are named `UV0`..`UV7`, vertex colours `Col0` / `Col1`.

## Animations (.bck)
**Import**: select the imported armature, then File -> Import -> *J3D Animation (.bck)*. A new action
is created (named after the file, with a fake user) and keyed on every frame with linear
interpolation; channels that never change get a single key. Frame `n` of the file is Blender frame
`n`; the scene range can be set to the animation length. The loop mode is stored on the action
(`bmd_loop_mode`).

**Export**: make the armature active, assign the action, then File -> Export -> *J3D Animation (.bck)*.
Every frame is sampled, converted back to joint-local scale / rotation / translation, and fitted
with cubic Hermite curves using as few keys as the tolerances allow (rotation 2e-4 rad, translation
0.01 units, scale 1e-4). Options: loop mode (Auto uses the stored one), action or scene frame range,
and the scale (by default the one stored at model import).

Notes
* The joint count of the animation equals the number of bones; bones keep their `bmd_index`.
* Joints with a non-uniform rest scale (e.g. the eyebrows of Uri, 1.6 on X) are handled: the stored
  scale is folded into the rest transform so the animation is not scaled twice. In Maya mode a joint
  that does not inherit its parent's scale gets `Inherit Scale: None` on its bone.
* The Blender pose reproduces the game's skinning (checked: skinned vertices match to 1e-4 units).

## Collision (.kcl)

Stage collision (`room.kcl`, used together with `room.plc`) is imported and exported as well:
*File > Import/Export > TP Collision (.kcl)*. The game derives **Link's shadow, standing height,
walls and slopes from the collision mesh, not from `model.bmd`**, so a changed visual mesh needs a
matching collision mesh.

* Import it with the same scale as the BMD (0.01) so both line up in Blender.
* Every collision attribute is a material called `KCL_<n>`, where `n` is the row of `room.plc`.
  The material of a face *is* its attribute: paint faces with another `KCL_<n>` material to change
  how they behave. Faces without such a material get the export option *Default Attribute*.
* Export the collision mesh only (*Selected Only*) with the inverse scale (100). The file contains
  the triangles, normals and the octree the game uses for every query.
* `room.plc` is **not** rewritten; keep the original file next to the new `room.kcl`. Attribute
  numbers beyond the number of rows in the PLC are reported as a warning.
* Limits: about 12000 triangles per file (the game asserts a 16-bit prism index), and the triangle
  winding defines the face normal, so make sure floors face upwards.

## Known limitations
* No TEV shading in Blender; the Blender materials are only a preview.
* Joint scales other than 1 are exact only for unchanged bones.
* At most 4 bone weights per vertex. Shapes are written as triangle lists (no strips), so files are
  larger than the originals. MDL3 (BDL display lists) is not written.
* Only `bck` (joint animations) is supported, not `btk` / `btp` / `bpk` / `brk` / `bca`.
* Not tested in a real Blender session: the `bpy` layer was exercised with a mock, the J3D code against
  the real files. Please report API errors with the console output.
