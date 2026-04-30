"""
Blender 5.0 scene builder + animation renderer for the residuals-visuals atlas flythrough.

Run via:
    blender --background --python blender/render_atlas.py -- \
        --csv cache/blender_points.csv \
        --output output/atlas/blender_frames/ \
        --frames 1350 \
        --resolution 1920 1080 \
        [--test-frame 600]   # render a single frame for preview

The script is idempotent: starts from a clean scene, builds everything from
the CSV, and writes either a PNG sequence (default) or a single PNG (--test-frame).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector


# ----------- argument parsing (Blender passes args after `--`) -----------
def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True, help="dir for PNG sequence (or single image if --test-frame)")
    p.add_argument("--frames", type=int, default=1350, help="total frames at 30 fps (default 45 sec)")
    p.add_argument("--resolution", nargs=2, type=int, default=[1920, 1080])
    p.add_argument("--samples", type=int, default=64, help="Eevee samples per pixel")
    p.add_argument("--test-frame", type=int, default=None, help="render only this frame")
    p.add_argument("--point-size", type=float, default=0.08)
    p.add_argument("--emission-strength", type=float, default=2.5)
    p.add_argument("--volumetric-density", type=float, default=0.004)
    p.add_argument("--motion-blur", action="store_true", help="enable motion blur (off by default for clarity)")
    p.add_argument("--save-blend", type=str, default=None,
                   help="if set, save the built scene as a .blend file at this path (no rendering)")
    p.add_argument("--no-render", action="store_true", help="skip rendering (useful with --save-blend)")
    return p.parse_args(argv)


# ----------- scene reset -----------
def clean_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for blocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                   bpy.data.lights, bpy.data.cameras, bpy.data.worlds,
                   bpy.data.node_groups, bpy.data.images):
        for b in list(blocks):
            try:
                blocks.remove(b)
            except RuntimeError:
                pass


# ----------- load points from CSV -----------
def load_points(csv_path: Path) -> tuple[list[Vector], list[tuple[float, float, float]], list[str]]:
    coords: list[Vector] = []
    colors: list[tuple[float, float, float]] = []
    cats: list[str] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append(Vector((float(row["x"]), float(row["y"]), float(row["z"]))))
            colors.append((float(row["r"]), float(row["g"]), float(row["b"])))
            cats.append(row["category"])
    return coords, colors, cats


# ----------- build the points mesh: 39k icospheres pre-baked into one mesh -----------
def make_points_mesh(
    coords: list[Vector],
    colors: list[tuple[float, float, float]],
    point_size: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """
    Build one big mesh containing N small icospheres at the given coords,
    each colored by its assigned RGB. Avoids GeoNodes entirely (more
    predictable across Blender versions).
    """
    import time
    t0 = time.time()

    # Build a single icosphere template via bmesh, then extract numpy arrays
    template_bm = bmesh.new()
    bmesh.ops.create_icosphere(template_bm, subdivisions=1, radius=point_size)
    template_verts = np.array([list(v.co) for v in template_bm.verts], dtype=np.float32)
    template_faces = np.array([[v.index for v in f.verts] for f in template_bm.faces], dtype=np.int32)
    template_bm.free()

    n_v_per = template_verts.shape[0]   # 12
    n_f_per = template_faces.shape[0]   # 20
    verts_per_face = template_faces.shape[1]  # 3
    n_pts = len(coords)

    # Bulk-build vertex positions: each point gets the template translated
    coords_arr = np.array([list(c) for c in coords], dtype=np.float32)
    verts = (coords_arr[:, None, :] + template_verts[None, :, :]).reshape(-1, 3)

    # Bulk-build face indices: each point's faces have indices offset by point_idx * n_v_per
    face_offsets = (np.arange(n_pts) * n_v_per).reshape(-1, 1, 1)
    faces = (template_faces[None, :, :] + face_offsets).reshape(-1, verts_per_face)

    # Create mesh with bulk fill
    mesh = bpy.data.meshes.new("atlas_points")
    mesh.vertices.add(n_pts * n_v_per)
    mesh.vertices.foreach_set("co", verts.flatten())

    n_loops = n_pts * n_f_per * verts_per_face
    n_polys = n_pts * n_f_per
    mesh.loops.add(n_loops)
    mesh.polygons.add(n_polys)
    mesh.polygons.foreach_set("loop_start", (np.arange(n_polys) * verts_per_face).astype(np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(n_polys, verts_per_face, dtype=np.int32))
    mesh.loops.foreach_set("vertex_index", faces.flatten().astype(np.int32))
    mesh.update(calc_edges=True)
    mesh.validate()

    # Per-vertex color attribute (POINT domain): each sphere's 12 verts share the point's color
    attr = mesh.color_attributes.new(name="cat_color", type="FLOAT_COLOR", domain="POINT")
    colors_arr = np.array(colors, dtype=np.float32)  # (n_pts, 3)
    color_per_vert = np.repeat(colors_arr, n_v_per, axis=0)
    rgba = np.column_stack([color_per_vert, np.ones(len(color_per_vert), dtype=np.float32)])
    attr.data.foreach_set("color", rgba.flatten())

    obj = bpy.data.objects.new("atlas_points", mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)

    print(f"[atlas] built {n_pts} icospheres ({n_polys:,} faces) in {time.time() - t0:.1f}s")
    return obj


# ----------- glowing sphere material -----------
def make_emission_material(name: str, strength: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "cat_color"
    attr.attribute_type = "GEOMETRY"  # per-vertex color attribute on the mesh
    attr.location = (-400, 0)

    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    em.location = (-150, 0)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (100, 0)

    nt.links.new(attr.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


# ----------- camera path: a Bezier curve + Track-To target -----------
def build_camera_path(n_frames: int) -> bpy.types.Object:
    """
    Hand-crafted control points designed to:
      - start outside, dive into central density
      - arc around wavelet zone
      - cross morphological lobe
      - pull back for a wide ending shot
    Coordinates are in the same frame as the points (~±10 units cloud).
    """
    # 8 keyframe positions for the camera through the cloud
    waypoints = [
        Vector(( 18.0,  -8.0,   6.0)),  # 0   wide outside, looking in
        Vector(( 10.0,  -3.0,   3.0)),  # 1   approaching
        Vector((  3.0,   2.0,   0.5)),  # 2   diving toward central density
        Vector(( -1.0,   1.0,  -1.0)),  # 3   inside central core
        Vector(( -3.5,   3.0,   1.5)),  # 4   banking through wavelet arms
        Vector((  1.0,  -2.5,   2.0)),  # 5   cross to morphological lobe
        Vector((  6.0,  -4.5,  -2.0)),  # 6   spiraling out below
        Vector(( 14.0, -10.0,  10.0)),  # 7   final wide pulled-back shot
    ]
    look_targets = [
        Vector((  0.0,   0.0,   0.0)),  # always look near the centroid
        Vector((  0.0,   0.0,   0.0)),
        Vector(( -1.0,   1.0,  -0.5)),  # look ahead into the dive
        Vector(( -3.0,   2.0,   0.5)),  # look into wavelet arm
        Vector((  1.0,  -1.0,   0.5)),  # turn toward morphological
        Vector((  4.0,  -3.0,  -1.0)),  # follow the lobe
        Vector((  0.0,   0.0,   0.0)),  # look back at center
        Vector((  0.0,   0.0,   0.0)),  # final framing
    ]

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 35  # wide-ish, dramatic
    cam_data.clip_start = 0.01
    cam_data.clip_end = 200.0
    # Depth of field
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 2.8
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    target = bpy.data.objects.new("CameraTarget", None)
    target.empty_display_size = 0.4
    target.empty_display_type = "PLAIN_AXES"
    bpy.context.collection.objects.link(target)
    cam_data.dof.focus_object = target  # focus on what we look at

    # Track-To constraint so camera always points at target
    tc = cam.constraints.new("TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"

    # Keyframe both objects
    n_keys = len(waypoints)
    for i in range(n_keys):
        f = int(round(i / (n_keys - 1) * (n_frames - 1))) + 1
        cam.location = waypoints[i]
        cam.keyframe_insert("location", frame=f)
        target.location = look_targets[i]
        target.keyframe_insert("location", frame=f)

    # Smooth Bezier tangents — drill into the slotted-action structure (Blender 4.4+)
    def _iter_fcurves(action):
        # Try legacy path first
        if hasattr(action, "fcurves") and action.fcurves:
            for fc in action.fcurves:
                yield fc
            return
        # Slotted action path (4.4+): action.layers[*].strips[*].channelbag(slot).fcurves
        for slot in getattr(action, "slots", ()):
            for layer in getattr(action, "layers", ()):
                for strip in getattr(layer, "strips", ()):
                    cb = strip.channelbag(slot) if hasattr(strip, "channelbag") else None
                    if cb is None:
                        continue
                    for fc in getattr(cb, "fcurves", ()):
                        yield fc

    for obj in (cam, target):
        if obj.animation_data and obj.animation_data.action:
            for fc in _iter_fcurves(obj.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"

    bpy.context.scene.camera = cam
    return cam


# ----------- world: dark background + very subtle volumetric -----------
def setup_world(volumetric_density: float = 0.004) -> None:
    world = bpy.data.worlds.new("AtlasWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.005, 0.005, 0.012, 1.0)
    bg.inputs["Strength"].default_value = 0.3
    bg.location = (-300, 100)

    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (0, 0)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    if volumetric_density > 0:
        vol = nt.nodes.new("ShaderNodeVolumeScatter")
        vol.inputs["Color"].default_value = (0.6, 0.7, 0.9, 1.0)
        vol.inputs["Density"].default_value = volumetric_density
        vol.inputs["Anisotropy"].default_value = 0.3
        vol.location = (-300, -100)
        nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])


# ----------- compositor: bloom (Blender 5 node-group-based compositor) -----------
def setup_compositor() -> None:
    scene = bpy.context.scene
    ng = bpy.data.node_groups.new("AtlasCompositor", "CompositorNodeTree")
    # Add an Image output socket to the group interface so the result is used as final composite
    ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = ng

    for n in list(ng.nodes):
        ng.nodes.remove(n)

    layers = ng.nodes.new("CompositorNodeRLayers")
    layers.location = (-400, 0)

    glare = ng.nodes.new("CompositorNodeGlare")
    for sock_name, value in (
        ("Type", "Fog Glow"),
        ("Size", 7.0),
        ("Threshold", 0.6),
        ("Strength", 1.0),
    ):
        try:
            glare.inputs[sock_name].default_value = value
        except (KeyError, TypeError):
            pass
    glare.location = (-150, 0)

    out = ng.nodes.new("NodeGroupOutput")
    out.location = (200, 0)

    ng.links.new(layers.outputs["Image"], glare.inputs["Image"])
    ng.links.new(glare.outputs["Image"], out.inputs["Image"])


# ----------- render settings -----------
def configure_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"  # in Blender 5 this IS Eevee Next
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15

    # Eevee settings — wrap each in hasattr/try since API drifts between versions
    eevee = scene.eevee
    for attr, value in (
        ("taa_render_samples", args.samples),
        ("use_volumetric_lights", True),
        ("use_volumetric_shadows", True),
        ("use_bokeh", True),
        ("volumetric_samples", 64),
    ):
        if hasattr(eevee, attr):
            try:
                setattr(eevee, attr, value)
            except (AttributeError, TypeError):
                pass
    if hasattr(scene.render, "use_motion_blur"):
        try:
            scene.render.use_motion_blur = bool(args.motion_blur)
        except (AttributeError, TypeError):
            pass
    if hasattr(eevee, "use_motion_blur"):
        try:
            eevee.use_motion_blur = bool(args.motion_blur)
        except (AttributeError, TypeError):
            pass

    # AgX color management for natural-looking glow (Blender 5 prefixes look names with "AgX -")
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.frame_start = 1
    scene.frame_end = args.frames


# ----------- main -----------
def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.output).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[atlas] CSV:    {csv_path}")
    print(f"[atlas] OUT:    {out_path}")
    print(f"[atlas] frames: {args.frames}, res: {args.resolution}, samples: {args.samples}")

    clean_scene()

    coords, colors, cats = load_points(csv_path)
    print(f"[atlas] loaded {len(coords)} points")

    mat = make_emission_material("atlas_emit", args.emission_strength)
    make_points_mesh(coords, colors, args.point_size, mat)

    setup_world(args.volumetric_density)
    setup_compositor()
    build_camera_path(args.frames)
    configure_render(args)

    if args.save_blend:
        blend_path = Path(args.save_blend).resolve()
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"[atlas] saved scene to {blend_path}")

    if args.no_render:
        return

    if args.test_frame is not None:
        bpy.context.scene.frame_set(args.test_frame)
        bpy.context.scene.render.filepath = str(out_path / f"test_frame_{args.test_frame:04d}.png")
        print(f"[atlas] rendering test frame {args.test_frame}")
        bpy.ops.render.render(write_still=True)
        print(f"[atlas] wrote {bpy.context.scene.render.filepath}")
    else:
        bpy.context.scene.render.filepath = str(out_path) + os.sep
        print("[atlas] rendering animation...")
        bpy.ops.render.render(animation=True)
        print("[atlas] animation done")


if __name__ == "__main__":
    main()
