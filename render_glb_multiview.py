#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy>=1.20.0",
#     "pillow>=8.0.0",
#     "trimesh>=3.9.0",
#     "pyrender>=0.1.45",
#     "pyglet>=1.5",
# ]
# ///
"""
Render GLB files from 4 viewpoints with textures and generate 2x2 composite image.

Views (elevation=0, horizontal):
  Top-left:  Front  (az=0°)
  Top-right: Left   (az=90°)
  Bottom-left:  Back  (az=180°)
  Bottom-right: Right (az=270°)

Output:
    GT/<model_name>/2x2_glb.png

Usage:
    python render_glb_multiview.py
    python render_glb_multiview.py --input-dir path/to/GT --resolution 512
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import trimesh
import pyrender

# ============================================================================
# Config  (elevation=0 to match render_parts_spheres.py)
# ============================================================================
RESOLUTION       = 512
CAMERA_ELEVATION = 0.0          # same as render_parts_spheres.py
CAMERA_DISTANCE  = 1.6          # relative to normalized model (fits in ~1-unit box)

# Front=0°, Left=90°, Back=180°, Right=270°
CAMERA_AZIMUTHS  = [0.0, 90.0, 180.0, 270.0]
VIEW_LABELS      = ["front", "left", "back", "right"]

# 2x2 layout:
#  [ front | left  ]
#  [ back  | right ]

# Per-model Y-axis pre-rotation (degrees, clockwise from above = negative Y).
# Add entries here when a model needs to be rotated before rendering.
MODEL_Y_ROTATIONS = {
    "tower":         -45.0,   # clockwise 45°
    "StandingApple": -90.0,   # clockwise 90°
}


# ============================================================================
# Helpers
# ============================================================================

def make_camera_pose(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """
    Return a 4x4 camera-to-world pose matrix.

    Convention (Y-up, right-handed):
      az=0   → camera on +Z axis, looking toward origin  (front)
      az=90  → camera on +X axis, looking toward origin  (left)
      az=180 → camera on -Z axis                         (back)
      az=270 → camera on -X axis                         (right)
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Spherical → Cartesian
    cx = distance * np.cos(el) * np.sin(az)
    cy = distance * np.sin(el)
    cz = distance * np.cos(el) * np.cos(az)
    cam_pos = np.array([cx, cy, cz])

    # Build orthonormal frame (look-at, Y-up)
    forward = -cam_pos / np.linalg.norm(cam_pos)   # toward origin
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:               # degenerate at poles
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward   # OpenGL: camera looks down -Z
    pose[:3, 3] = cam_pos
    return pose


def normalize_scene(scene):
    """Center and scale so the model fits inside a unit bounding box."""
    if isinstance(scene, trimesh.Scene):
        bounds = scene.bounds          # (2, 3)  or None
        if bounds is None:
            verts = np.vstack([g.vertices for g in scene.geometry.values()
                               if hasattr(g, "vertices") and len(g.vertices)])
            bounds = np.array([verts.min(0), verts.max(0)])
    else:
        bounds = scene.bounds

    centroid = bounds.mean(axis=0)
    scale    = np.ptp(bounds, axis=0).max()
    if scale < 1e-9:
        return scene

    T = np.eye(4)
    T[:3, 3] = -centroid
    S = np.diag([1/scale, 1/scale, 1/scale, 1.0])
    scene.apply_transform(S @ T)
    return scene


def build_pyrender_scene(trimesh_scene):
    """Convert a trimesh Scene (or Trimesh) to a pyrender Scene with lighting."""
    if isinstance(trimesh_scene, trimesh.Scene):
        pr_scene = pyrender.Scene.from_trimesh_scene(
            trimesh_scene,
            ambient_light=np.array([0.35, 0.35, 0.35])
        )
    else:
        pr_mesh  = pyrender.Mesh.from_trimesh(trimesh_scene, smooth=False)
        pr_scene = pyrender.Scene(ambient_light=np.array([0.35, 0.35, 0.35]))
        pr_scene.add(pr_mesh)

    # Add a sun-like directional light from above-front
    sun_pose = make_camera_pose(0.0, 30.0, 1.0)
    sun = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    pr_scene.add(sun, pose=sun_pose)

    return pr_scene


def render_one_view(pr_scene, azimuth, elevation, distance, resolution):
    """Render a single view; camera + its own fill-light are added then removed."""
    pose    = make_camera_pose(azimuth, elevation, distance)
    camera  = pyrender.PerspectiveCamera(yfov=np.radians(45.0), aspectRatio=1.0)
    cam_node = pr_scene.add(camera, pose=pose)

    # Per-view fill light (avoids pure-black silhouettes on back views)
    fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    fill_node = pr_scene.add(fill, pose=pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution,
                                          point_size=1.0)
    color, _ = renderer.render(pr_scene,
                               flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    pr_scene.remove_node(cam_node)
    pr_scene.remove_node(fill_node)

    # Return RGB (drop alpha)
    return color[:, :, :3]


def create_2x2_grid(views):
    """
    views[0]=front, views[1]=left, views[2]=back, views[3]=right
    Layout:
      [ front | left  ]
      [ back  | right ]
    """
    top    = np.hstack([views[0], views[1]])
    bottom = np.hstack([views[2], views[3]])
    return np.vstack([top, bottom])


# ============================================================================
# Per-model entry point
# ============================================================================

def render_glb_multiview(glb_path: Path, output_dir: Path, resolution: int = RESOLUTION):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading  : {glb_path.name}")
    scene = trimesh.load(str(glb_path), force="scene")
    scene = normalize_scene(scene)

    # Apply per-model Y-axis rotation (after normalization so center stays at origin)
    model_name = glb_path.stem
    y_rot_deg = MODEL_Y_ROTATIONS.get(model_name, 0.0)
    if y_rot_deg != 0.0:
        angle = np.radians(y_rot_deg)
        R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        scene.apply_transform(R)
        print(f"  Rotated  : Y-axis {y_rot_deg:+.1f}° (clockwise from above)")

    print("  Building pyrender scene …")
    pr_scene = build_pyrender_scene(scene)

    views = []
    for az, label in zip(CAMERA_AZIMUTHS, VIEW_LABELS):
        print(f"  Rendering: {label:5s}  az={az:5.1f}°  el={CAMERA_ELEVATION}°")
        img = render_one_view(pr_scene, az, CAMERA_ELEVATION,
                              CAMERA_DISTANCE, resolution)
        views.append(img)

    grid = create_2x2_grid(views)
    out_path = output_dir / "2x2_glb.png"
    Image.fromarray(grid).save(out_path)
    print(f"  Saved    : {out_path}")
    return out_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render GLB files with 4-view 2x2 composite (textured)"
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).parent / "GT"),
        help="Directory containing .glb files  (default: ./GT)",
    )
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help=f"Per-view resolution in pixels (default: {RESOLUTION})")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Process only these model names (stems), e.g. --models tower StandingApple")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    glb_files = sorted(input_dir.glob("*.glb"))
    if args.models:
        glb_files = [f for f in glb_files if f.stem in args.models]

    if not glb_files:
        print(f"No .glb files found in {input_dir}")
        return

    print(f"Found {len(glb_files)} GLB file(s) in {input_dir}\n")

    for glb_path in glb_files:
        model_name = glb_path.stem
        output_dir = glb_path.parent / model_name
        print(f"[{model_name}]")
        try:
            render_glb_multiview(glb_path, output_dir, args.resolution)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()
        print()

    print("Done!")


if __name__ == "__main__":
    main()
