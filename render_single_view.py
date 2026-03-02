"""
Render a single-view sphere composite mask from OBJ parts.
Based on render_parts_spheres.py, but renders one view only.
Default: azimuth=135°, elevation=29°

Output:
    <subfolder>/parts_sphere_mask/view_az135_el29.png

Usage:
    python render_single_view.py
    python render_single_view.py --azimuth 45 --elevation 29
    python render_single_view.py --azimuth 135 --elevation 29 --radius 0.02
    python render_single_view.py --auto-radius
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

DEFAULT_NUM_POINTS = 5000
DEFAULT_RESOLUTION = 512
DEFAULT_RADIUS     = 0.015

PART_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
]


def load_points_from_obj(obj_file, unity_to_standard=True):
    vertices = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if unity_to_standard:
                    z = -z
                    y, z = -z, y
                vertices.append([x, y, z])
    return np.array(vertices)


def get_global_bounds(obj_files):
    all_vertices = []
    for f in obj_files:
        verts = load_points_from_obj(f)
        if len(verts) > 0:
            all_vertices.append(verts)
    if not all_vertices:
        raise ValueError("No vertices found in any OBJ file")
    all_vertices = np.vstack(all_vertices)
    centroid = all_vertices.mean(axis=0)
    scale = np.abs(all_vertices - centroid).max()
    return centroid, scale


def rotate_points(points, azimuth, elevation):
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    Rz = np.array([
        [np.cos(az_rad), -np.sin(az_rad), 0],
        [np.sin(az_rad),  np.cos(az_rad), 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0,  np.cos(el_rad), -np.sin(el_rad)],
        [0,  np.sin(el_rad),  np.cos(el_rad)]
    ])
    return points @ Rz.T @ Rx.T


def render_spheres_composite(part_data, azimuth, elevation, resolution=512, radius=None,
                              xlim=(-1.2, 1.2), ylim=(-1.2, 1.2)):
    """Render all parts as depth-sorted colored circles (projected spheres)."""
    all_circles = []  # (depth, x, y, radius, color)

    for name, points, color in part_data:
        rotated = rotate_points(points, azimuth, -elevation)  # negate: +el = camera above looking down
        pts_2d  = np.column_stack([-rotated[:, 0], rotated[:, 2]])  # flip x to fix handedness
        depths  = rotated[:, 1]
        r = radius if radius else DEFAULT_RADIUS
        for i in range(len(points)):
            all_circles.append((depths[i], pts_2d[i, 0], pts_2d[i, 1], r, color))

    all_circles.sort(key=lambda c: c[0])  # farthest first

    fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for depth, x, y, r, color in all_circles:
        ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=color, linewidth=0))

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(resolution, resolution, 4)[:, :, :3]
    plt.close(fig)
    return img


def process_model(model_dir, azimuth, elevation, num_points=DEFAULT_NUM_POINTS,
                  resolution=DEFAULT_RESOLUTION, radius=None):
    model_dir = Path(model_dir)
    print(f"\nProcessing: {model_dir.name}")

    parts_dir = model_dir / "parts"
    if parts_dir.is_dir():
        obj_files = sorted(parts_dir.glob("*.obj"))
    else:
        obj_files = sorted(model_dir.glob("*.obj"))
    obj_files = [f for f in obj_files if not f.stem.startswith("finger_traj")]

    if not obj_files:
        print("  No .obj files found – skipping.")
        return

    print(f"  Parts: {[f.stem for f in obj_files]}")
    centroid, scale = get_global_bounds([str(f) for f in obj_files])

    part_data = []
    for i, obj_file in enumerate(obj_files):
        points = load_points_from_obj(str(obj_file))
        points = (points - centroid) / scale
        if len(points) > num_points:
            idx = np.random.choice(len(points), num_points, replace=False)
            points = points[idx]
        color = PART_COLORS[i % len(PART_COLORS)]
        part_data.append((obj_file.stem, points, color))
        print(f"    {obj_file.stem}: {len(points)} points, color={color}")

    img = render_spheres_composite(part_data, azimuth, elevation, resolution, radius)

    output_dir = model_dir / "parts_sphere_mask"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / f"sphere_mask_sview_az{int(azimuth)}_el{int(elevation)}.png"
    Image.fromarray(img).save(str(out_path))
    print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render single-view sphere composite mask")
    parser.add_argument("--azimuth",    type=float, default=135.0)
    parser.add_argument("--elevation",  type=float, default=29.0)
    parser.add_argument("--num-points", type=int,   default=DEFAULT_NUM_POINTS)
    parser.add_argument("--resolution", type=int,   default=DEFAULT_RESOLUTION)
    parser.add_argument("--radius",     type=float, default=None,
                        help=f"Sphere radius (default: {DEFAULT_RADIUS})")
    parser.add_argument("--model",      type=str,   default=None,
                        help="Only process this subfolder (e.g. bladeLego)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    print(f"Base directory: {base_dir}")
    print(f"View: az={args.azimuth}°  el={args.elevation}°")

    found = 0
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if args.model and subdir.name != args.model:
            continue
        obj_files = list(subdir.glob("*.obj")) + list((subdir / "parts").glob("*.obj") if (subdir / "parts").is_dir() else [])
        if not obj_files:
            continue
        process_model(subdir, args.azimuth, args.elevation,
                      args.num_points, args.resolution, args.radius)
        found += 1

    if found == 0:
        print("\nNo OBJ files found in any subfolder.")
    else:
        print(f"\nDone. {found} model(s) rendered.")


if __name__ == "__main__":
    main()
