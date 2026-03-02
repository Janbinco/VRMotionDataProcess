"""
Unity Parts Sphere Renderer
============================

Renders each point as a small sphere (circle in 2D projection).
Points are depth-sorted so front points occlude back points.
No alpha shapes - just direct point projection with circles.

Output:
    data/unity/<model_name>/<variant>/
    └── composite_check/
        ├── sphere_composite_mask.png   (2x2 grid, each part a different color)
        └── color_legend.png

Usage:
    python render_parts_spheres.py --model cactus --variant wholehand_visible
    python render_parts_spheres.py --model cactus --variant wholehand_visible --radius 0.02
    python render_parts_spheres.py --model cactus --variant wholehand_visible --auto-radius
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
from matplotlib.collections import PatchCollection
from PIL import Image
from scipy.spatial import cKDTree

# ============================================================================
# Config
# ============================================================================
DEFAULT_NUM_POINTS = 5000
DEFAULT_RESOLUTION = 512
DEFAULT_RADIUS = 0.015  # Fixed sphere radius in normalized coordinates
CAMERA_ELEVATION = 0.0
CAMERA_AZIMUTHS = [90.0, 180.0, 270.0, 0.0]

# Distinct colors for up to 8 parts
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

# ============================================================================
# Core functions
# ============================================================================

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
        [np.sin(az_rad), np.cos(az_rad), 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(el_rad), -np.sin(el_rad)],
        [0, np.sin(el_rad), np.cos(el_rad)]
    ])
    return points @ Rz.T @ Rx.T


def create_2x2_grid(images):
    top = np.hstack([images[1], images[0]])
    bottom = np.hstack([images[3], images[2]])
    return np.vstack([top, bottom])


def compute_optimal_radius(points, k=5):
    """Compute optimal radius based on average k-nearest neighbor distance."""
    if len(points) < k + 1:
        return DEFAULT_RADIUS
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)  # k+1 because first neighbor is itself
    avg_dist = distances[:, 1:].mean()  # exclude self
    # Use half the average distance as radius for good coverage without too much overlap
    return avg_dist * 0.5


# ============================================================================
# Sphere-based rendering
# ============================================================================

def render_spheres_composite(part_data, azimuth, elevation, resolution=512, radius=None,
                             auto_radius=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2)):
    """Render all parts using circles (projected spheres), depth-sorted.

    Args:
        part_data: list of (name, normalized_points, color) tuples
        radius: fixed radius for all spheres (if not auto_radius)
        auto_radius: if True, compute optimal radius per part
    Returns:
        RGB numpy array (resolution x resolution x 3)
    """
    # Collect all points with their depth, 2D position, and color
    all_circles = []  # (depth, x, y, radius, color)

    for name, points, color in part_data:
        rotated = rotate_points(points, azimuth, elevation)
        # x, z are the 2D coordinates; y is depth (smaller y = farther)
        pts_2d = rotated[:, [0, 2]]
        depths = rotated[:, 1]

        if auto_radius:
            r = compute_optimal_radius(pts_2d)
        else:
            r = radius if radius else DEFAULT_RADIUS

        for i in range(len(points)):
            all_circles.append((depths[i], pts_2d[i, 0], pts_2d[i, 1], r, color))

    # Sort by depth: farthest first (smallest y), closest last (drawn on top)
    all_circles.sort(key=lambda x: x[0])

    # Create figure
    fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Draw circles back-to-front
    for depth, x, y, r, color in all_circles:
        circle = Circle((x, y), r, facecolor=color, edgecolor=color, linewidth=0)
        ax.add_patch(circle)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(resolution, resolution, 4)[:, :, :3]
    plt.close(fig)
    return img


def process_model(model_dir, num_points=5000, resolution=512, radius=None, auto_radius=False):
    model_name = Path(model_dir).name
    print(f"\nProcessing: {model_dir}")

    parts_dir = os.path.join(model_dir, "parts")
    if os.path.isdir(parts_dir):
        obj_files = sorted(glob.glob(os.path.join(parts_dir, "*.obj")))
    else:
        obj_files = sorted(glob.glob(os.path.join(model_dir, "*.obj")))
    obj_files = [f for f in obj_files if not Path(f).stem.startswith("finger_traj")]

    if not obj_files:
        print("  No .obj files found!")
        return

    part_names = [Path(f).stem for f in obj_files]
    print(f"  Parts: {part_names}")

    centroid, scale = get_global_bounds(obj_files)

    # Load and normalize all parts
    part_data = []
    for i, obj_file in enumerate(obj_files):
        name = Path(obj_file).stem
        points = load_points_from_obj(obj_file)
        points = (points - centroid) / scale
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        color = PART_COLORS[i % len(PART_COLORS)]
        part_data.append((name, points, color))
        print(f"    {name}: {len(points)} points, color={color}")

    # Render 4 views
    views = []
    for az in CAMERA_AZIMUTHS:
        img = render_spheres_composite(part_data, az, CAMERA_ELEVATION, resolution,
                                       radius=radius, auto_radius=auto_radius)
        views.append(img)

    # Create 2x2 grid and save
    output_dir = os.path.join(model_dir, "parts_sphere_mask")
    os.makedirs(output_dir, exist_ok=True)
    grid = create_2x2_grid(views)

    output_path = os.path.join(output_dir, "sphere_composite_mask.png")
    Image.fromarray(grid).save(output_path)
    print(f"  Saved: {output_path}")

    # Save color legend
    fig, ax = plt.subplots(figsize=(4, 0.4 * len(part_data)), dpi=100)
    for i, (name, _, color) in enumerate(part_data):
        ax.barh(i, 1, color=color, height=0.8)
        ax.text(0.5, i, name, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(part_data) - 0.5)
    ax.axis('off')
    fig.tight_layout()
    legend_path = os.path.join(output_dir, "color_legend.png")
    fig.savefig(legend_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"  Saved: {legend_path}")


def main():
    parser = argparse.ArgumentParser(description="Render parts using sphere projection")
    parser.add_argument("--model", required=True, help="Model name (e.g., cactus)")
    parser.add_argument("--data-dir", required=True, help="Root data directory (e.g., DataCheck)")
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--radius", type=float, default=None,
                        help=f"Fixed sphere radius (default: {DEFAULT_RADIUS})")
    parser.add_argument("--auto-radius", action="store_true",
                        help="Automatically compute optimal radius based on point density")
    args = parser.parse_args()

    model_dir = Path(args.data_dir) / args.model

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    process_model(str(model_dir), args.num_points, args.resolution,
                  args.radius, args.auto_radius)


if __name__ == "__main__":
    main()
