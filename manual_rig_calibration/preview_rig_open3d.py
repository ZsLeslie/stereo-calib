"""Interactive four-camera preview in the front camera frame.

Dependencies: numpy, opencv-python, open3d (GUI-enabled build).
Usage: python preview_rig_open3d.py [rig_extrinsics.yaml]
       python preview_rig_open3d.py --check
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d


COLORS = {
    "front": (0.18, 0.30, 0.48),
    "mid": (0.90, 0.58, 0.08),
    "left": (0.02, 0.65, 0.65),
    "right": (0.65, 0.32, 0.82),
}


@dataclass
class CameraPose:
    name: str
    front_from_camera: np.ndarray

    @property
    def center(self):
        return self.front_from_camera[:3, 3]

    @property
    def optical_axis(self):
        return self.front_from_camera[:3, 2]


def load_poses(path: Path) -> list[CameraPose]:
    """Invert X_camera = R X_front + t, keeping scene coordinates in meters."""
    if not path.is_file():
        raise FileNotFoundError(path)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise ValueError(f"Cannot open OpenCV YAML: {path}")
        if storage.getNode("reference_camera").string() != "front":
            raise ValueError("Expected reference_camera: front")
        if storage.getNode("translation_unit").string() != "m":
            raise ValueError("Expected translation_unit: m")
        poses = []
        for name in COLORS:
            key = f"H_{name}_from_front"
            node = storage.getNode(key)
            if node.empty():
                raise ValueError(f"Missing matrix: {key}")
            H = node.mat()
            if H is None or H.shape != (4, 4) or not np.isfinite(H).all():
                raise ValueError(f"{key} must be a finite 4x4 matrix")
            R, t = H[:3, :3], H[:3, 3]
            np.testing.assert_allclose(H[3], [0, 0, 0, 1], atol=1e-7)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-7)
            np.testing.assert_allclose(np.linalg.det(R), 1, atol=1e-7)
            pose = np.eye(4)
            pose[:3, :3] = R.T
            pose[:3, 3] = -R.T @ t
            np.testing.assert_allclose(H @ pose, np.eye(4), atol=1e-7)
            if name == "front":
                np.testing.assert_allclose(pose, np.eye(4), atol=1e-7)
            for stored_key, expected in [
                (f"H_front_from_{name}", pose),
                (f"{name}_camera_center_in_front_m", pose[:3, 3:4]),
            ]:
                stored = storage.getNode(stored_key)
                if not stored.empty():
                    np.testing.assert_allclose(stored.mat(), expected, atol=1e-7)
            poses.append(CameraPose(name, pose))
        return poses
    finally:
        storage.release()


def line_set(points, lines, color):
    result = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=float)),
        o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    result.paint_uniform_color(color)
    return result


def build_geometry(poses, axis_length):
    """Return named geometry; all lengths and transforms are in meters."""
    items = []
    for pose in poses:
        color = COLORS[pose.name]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length)
        frame.transform(pose.front_from_camera)
        items.append((f"{pose.name}/XYZ", frame, pose.name))

        center = o3d.geometry.TriangleMesh.create_sphere(radius=axis_length * .085)
        center.compute_vertex_normals()
        center.paint_uniform_color(color)
        center.translate(pose.center)
        items.append((f"{pose.name}/center", center, pose.name))

        # Illustrative camera frustum, NOT a calibrated field of view.
        d = axis_length * .8
        points = np.array([[0, 0, 0], [-.45*d, -.3*d, d],
                           [.45*d, -.3*d, d], [.45*d, .3*d, d], [-.45*d, .3*d, d]])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        frustum = line_set(points, lines, color)
        frustum.transform(pose.front_from_camera)
        items.append((f"{pose.name}/frustum", frustum, pose.name))

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=axis_length * .025, cone_radius=axis_length * .07,
            cylinder_height=axis_length * 1.35, cone_height=axis_length * .25)
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color(color)
        arrow.transform(pose.front_from_camera)
        items.append((f"{pose.name}/optical_axis", arrow, pose.name))

    points = [p.center for p in poses]
    items.append(("baselines", line_set(points, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]], [.55, .59, .65]), "Reference"))
    grid_points, grid_lines = [], []
    for v in np.arange(-.15, .1501, .05):
        for a, b in [([v, 0, -.15], [v, 0, .15]), ([-.15, 0, v], [.15, 0, v])]:
            i = len(grid_points)
            grid_points.extend([a, b])
            grid_lines.append([i, i+1])
    items.append(("XZ grid at Y=0 (50 mm)", line_set(grid_points, grid_lines, [.81, .84, .89]), "Reference"))
    return items


def print_poses(poses):
    print("Front reference frame; scene unit: meters; table unit: mm.")
    print("OpenCV convention: +X right, +Y down, +Z optical forward.")
    print(f"{'Camera':<8} {'X (mm)':>10} {'Y (mm)':>10} {'Z (mm)':>10} {'Distance (mm)':>15}")
    for p in poses:
        x, y, z = p.center * 1000
        print(f"{p.name:<8} {x:10.3f} {y:10.3f} {z:10.3f} {np.linalg.norm(p.center)*1000:15.3f}")


def show_preview(poses, geometry, screenshot=None):
    # The legacy OpenGL visualizer avoids the Windows Vulkan shutdown issue
    # in the Open3D 0.20 GUI backend (upstream issue #7593).
    window = o3d.visualization.VisualizerWithKeyCallback()
    if not window.create_window("Four cameras | front frame | 1:3D  2:XZ  3:XY  L:labels  Q:quit", 1440, 960):
        raise RuntimeError("Cannot create Open3D window; a desktop OpenGL context is required")
    labels = []
    for pose in poses:
        mesh = o3d.t.geometry.TriangleMesh.create_text(pose.name, depth=.1).to_legacy()
        mesh.translate(-mesh.get_min_bound())
        mesh.scale(.001, center=[0, 0, 0])
        mesh.paint_uniform_color(COLORS[pose.name])
        local = np.asarray(mesh.vertices).copy()
        # Screen-facing labels are positioned relative to each optical center.
        local += [.012, .012, 0]
        labels.append((mesh, local, pose.center))
    centers = np.array([p.center for p in poses])
    target = (centers.min(axis=0) + centers.max(axis=0)) / 2
    for _, geom, _ in geometry:
        window.add_geometry(geom)
    for mesh, _, _ in labels:
        window.add_geometry(mesh, reset_bounding_box=False)
    options = window.get_render_option()
    options.background_color = np.array([.965, .975, .99])
    options.line_width = 2
    options.light_on = False
    options.mesh_show_back_face = True
    previous_rotation = None
    labels_visible = True

    def update_labels(vis):
        nonlocal previous_rotation
        R = vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic[:3, :3]
        if previous_rotation is not None and np.allclose(R, previous_rotation, atol=1e-7):
            return False
        previous_rotation = R.copy()
        world_from_text = R.T @ np.diag([1, -1, -1])
        for mesh, local, center in labels:
            mesh.vertices = o3d.utility.Vector3dVector(local @ world_from_text.T + center)
            if labels_visible:
                vis.update_geometry(mesh)
        return False

    def set_view(front, up):
        def callback(vis):
            control = vis.get_view_control()
            control.set_lookat(target)
            control.set_front(np.asarray(front) / np.linalg.norm(front))
            control.set_up(up)
            control.set_zoom(.72)
            update_labels(vis)
            return False
        return callback

    views = [set_view([-1.4, -.6, -1.8], [0, -1, 0]),
             set_view([0, -1, 0], [0, 0, 1]),
             set_view([0, 0, -1], [0, -1, 0])]
    for key, callback in zip("123", views):
        window.register_key_callback(ord(key), callback)

    def toggle_labels(vis):
        nonlocal labels_visible, previous_rotation
        labels_visible = not labels_visible
        for mesh, _, _ in labels:
            if labels_visible:
                vis.add_geometry(mesh, reset_bounding_box=False)
            else:
                vis.remove_geometry(mesh, reset_bounding_box=False)
        previous_rotation = None
        update_labels(vis)
        return False

    window.register_key_callback(ord("L"), toggle_labels)
    window.register_animation_callback(update_labels)
    views[0](window)
    print("Mouse drag: orbit; Ctrl+drag: pan; wheel: zoom.")
    print("Keys: 1=3D/reset, 2=XZ, 3=XY, L=toggle labels, Q/Esc=quit.")
    print("Local axes: X red, Y green, Z blue. Colored long arrows: optical axes.")
    print("Wireframes illustrate orientation only; they are not calibrated FOVs.")
    try:
        if screenshot is None:
            window.run()
        else:
            screenshot = screenshot.resolve()
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            # Exercise every preset plus label hiding/restoration before capture.
            for view in [*views, views[0]]:
                view(window)
                window.poll_events()
                window.update_renderer()
            toggle_labels(window)
            toggle_labels(window)
            start = time.monotonic()
            while time.monotonic() - start < 1:
                if not window.poll_events():
                    raise RuntimeError("Window closed before rendering check")
                update_labels(window)
                window.update_renderer()
                time.sleep(.01)
            window.capture_screen_image(str(screenshot), do_render=True)
            if not screenshot.is_file() or screenshot.stat().st_size == 0:
                raise RuntimeError("GUI screenshot was not created")
            print(f"GUI rendering check passed: {screenshot}")
    finally:
        window.destroy_window()


def positive_float(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path(__file__).with_name("rig_extrinsics.yaml"))
    parser.add_argument("--axis-length", type=positive_float, default=.04, help="Coordinate axis length in meters (default: 0.04)")
    parser.add_argument("--check", action="store_true", help="Validate transforms and build geometry without opening a window")
    parser.add_argument("--smoke-test", type=Path, metavar="PNG", help="Open the GUI, export a screenshot, and close automatically")
    args = parser.parse_args()
    if args.check and args.smoke_test:
        parser.error("--check and --smoke-test cannot be combined")
    poses = load_poses(args.input)
    geometry = build_geometry(poses, args.axis_length)
    print_poses(poses)
    if args.check:
        for name, geom, _ in geometry:
            if geom.is_empty() or not np.isfinite(geom.get_min_bound()).all() or not np.isfinite(geom.get_max_bound()).all():
                raise ValueError(f"Invalid geometry: {name}")
        print(f"Validated 4 poses and {len(geometry)} geometries. Open3D {o3d.__version__}.")
        return
    show_preview(poses, geometry, args.smoke_test)


if __name__ == "__main__":
    main()
