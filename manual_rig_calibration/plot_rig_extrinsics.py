"""Plot OpenCV rig extrinsics in the front camera frame (numpy + matplotlib)."""
from pathlib import Path
import argparse
import csv
import os
import re

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


def read_matrix(text, key):
    match = re.search(r"^" + re.escape(key) + r": !!opencv-matrix\s+rows:\s*(\d+)\s+cols:\s*(\d+)\s+dt:\s*\w+\s+data:\s*\[([^]]+)\]", text, re.M)
    if not match:
        raise ValueError(f"Missing matrix: {key}")
    rows, cols, data = match.groups()
    return np.array([float(x) for x in data.split(",")]).reshape(int(rows), int(cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path(__file__).with_name("rig_extrinsics.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("rig_visualization"))
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    if not re.search(r"^translation_unit:\s*m\s*$", text, re.M):
        raise ValueError("Expected translation_unit: m")
    names = ["front", "mid", "left", "right"]
    labels = ["前相机 front", "中相机 mid", "左相机 left", "右相机 right"]
    colors = ["#202f4d", "#a86c09", "#007d83", "#8b50b5"]
    poses = []
    for name in names:
        H = read_matrix(text, f"H_{name}_from_front")
        R, t = H[:3, :3], H[:3, 3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1, atol=1e-8)
        np.testing.assert_allclose(H[3], [0, 0, 0, 1], atol=1e-8)
        center = -R.T @ t
        inverse = np.eye(4)
        inverse[:3, :3], inverse[:3, 3] = R.T, center
        np.testing.assert_allclose(inverse, read_matrix(text, f"H_front_from_{name}"), atol=1e-8)
        np.testing.assert_allclose(center, read_matrix(text, f"{name}_camera_center_in_front_m").ravel(), atol=1e-8)
        poses.append((center * 1000, R.T))

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "axes.edgecolor": "#b8c1cb", "text.color": "#243248", "axes.labelcolor": "#34465e", "xtick.color": "#526278", "ytick.color": "#526278", "svg.fonttype": "path"})
    fig = plt.figure(figsize=(17, 11), facecolor="#f7f9fc")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1], height_ratios=[1, 1, .63], left=.035, right=.97, bottom=.10, top=.87, wspace=.14, hspace=.43)
    fig.text(.045, .947, "四相机位置与朝向", fontsize=25, weight="bold")
    fig.text(.045, .908, "参考坐标系：前相机 front  ·  位置单位：mm  ·  圆点为光心，箭头为各相机坐标轴", fontsize=12, color="#586c85")
    ax = fig.add_subplot(gs[:2, 0], projection="3d", facecolor="#f7f9fc")
    # Display (X, Z, -Y) so the usual OpenCV up direction appears upright.
    display = lambda p: np.asarray(p)[..., [0, 2, 1]] * [1, 1, -1]
    axis_colors = ["#d54b47", "#2c9758", "#2867cd"]
    for (center, rotation), label, color in zip(poses, labels, colors):
        p = display(center)
        ax.scatter(*p, s=65, c=color, depthshade=False)
        for j, axis_color in enumerate(axis_colors):
            d = display(rotation[:, j] * (62 if j == 2 else 32))
            ax.quiver(*p, *d, color=axis_color, linewidth=2, arrow_length_ratio=.18)
        # Same illustrative pyramid for every camera; no intrinsics/FOV assumed.
        corners = np.array([[-15, -10, 35], [15, -10, 35], [15, 10, 35], [-15, 10, 35]])
        corners = display(center + corners @ rotation.T)
        for corner in corners:
            ax.plot(*np.stack([p, corner]).T, color=color, alpha=.55, lw=1)
        ax.plot(*np.vstack([corners, corners[0]]).T, color=color, alpha=.7, lw=1)
        offset = {"前相机 front": [8, -7, -24], "中相机 mid": [-8, 20, 15], "左相机 left": [-32, -13, 15], "右相机 right": [7, -5, -25]}[label]
        ax.text(*(p + offset), label, fontsize=11, color=color, weight="bold")
        if label != labels[0]:
            ax.plot(*np.stack([np.zeros(3), p]).T, color="#a6b0bf", ls="--", lw=1)
    ax.set(xlabel="X_front (mm)", ylabel="Z_front (mm)", zlabel="−Y_front (mm)")
    ax.set_xlim(-115, 105)
    ax.set_ylim(-150, 105)
    ax.set_zlim(-45, 330)
    ax.set_box_aspect((220, 255, 375))
    ax.view_init(elev=22, azim=-62)
    ax.set_title("三维视图 / 等比例空间坐标", loc="left", pad=14, fontsize=14, weight="bold")
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2, label=l) for c, l in zip(axis_colors, ["局部 +X", "局部 +Y", "局部 +Z（光轴）"])], loc="lower left", bbox_to_anchor=(.10, .26), ncol=3, frameon=False, fontsize=10)

    def projection(subplot, dims, title, ylabel, offsets, ylim):
        a = fig.add_subplot(subplot, facecolor="white")
        for (center, rotation), name, color, offset in zip(poses, names, colors, offsets):
            p = center[list(dims)]
            d = rotation[list(dims), 2] * 65
            a.scatter(*p, s=55, c=color, zorder=4)
            a.annotate("", xy=p+d, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=13))
            a.annotate(name, xy=p, xytext=offset, textcoords="offset points", color=color, weight="bold", fontsize=10, arrowprops=dict(arrowstyle="-", color=color, alpha=.45))
            if name != "front":
                a.plot([0, p[0]], [0, p[1]], "--", color="#c5ccd6", lw=.8, zorder=0)
        a.axhline(0, color="#d9dfe7", lw=.8)
        a.axvline(0, color="#d9dfe7", lw=.8)
        a.set(xlim=(-120, 110), ylim=ylim, xlabel="X_front (mm)", ylabel=ylabel)
        a.set_aspect("equal", adjustable="box")
        a.grid(alpha=.18)
        a.set_title(title, loc="left", fontsize=13, weight="bold", pad=10)
        return a

    projection(gs[0, 1], (0, 2), "XZ 平面投影 · 箭头为光轴投影", "Z_front (mm)", [(12, 4), (-52, 8), (-32, -19), (10, -19)], (-140, 90))
    projection(gs[1, 1], (0, 1), "XY 平面投影 · Y 轴向下", "Y_front (mm)", [(12, 0), (-8, 14), (-25, -20), (16, -18)], (45, -330))

    table_ax = fig.add_subplot(gs[2, :])
    table_ax.axis("off")
    rows = []
    for (center, rotation), label in zip(poses, labels):
        angle = np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))
        rows.append([label, *[f"{v:.2f}" for v in center], f"{np.linalg.norm(center):.2f}", f"{angle:.2f}°"])
    table = table_ax.table(cellText=rows, colLabels=["相机", "X_front / mm", "Y_front / mm", "Z_front / mm", "距前相机 / mm", "光轴与前相机夹角"], cellLoc="center", loc="center", colWidths=[.22, .14, .14, .14, .17, .19])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e6ef")
        cell.set_facecolor("#e8eef6" if r == 0 else ("white" if r % 2 else "#f1f5fa"))
        if r == 0:
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(color=colors[r-1], weight="bold")
    fig.text(.045, .055, "计算：X_cam = R · X_front + t；光心 C_front = −R^T · t；相机朝向 R_front_from_cam = R^T。", fontsize=11)
    fig.text(.045, .026, "按常用 OpenCV 约定：X 向右、Y 向下、Z 沿光轴向前。线框仅示意朝向，不代表实际视场角；输入：" + args.input.name, fontsize=10, color="#64748b")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(args.output_dir / f"four_camera_rig_front_frame.{extension}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    with (args.output_dir / "camera_poses_front_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["camera", "x_mm", "y_mm", "z_mm", "distance_to_front_mm", "optical_axis_x", "optical_axis_y", "optical_axis_z", "optical_axis_angle_to_front_deg"])
        for name, (center, rotation) in zip(names, poses):
            writer.writerow([name, *center, np.linalg.norm(center), *rotation[:, 2], np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))])
    print("Validated rotation matrices, inverse transforms, and stored camera centers.")
    for name, (center, rotation) in zip(names, poses):
        print(f"{name}: center_mm={center.round(3)}, optical_axis={rotation[:, 2].round(6)}")
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
