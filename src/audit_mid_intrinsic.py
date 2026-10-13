"""复现用户指定内参程序并验证边缘表现；仅写入 intrinsic_audit_mid02。"""
import contextlib
import hashlib
import json
import os
import runpy
from pathlib import Path

import cv2
import numpy as np

from diagnose_pair_images import best_pose

ROOT = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/111/Desktop/Data/标定/mid-02/charuco_intrinsic_calib.py')
OUT = ROOT / 'intrinsic_audit_mid02'


def main():
    OUT.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    # 原程序输入相对 __file__，输出相对 cwd；切换至独立目录安全复现原程序。
    previous_cwd = Path.cwd()
    try:
        os.chdir(OUT)
        with open(OUT / 'original_reproduction.log', 'w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log):
                g = runpy.run_path(str(SOURCE), run_name='__intrinsic_audit__')
    finally:
        os.chdir(previous_cwd)
    objects, pixels = g['all_object_points'], g['all_image_points']
    K0, D0 = g['K'], g['D']
    fs = cv2.FileStorage(str(ROOT / 'mid_camera_intrinsic.yaml'), cv2.FILE_STORAGE_READ)
    try:
        np.testing.assert_allclose(K0, fs.getNode('camera_matrix').mat(), atol=1e-8)
        np.testing.assert_allclose(D0, fs.getNode('distortion_coefficients').mat(), atol=1e-8)
    finally:
        fs.release()
    training = np.vstack(pixels)
    hull_fraction = cv2.contourArea(cv2.convexHull(training)) / np.prod(g['IMAGE_SIZE'])
    same_files = []
    for p in g['image_files']:
        other = ROOT / 'images/front-mid' / p.name
        if other.exists() and hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(other.read_bytes()).digest():
            same_files.append(p.name)

    edge_views = {}
    for group in ('mid-left', 'mid-right'):
        views = []
        for path in sorted((ROOT / 'images' / group).glob('*.jpg')):
            im = g['read_image'](path)
            im = cv2.flip(im, 1)
            corners, ids, _, _ = g['detector'].detectBoard(im)
            if ids is not None and len(ids) >= 6:
                ids = ids.ravel()
                obj = g['board'].getChessboardCorners()[ids].astype(np.float64)
                pts = corners.reshape(-1, 2).astype(np.float64)
                views.append((path.name, obj, pts))
        edge_views[group] = views
    model_results = []

    def evaluate(label, result):
        rms, K, D = result[:3]
        row = dict(model=label, train_rms_px=float(rms), K=K.tolist(), D=D.ravel().tolist())
        if len(result) >= 8:
            row['std_intrinsics'] = result[5].ravel().tolist()
        for group, views in edge_views.items():
            errors = [best_pose(obj, pts, K, D)[0] for _, obj, pts in views]
            row[group] = dict(count=len(errors), median_pnp_px=float(np.median(errors)),
                              max_pnp_px=float(np.max(errors)), per_view=dict(zip([v[0] for v in views], errors)))
        model_results.append(row)
        print(label, 'training RMS', rms, 'D', D.ravel(), flush=True)
        for group in edge_views:
            print(group, row[group]['count'], 'median/max', row[group]['median_pnp_px'], row[group]['max_pnp_px'], flush=True)
        return row

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-10)
    evaluate('original_default', cv2.calibrateCameraExtended(objects, pixels, g['IMAGE_SIZE'], None, None))
    evaluate('full_300_iterations', cv2.calibrateCameraExtended(objects, pixels, g['IMAGE_SIZE'], None, None, criteria=criteria))
    evaluate('fix_k3', cv2.calibrateCameraExtended(objects, pixels, g['IMAGE_SIZE'], None, None, flags=cv2.CALIB_FIX_K3, criteria=criteria))
    flags = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    evaluate('fit_K_with_zero_D', cv2.calibrateCameraExtended(objects, pixels, g['IMAGE_SIZE'], None, None, flags=flags, criteria=criteria))
    evaluate('original_K_zero_D', (float('nan'), K0, np.zeros_like(D0)))
    # 仅关闭 ArUco marker 的角点 refinement，ChArUco 角点流程仍由 detectBoard 处理。
    parameters = g['parameters']
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    detector = cv2.aruco.CharucoDetector(g['board'], detectorParams=parameters)
    objects_none, pixels_none = [], []
    for path in g['image_files']:
        im = cv2.flip(g['read_image'](path), 1)
        corners, ids, _, _ = detector.detectBoard(im)
        if ids is not None and len(ids) >= 6:
            objects_none.append(g['board'].getChessboardCorners()[ids.ravel()].astype(np.float32))
            pixels_none.append(corners.reshape(-1, 2).astype(np.float32))
    evaluate('marker_refine_none_default', cv2.calibrateCameraExtended(objects_none, pixels_none, g['IMAGE_SIZE'], None, None))

    leave_out = []
    for i, name in enumerate(g['used_image_files']):
        fit = cv2.calibrateCamera(objects[:i]+objects[i+1:], pixels[:i]+pixels[i+1:], g['IMAGE_SIZE'], None, None)
        leave_out.append(dict(omitted=name, train_rms_px=float(fit[0]), K=fit[1].tolist(), D=fit[2].ravel().tolist()))
    report = dict(source=str(SOURCE), source_sha256=source_hash, exact_front_mid_image_matches=len(same_files),
        image_size=g['IMAGE_SIZE'], valid_images=len(pixels), corner_count=len(training),
        training_min_xy=training.min(axis=0).tolist(), training_max_xy=training.max(axis=0).tolist(),
        training_convex_hull_fraction=float(hull_fraction), models=model_results, leave_one_out=leave_out)
    # JSON 不写非标准 NaN。
    model_results[4]['train_rms_px'] = None
    (OUT / 'audit_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('coverage', report['training_min_xy'], report['training_max_xy'], 'hull%', 100*hull_fraction)
    print('exact matching front-mid images', len(same_files))
    print('leave-one-out k3 range', min(r['D'][4] for r in leave_out), max(r['D'][4] for r in leave_out))

    os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.scatter(training[:,0], training[:,1], s=8, alpha=.65, label='Intrinsic training (15 views)', color='#2563eb')
    for group, color in [('mid-left', '#dc2626'), ('mid-right', '#16a34a')]:
        pts = np.vstack([v[2] for v in edge_views[group]])
        ax.scatter(pts[:,0], pts[:,1], s=7, alpha=.45, label=f'{group} validation', color=color)
    ax.set(xlim=(0,2688), ylim=(1512,0), xlabel='Image x (px)', ylabel='Image y (px)',
           title='Mid camera: intrinsic training coverage vs. stereo calibration regions')
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.legend(loc='lower center')
    fig.tight_layout()
    fig.savefig(OUT / 'corner_coverage.png', dpi=160)
    plt.close(fig)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    print('Audit complete. Source and existing calibration files were not modified.')


if __name__ == '__main__':
    main()
