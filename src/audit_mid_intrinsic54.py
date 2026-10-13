"""分析新采集的中相机内参，独立输出到 intrinsic_audit_mid54。"""
import contextlib
import hashlib
import json
import os
import runpy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import calibrate_rig_manual as c
from diagnose_pair_images import best_pose

ROOT = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/111/Desktop/Data/标定/mid-02/charuco_intrinsic_calib.py')
OUT = ROOT / 'intrinsic_audit_mid54'


def main():
    OUT.mkdir(exist_ok=True)
    original_hash = hashlib.sha256((ROOT/'mid_camera_intrinsic.yaml').read_bytes()).hexdigest()
    oldcwd = Path.cwd()
    try:
        os.chdir(OUT)
        with open(OUT/'reproduction.log', 'w', encoding='utf-8') as f, contextlib.redirect_stdout(f):
            g = runpy.run_path(str(SOURCE), run_name='__audit__')
    finally:
        os.chdir(oldcwd)
    obj, pts, size = g['all_object_points'], g['all_image_points'], g['IMAGE_SIZE']
    np.testing.assert_allclose(g['K'], c.K_CENTER, atol=1e-8)
    np.testing.assert_allclose(g['D'], c.D_CENTER, atol=1e-8)
    training = np.vstack(pts)
    coverage = dict(min_xy=training.min(axis=0).tolist(), max_xy=training.max(axis=0).tolist(),
                    hull_fraction=float(cv2.contourArea(cv2.convexHull(training))/np.prod(size)))
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in g['image_files']}
    validation = {}
    overlap = {}
    for group in ('front-mid', 'mid-left', 'mid-right'):
        views = []
        overlap[group] = 0
        for path in sorted((ROOT/'images'/group).glob('*.jpg')):
            overlap[group] += int(hashlib.sha256(path.read_bytes()).hexdigest() in hashes)
            im = cv2.flip(g['read_image'](path), 1)
            corners, ids, _, _ = g['detector'].detectBoard(im)
            if ids is not None and len(ids) >= 6:
                views.append((path.name, g['board'].getChessboardCorners()[ids.ravel()].astype(np.float64),
                              corners.reshape(-1,2).astype(np.float64)))
        validation[group] = views
    previous = json.loads((ROOT/'intrinsic_audit_mid02/audit_results.json').read_text(encoding='utf-8'))
    models = []
    for old_name, new_name in [('original_default', 'old15_full'), ('fix_k3', 'old15_fix_k3')]:
        m = next(m for m in previous['models'] if m['model'] == old_name)
        models.append(dict(name=new_name, K=np.asarray(m['K']), D=np.asarray(m['D']).reshape(1,-1), train_rms=m['train_rms_px']))
    fit = cv2.calibrateCameraExtended(obj, pts, size, None, None)
    models.append(dict(name='new54_full', K=fit[1], D=fit[2], train_rms=fit[0], std_intrinsics=fit[5].ravel().tolist()))
    fixed = cv2.calibrateCameraExtended(obj, pts, size, None, None, flags=cv2.CALIB_FIX_K3)
    models.append(dict(name='new54_fix_k3', K=fixed[1], D=fixed[2], train_rms=fixed[0]))
    models.append(dict(name='new54_zero_D_same_K', K=fit[1], D=np.zeros_like(fit[2]), train_rms=None))
    for model in models:
        errors = {}
        for group, views in validation.items():
            values = [best_pose(o, p, model['K'], model['D'])[0] for _,o,p in views]
            errors[group] = dict(count=len(values), median=float(np.median(values)), max=float(np.max(values)),
                                 per_view=dict(zip([v[0] for v in views], values)))
        model['validation_pnp_px'] = errors
        print(model['name'], 'train', model['train_rms'], 'D', model['D'].ravel(), flush=True)
        print('validation medians', {k:v['median'] for k,v in errors.items()}, flush=True)

    # 三折检验：每折以约 36 张训练、18 张验证，检验模型和参数稳定性。
    folds = []
    permutation = np.random.default_rng(54).permutation(len(obj))
    for split in np.array_split(permutation, 3):
        train = [i for i in range(len(obj)) if i not in split]
        f = cv2.calibrateCamera([obj[i] for i in train], [pts[i] for i in train], size, None, None)
        errors = [best_pose(obj[i].astype(np.float64),pts[i].astype(np.float64),f[1],f[2])[0] for i in split]
        folds.append(dict(train_count=len(train), test_count=len(split), train_rms=f[0], K=f[1].tolist(), D=f[2].ravel().tolist(),
                          heldout_median=float(np.median(errors)), heldout_max=float(np.max(errors))))
        print('fold', len(folds), folds[-1], flush=True)

    # 外参对照：每一模型都使用同一组全部检测可用图像对，不进行模型相关剔除。
    stereo = []
    for name, target_dir, ref_dir, K, D, size_side, flip in [
        ('mid', c.FRONT_MID_DIR, c.FRONT_DIR, c.K_CENTER, c.D_CENTER, c.SIZE_CENTER, c.FLIP_CENTER),
        ('left', c.LEFT_DIR, c.CENTER_LEFT_DIR, c.K_LEFT, c.D_LEFT, c.SIZE_LEFT, c.FLIP_LEFT),
        ('right', c.RIGHT_DIR, c.CENTER_RIGHT_DIR, c.K_RIGHT, c.D_RIGHT, c.SIZE_RIGHT, c.FLIP_RIGHT),
    ]:
        reference = c.FRONT_CAMERA if name == 'mid' else c.CENTER_CAMERA
        with open(OUT/f'{name}_detection.log','w',encoding='utf-8') as log, contextlib.redirect_stdout(log):
            pairs = c.match_images_by_timestamp(target_dir, ref_dir, name, reference.name)
            dataset = c.build_dataset(name, pairs, K, D, flip, reference=reference, side_size=size_side)
        for model in models[:4]:
            ref = reference if name == 'mid' else replace(reference,K=model['K'],D=model['D'])
            target_K, target_D = (model['K'],model['D']) if name == 'mid' else (K,D)
            result = c.stereo_calibrate(dataset, list(range(len(dataset))), target_K, target_D, reference=ref)
            row = dict(pair='front-mid' if name=='mid' else f'mid-{name}', model=model['name'], pairs=len(dataset),
                       rms=result['rms'], baseline_mm=float(np.linalg.norm(result['T'])*1000),
                       rotation_deg=c.rotation_angle_deg(result['R']), R=result['R'].tolist(), T_m=result['T'].tolist())
            stereo.append(row)
            print('stereo', {k:v for k,v in row.items() if k not in ('R','T_m')}, flush=True)

    for model in models:
        model['K'] = model['K'].tolist()
        model['D'] = model['D'].ravel().tolist()
    report = dict(source=str(SOURCE), intrinsic_sha256=original_hash, training_report=g['report'],
                  coverage=coverage, shared_images_with_validation=overlap, models=models, folds=folds, stereo=stereo)
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('coverage',coverage,'shared',overlap,flush=True)

    os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,6.8))
    ax.scatter(training[:,0],training[:,1],s=9,alpha=.5,color='#2563eb',label=f'New intrinsic training ({len(pts)} views)')
    for group,color in [('mid-left','#dc2626'),('mid-right','#16a34a')]:
        data = np.vstack([v[2] for v in validation[group]])
        ax.scatter(data[:,0],data[:,1],s=8,alpha=.5,color=color,label=group+' validation')
    ax.set(xlim=(0,2688),ylim=(1512,0),xlabel='Image x (px)',ylabel='Image y (px)',title='New mid camera training coverage and stereo validation regions')
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.legend(loc='lower center')
    fig.tight_layout()
    fig.savefig(OUT/'coverage.png',dpi=160)
    plt.close(fig)
    assert hashlib.sha256((ROOT/'mid_camera_intrinsic.yaml').read_bytes()).hexdigest()==original_hash
    print('Complete; provided calibration files unchanged.')


if __name__ == '__main__':
    main()
