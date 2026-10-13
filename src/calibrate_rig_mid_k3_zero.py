"""使用内参审查中固定 k3=0 并重新拟合的 K/D，固定原保留图像对标定外参。"""
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import calibrate_rig_manual as c
from calibrate_rig_mid_no_distortion import compare, read_matrix


def main():
    audit_path = Path('intrinsic_audit_mid02/audit_results.json')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    original_model = next(m for m in audit['models'] if m['model'] == 'original_default')
    candidate = next(m for m in audit['models'] if m['model'] == 'fix_k3')
    np.testing.assert_allclose(c.K_CENTER, original_model['K'], atol=1e-8)
    np.testing.assert_allclose(c.D_CENTER.ravel(), original_model['D'], atol=1e-8)
    mid = replace(c.CENTER_CAMERA, K=np.asarray(candidate['K'], dtype=np.float64),
                  D=np.asarray(candidate['D'], dtype=np.float64).reshape(1, -1))
    assert mid.D[0, 4] == 0
    original_dir = c.OUTPUT_DIR
    output = Path('manual_rig_calibration_mid_k3_zero')
    output.mkdir(exist_ok=True)
    protected = list(Path('.').glob('*camera_intrinsic.yaml')) + list(original_dir.glob('*.yaml')) + list(original_dir.glob('*pair_metrics.csv'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    comparison, matrices, selection = [], {}, {}
    jobs = [
        (c.FRONT_CAMERA, c.FRONT_CAMERA, 'mid', c.FRONT_MID_DIR, c.FRONT_DIR,
         c.K_CENTER, c.D_CENTER, mid.K, mid.D, c.SIZE_CENTER, c.FLIP_CENTER),
        (c.CENTER_CAMERA, mid, 'left', c.LEFT_DIR, c.CENTER_LEFT_DIR,
         c.K_LEFT, c.D_LEFT, c.K_LEFT, c.D_LEFT, c.SIZE_LEFT, c.FLIP_LEFT),
        (c.CENTER_CAMERA, mid, 'right', c.RIGHT_DIR, c.CENTER_RIGHT_DIR,
         c.K_RIGHT, c.D_RIGHT, c.K_RIGHT, c.D_RIGHT, c.SIZE_RIGHT, c.FLIP_RIGHT),
    ]
    try:
        c.OUTPUT_DIR = output
        for old_ref, ref, name, target_dir, reference_dir, old_K, old_D, K, D, size, flip in jobs:
            with open(original_dir / f'{ref.name}_{name}_pair_metrics.csv', encoding='utf-8-sig') as f:
                ids = {int(r['pair_index']) for r in csv.DictReader(f) if r['final_inlier'] == '1'}
            pairs = c.match_images_by_timestamp(target_dir, reference_dir, name, ref.name)
            dataset = c.build_dataset(name, pairs, K, D, flip, reference=ref, side_size=size)
            active = [i for i, item in enumerate(dataset) if item['pair_index'] in ids]
            assert {dataset[i]['pair_index'] for i in active} == ids
            assert len(active) >= c.MIN_FINAL_PAIRS
            selection[name] = sorted(ids)
            old = c.stereo_calibrate(dataset, active, old_K, old_D, reference=old_ref)
            old_H = read_matrix(original_dir / f'{ref.name}_to_{name}.yaml', f'H_{name}_from_{ref.name}')
            np.testing.assert_allclose(c.make_transform(old['R'], old['T']), old_H, atol=1e-8)
            result = c.stereo_calibrate(dataset, active, K, D, reference=ref)
            for i, item in enumerate(dataset):
                item['rejected_iteration'] = 0 if i in active else -1
            _, _, H = c.save_result(name, dataset, result, set(active), K, D, reference=ref)
            matrices[name] = H
            c.print_final_result(name, result, len(active), reference=ref)
            source = 'mid' if ref.name == 'center' else ref.name
            comparison.append(compare(f'{source}->{name}', old_H, H, len(active), old['rms'], result['rms']))
        rig = c.save_front_rig(matrices['mid'], matrices['left'], matrices['right'])
        for name in ('left', 'right'):
            key = f'H_{name}_from_front'
            comparison.append(compare(f'front->{name} (via mid)', read_matrix(original_dir/'rig_extrinsics.yaml', key), read_matrix(rig, key)))
        fs = cv2.FileStorage(str(output/'mid_camera_intrinsic.yaml'), cv2.FILE_STORAGE_WRITE)
        fs.write('image_width', mid.size[0])
        fs.write('image_height', mid.size[1])
        fs.write('camera_matrix', mid.K)
        fs.write('distortion_coefficients', mid.D)
        fs.write('rms_reprojection_error', candidate['train_rms_px'])
        fs.write('valid_images', audit['valid_images'])
        fs.write('calibration_flags', 'CALIB_FIX_K3')
        fs.write('flip_horizontal', int(mid.flip))
        fs.release()
        settings = dict(mode='mid_k3_zero_refit', source=str(audit_path), source_model='fix_k3',
                        selection='same_final_inliers_as_current_original', pair_ids=selection,
                        mid_K=mid.K.tolist(), mid_D=mid.D.tolist(), other_intrinsics='unchanged',
                        original_file_sha256=hashes)
        (output/'experiment_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        (output/'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        with open(output/'comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        lines = ['# 中相机固定 k3=0 后的外参', '',
                 '中相机内参采用 audit_results.json 中 fix_k3 模型：固定 k3=0 后，重新拟合 K、k1、k2、p1、p2；不是直接把旧 D 的最后一项改为 0。其他相机内参不变。', '',
                 '使用当前 manual_rig_calibration 的最终保留图像对，不重新筛选。此比较基准使用的是训练 RMS=0.332366 px 的 mid-02 原内参，不是更早对话中的旧版内参。', '',
                 '| 相机对 | 基线 mm | 旋转角 ° | 光轴夹角 ° | 原基线 mm | 原旋转角 ° |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in comparison:
            lines.append(f"| {row['pair']} | {row['new_baseline_mm']:.3f} | {row['new_rotation_angle_deg']:.3f} | {row['new_optical_axis_angle_deg']:.3f} | {row['old_baseline_mm']:.3f} | {row['old_rotation_angle_deg']:.3f} |")
        lines += ['', '| 直接标定 | 固定图像对数 | 原 RMS px | 新 RMS px |', '| --- | ---: | ---: | ---: |']
        for row in comparison[:3]:
            lines.append(f"| {row['pair']} | {row['num_fixed_pairs']} | {row['old_stereo_rms_px']:.6f} | {row['new_stereo_rms_px']:.6f} |")
        lines += ['', '旋转角是完整三维旋转的轴角幅值，光轴夹角仅衡量两相机 Z 轴之间的夹角。平移使用米，表格基线为毫米。', '',
                  '完整结果见 rig_extrinsics.yaml；投影时使用此目录中的 mid_camera_intrinsic.yaml。当前模型来自覆盖有限的内参图像，结果仍需与实测基线或独立数据核对。', '',
                  '运行：`.\\.venv\\Scripts\\python.exe -X utf8 calibrate_rig_mid_k3_zero.py`。']
        (output/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines))
        for p, digest in hashes.items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
        print('Verified: original transforms reproduced; original files unchanged.')
    finally:
        c.OUTPUT_DIR = original_dir


if __name__ == '__main__':
    main()
