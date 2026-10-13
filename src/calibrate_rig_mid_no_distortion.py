"""固定原标定的保留图像对，仅令中相机 D=0，比较基线和旋转。

在项目根目录运行：python calibrate_rig_mid_no_distortion.py
结果写入 manual_rig_calibration_mid_no_distortion，不覆盖原结果。
"""
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import calibrate_rig_manual as c


def read_matrix(path, key):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f'Cannot open {path}')
    try:
        value = fs.getNode(key).mat()
        if value is None:
            raise RuntimeError(f'Missing {key} in {path}')
        return value
    finally:
        fs.release()


def compare(pair, old, new, count=None, old_rms=None, new_rms=None):
    old_baseline = float(np.linalg.norm(old[:3, 3]) * 1000)
    new_baseline = float(np.linalg.norm(new[:3, 3]) * 1000)
    old_angle = c.rotation_angle_deg(old[:3, :3])
    new_angle = c.rotation_angle_deg(new[:3, :3])
    return dict(
        pair=pair, num_fixed_pairs=count,
        old_baseline_mm=old_baseline, new_baseline_mm=new_baseline,
        baseline_change_mm=new_baseline-old_baseline,
        baseline_change_percent=100*(new_baseline-old_baseline)/old_baseline,
        old_rotation_angle_deg=old_angle, new_rotation_angle_deg=new_angle,
        rotation_angle_change_deg=new_angle-old_angle,
        rotation_matrix_difference_deg=c.rotation_angle_deg(new[:3, :3] @ old[:3, :3].T),
        old_optical_axis_angle_deg=float(np.degrees(np.arccos(np.clip(old[2, 2], -1, 1)))),
        new_optical_axis_angle_deg=float(np.degrees(np.arccos(np.clip(new[2, 2], -1, 1)))),
        old_stereo_rms_px=old_rms, new_stereo_rms_px=new_rms,
    )


def main():
    original_dir = c.OUTPUT_DIR
    output_dir = Path('manual_rig_calibration_mid_no_distortion')
    output_dir.mkdir(exist_ok=True)
    protected = list(original_dir.glob('*.yaml')) + list(original_dir.glob('*pair_metrics.csv'))
    protected += list(Path('.').glob('*camera_intrinsic.yaml'))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    reference_mid = replace(c.CENTER_CAMERA, D=np.zeros_like(c.D_CENTER))
    jobs = [
        (c.FRONT_CAMERA, c.FRONT_CAMERA, 'mid', c.FRONT_MID_DIR, c.FRONT_DIR,
         c.K_CENTER, c.D_CENTER, reference_mid.D, c.SIZE_CENTER, c.FLIP_CENTER),
        (c.CENTER_CAMERA, reference_mid, 'left', c.LEFT_DIR, c.CENTER_LEFT_DIR,
         c.K_LEFT, c.D_LEFT, c.D_LEFT, c.SIZE_LEFT, c.FLIP_LEFT),
        (c.CENTER_CAMERA, reference_mid, 'right', c.RIGHT_DIR, c.CENTER_RIGHT_DIR,
         c.K_RIGHT, c.D_RIGHT, c.D_RIGHT, c.SIZE_RIGHT, c.FLIP_RIGHT),
    ]
    comparison, transforms, selection = [], {}, {}
    try:
        c.OUTPUT_DIR = output_dir
        for old_ref, ref, name, target_dir, reference_dir, K, old_D, new_D, size, flip in jobs:
            with open(original_dir / f'{ref.name}_{name}_pair_metrics.csv', encoding='utf-8-sig') as f:
                fixed_ids = {int(r['pair_index']) for r in csv.DictReader(f) if r['final_inlier'] == '1'}
            pairs = c.match_images_by_timestamp(target_dir, reference_dir, name, ref.name)
            dataset = c.build_dataset(name, pairs, K, new_D, flip, reference=ref, side_size=size)
            active = [i for i, item in enumerate(dataset) if item['pair_index'] in fixed_ids]
            if {dataset[i]['pair_index'] for i in active} != fixed_ids:
                raise RuntimeError(f'{name}: cannot reproduce original inlier set')
            if len(active) < c.MIN_FINAL_PAIRS:
                raise RuntimeError(f'{name}: insufficient original pairs')
            selection[name] = sorted(fixed_ids)
            # 同样的角点以原参数重算，应复现原文件，确保比较基准未漂移。
            old_result = c.stereo_calibrate(dataset, active, K, old_D, reference=old_ref)
            old_H = read_matrix(original_dir / f'{ref.name}_to_{name}.yaml', f'H_{name}_from_{ref.name}')
            np.testing.assert_allclose(c.make_transform(old_result['R'], old_result['T']), old_H, atol=1e-8)
            result = c.stereo_calibrate(dataset, active, K, new_D, reference=ref)
            # 不重新筛选，避免 D 和图像集合同时改变；-1 表示继承原排除决定。
            for i, item in enumerate(dataset):
                item['rejected_iteration'] = 0 if i in active else -1
            _, _, H = c.save_result(name, dataset, result, set(active), K, new_D, reference=ref)
            transforms[name] = H
            c.print_final_result(name, result, len(active), reference=ref)
            source = 'mid' if ref.name == 'center' else ref.name
            comparison.append(compare(f'{source}->{name}', old_H, H, len(active), old_result['rms'], result['rms']))

        rig_path = c.save_front_rig(transforms['mid'], transforms['left'], transforms['right'])
        for name in ('left', 'right'):
            key = f'H_{name}_from_front'
            comparison.append(compare(f'front->{name} (via mid)',
                read_matrix(original_dir / 'rig_extrinsics.yaml', key), read_matrix(rig_path, key)))

        settings = dict(mode='mid_D_zero', selection='same_final_inliers_as_original', fixed_pair_ids=selection,
            intrinsic_K='unchanged', front_left_right_D='unchanged', mid_D=reference_mid.D.tolist(),
            image_preprocessing='unchanged', excluded_pair_iteration=-1,
            original_result_directory=str(original_dir),
            original_file_sha256={str(p): digest for p, digest in hashes.items()})
        (output_dir / 'experiment_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        (output_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        with open(output_dir / 'comparison.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        report = [
            '# 中相机 D=0 标定对照', '',
            '只把中相机畸变 D 设为 0；K、其他相机 D、图像方向不变。使用原标定最终保留的图像对，未重新筛选。front-mid 也用中相机 D=0 重新标定后，再组合 front-left/right。', '',
            '| 相机对 | 原基线 mm | 新基线 mm | 变化 mm | 原旋转角 ° | 新旋转角 ° | 变化 ° |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
        for r in comparison:
            report.append(f"| {r['pair']} | {r['old_baseline_mm']:.3f} | {r['new_baseline_mm']:.3f} | {r['baseline_change_mm']:+.3f} | {r['old_rotation_angle_deg']:.3f} | {r['new_rotation_angle_deg']:.3f} | {r['rotation_angle_change_deg']:+.3f} |")
        report += ['', '旋转角为完整三维旋转的轴角幅值，不是光轴夹角。comparison.csv 另存光轴夹角及 R_new @ R_old.T 的旋转角，后者表示两次估计的实际姿态差，不能用两个轴角幅值之差替代。', '',
                   '| 相机对 | 固定图像对数 | 原 RMS px | 新 RMS px | 两次估计姿态差 ° |',
                   '| --- | ---: | ---: | ---: | ---: |']
        for r in comparison[:3]:
            report.append(f"| {r['pair']} | {r['num_fixed_pairs']} | {r['old_stereo_rms_px']:.6f} | {r['new_stereo_rms_px']:.6f} | {r['rotation_matrix_difference_deg']:.3f} |")
        report += ['', '这是忽略中相机畸变的对照结果。误差下降不等于已经验证真实基线或真实姿态，需要实际测量或独立数据检验。', '',
                   '输出的外参字段与原结果保持一致，mid 在旧字段中称作 center。使用这些结果投影时须使用中相机 D=0；对应设置及固定图像对编号见 experiment_settings.json。', '',
                   '复现：在项目根目录运行 `.\\.venv\\Scripts\\python.exe calibrate_rig_mid_no_distortion.py`。']
        (output_dir / 'comparison.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
        for p, digest in hashes.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f'Original changed: {p}'
        print('\n'.join(report))
        print('Verified: original calibration reproduced; original files unchanged.')
    finally:
        c.OUTPUT_DIR = original_dir


if __name__ == '__main__':
    main()
