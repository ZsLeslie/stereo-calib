"""畸变参数对照实验；仅写入 image_diagnostics，不覆盖正式外参或内参。"""
import contextlib
import csv
import io
import json
from dataclasses import replace
import numpy as np
import calibrate_rig_manual as c

results = []
for name, side_dir, ref_dir, K, D, flip in [
    ('left', c.LEFT_DIR, c.CENTER_LEFT_DIR, c.K_LEFT, c.D_LEFT, c.FLIP_LEFT),
    ('right', c.RIGHT_DIR, c.CENTER_RIGHT_DIR, c.K_RIGHT, c.D_RIGHT, c.FLIP_RIGHT),
]:
    with contextlib.redirect_stdout(io.StringIO()):
        pairs = c.match_images_by_timestamp(side_dir, ref_dir, name)
        dataset = c.build_dataset(name, pairs, K, D, flip)
    with open(c.OUTPUT_DIR / f'center_{name}_pair_metrics.csv', encoding='utf-8-sig') as f:
        inlier_ids = {int(r['pair_index']) for r in csv.DictReader(f) if r['final_inlier'] == '1'}
    for subset, indices in [('all_detected', list(range(len(dataset)))),
                            ('same_final_inliers', [i for i, r in enumerate(dataset) if r['pair_index'] in inlier_ids])]:
        for mode, ref, side_D in [
            ('original', c.CENTER_CAMERA, D),
            ('mid_D_zero', replace(c.CENTER_CAMERA, D=np.zeros_like(c.D_CENTER)), D),
            ('both_D_zero', replace(c.CENTER_CAMERA, D=np.zeros_like(c.D_CENTER)), np.zeros_like(D)),
        ]:
            result = c.stereo_calibrate(dataset, indices, K, side_D, reference=ref)
            row = dict(pair=f'mid-{name}', subset=subset, count=len(indices), mode=mode,
                       stereo_rms_px=result['rms'], baseline_mm=float(np.linalg.norm(result['T'])*1000))
            results.append(row)
            print(row, flush=True)

k1, k2, p1, p2, k3 = c.D_CENTER.ravel()
roots = np.roots([7*k3, 5*k2, 3*k1, 1])
for s in roots:
    if abs(s.imag) < 1e-9 and s.real > 0:
        r = float(np.sqrt(s.real))
        print('mid radial turning point ru, rd:', r, r*(1+k1*r*r+k2*r**4+k3*r**6))
(c.OUTPUT_DIR/'image_diagnostics'/'stereo_ablation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
