"""只读输入图像/内参，诊断结果写入 manual_rig_calibration/image_diagnostics。"""
import csv
import cv2
import numpy as np
import calibrate_rig_manual as c

OUT = c.OUTPUT_DIR / 'image_diagnostics'
OUT.mkdir(exist_ok=True)


def best_pose(obj, pts, K, D):
    candidates = []
    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
        try:
            result = cv2.solvePnPGeneric(obj, pts, K, D, flags=flag)
            for r, t in zip(result[1], result[2]):
                r, t = cv2.solvePnPRefineLM(obj, pts, K, D, r, t)
                pred = cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
                error = np.linalg.norm(pred - pts, axis=1)
                candidates.append((float(np.sqrt(np.mean(error**2))), r, t, pred, error))
        except cv2.error:
            pass
    return min(candidates, key=lambda x: x[0])


def main():
    rows = []
    selected = {'left': {3, 4, 12}, 'right': {4, 9, 12}, 'mid': {1, 8, 12}}
    for side, target_dir, ref_dir, K, D, ref in [
        ('left', c.LEFT_DIR, c.CENTER_LEFT_DIR, c.K_LEFT, c.D_LEFT, c.CENTER_CAMERA),
        ('right', c.RIGHT_DIR, c.CENTER_RIGHT_DIR, c.K_RIGHT, c.D_RIGHT, c.CENTER_CAMERA),
        ('mid', c.FRONT_MID_DIR, c.FRONT_DIR, c.K_CENTER, c.D_CENTER, c.FRONT_CAMERA),
    ]:
        with open(c.OUTPUT_DIR / f'{ref.name}_{side}_pair_metrics.csv', encoding='utf-8-sig') as f:
            metrics = list(csv.DictReader(f))
        tiles = []
        for metric in metrics:
            idx = int(metric['pair_index'])
            a = cv2.imread(str(ref_dir / metric[f'{ref.name}_filename']))
            b = cv2.imread(str(target_dir / metric[f'{side}_filename']))
            ma, mb = c.detect_charuco(a, ref.flip), c.detect_charuco(b, True)
            ids = sorted(set(ma) & set(mb))
            obj = c.BOARD_POINTS[ids]
            panels = []
            for name, im, pointmap, Ki, Di, flip in [(ref.name, a, ma, ref.K, ref.D, ref.flip), (side, b, mb, K, D, True)]:
                if flip:
                    im = cv2.flip(im, 1)
                pts = np.array([pointmap[i] for i in ids], dtype=np.float64)
                baseline = c.solve_pose(obj, pts, Ki, Di)['rmse']
                best = best_pose(obj, pts, Ki, Di)
                zero = best_pose(obj, pts, Ki, np.zeros_like(Di))[0]
                mirrored_K, mirrored_D = Ki.copy(), Di.copy().reshape(-1)
                mirrored_K[0, 2] = im.shape[1] - 1 - Ki[0, 2]
                mirrored_D[3] *= -1
                mirrored = best_pose(obj, pts, mirrored_K, mirrored_D)[0]
                rows.append(dict(pair=f'{ref.name}-{side}', index=idx, camera=name,
                    count=len(ids), x_min=pts[:,0].min(), x_max=pts[:,0].max(),
                    y_min=pts[:,1].min(), y_max=pts[:,1].max(),
                    original_rmse=baseline, multistart_rmse=best[0],
                    no_distortion_rmse=zero, mirrored_intrinsics_rmse=mirrored,
                    dt_ms=metric['dt_ms']))
                if idx in selected[side]:
                    for pt, pred in zip(pts, best[3]):
                        xy = tuple(np.rint(pt).astype(int))
                        cv2.circle(im, xy, 6, (0, 255, 0), 2)
                        # 红色箭头将残差放大 8 倍，便于看空间趋势。
                        end = tuple(np.rint(pt + 8 * (pred - pt)).astype(int))
                        cv2.arrowedLine(im, xy, end, (0, 0, 255), 2, tipLength=.2)
                    panel = cv2.resize(im, (960, 540))
                    cv2.rectangle(panel, (0, 0), (960, 58), (20, 20, 20), -1)
                    cv2.putText(panel, f'{ref.name}-{side} #{idx} | {name} | PnP {baseline:.2f}px / best {best[0]:.2f}px',
                                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(panel, 'Green: detected; red: reprojection residual x8', (12, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1, cv2.LINE_AA)
                    panels.append(panel)
            if panels:
                tiles.append(np.hstack(panels))
        cv2.imwrite(str(OUT / f'{ref.name}_{side}_examples.jpg'), np.vstack(tiles))
    with open(OUT / 'per_image_diagnostics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for pair in sorted(set(r['pair'] for r in rows)):
        for camera in sorted(set(r['camera'] for r in rows if r['pair'] == pair)):
            subset = [r for r in rows if r['pair'] == pair and r['camera'] == camera]
            print(pair, camera, 'count', len(subset))
            for key in ('original_rmse', 'multistart_rmse', 'no_distortion_rmse', 'mirrored_intrinsics_rmse', 'x_min', 'x_max'):
                print(key, 'median/min/max', np.median([r[key] for r in subset]), min(r[key] for r in subset), max(r[key] for r in subset))


if __name__ == '__main__':
    main()
