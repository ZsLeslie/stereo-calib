"""在项目根目录运行：python -m unittest discover -s src -p test_calibrate_rig_manual.py -v"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import calibrate_rig_manual as calib


class RigCalibrationTests(unittest.TestCase):
    def transform(self, rotation, translation):
        R, _ = cv2.Rodrigues(np.array(rotation, dtype=np.float64))
        return calib.make_transform(R, np.array(translation, dtype=np.float64))

    def test_chain_and_inverse_with_noncommuting_rotations(self):
        mid = self.transform([0.12, -0.3, 0.07], [0.2, -0.08, 0.13])
        left = self.transform([-0.1, 0.45, 0.05], [-0.15, 0.03, 0.02])
        right = self.transform([0.05, -0.4, -0.08], [0.16, -0.02, 0.01])
        transforms = calib.compose_front_transforms(mid, left, right)
        point_front = np.array([0.17, -0.24, 1.8, 1.0])
        np.testing.assert_allclose(transforms['front'], np.eye(4))
        for name, side in [('left', left), ('right', right)]:
            H = transforms[name]
            point_side = side @ (mid @ point_front)
            np.testing.assert_allclose(H @ point_front, point_side, atol=1e-12)
            np.testing.assert_allclose(np.linalg.inv(H) @ point_side, point_front, atol=1e-12)
            np.testing.assert_allclose(H @ np.linalg.inv(H)[:, 3], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(
            transforms['right'] @ np.linalg.inv(transforms['left']),
            right @ np.linalg.inv(left), atol=1e-12,
        )

    def test_different_resolutions_and_reference_intrinsics(self):
        # 使用已知外参生成多视角投影；前相机的尺寸/内参与中相机明显不同。
        ref = calib.FRONT_CAMERA
        K, D = calib.K_CENTER, calib.D_CENTER
        expected = self.transform([0.03, -0.12, 0.01], [0.15, 0.02, 0.04])
        dataset = []
        rng = np.random.default_rng(8)
        for i in range(12):
            rvec = rng.uniform(-0.3, 0.3, 3)
            R, _ = cv2.Rodrigues(rvec)
            t = np.array([rng.uniform(-0.2, 0.0), rng.uniform(-0.15, 0), rng.uniform(0.8, 1.6)])
            side_R = expected[:3, :3] @ R
            side_t = expected[:3, :3] @ t + expected[:3, 3]
            center_pts, _ = cv2.projectPoints(calib.BOARD_POINTS, rvec, t, ref.K, ref.D)
            side_pts, _ = cv2.projectPoints(calib.BOARD_POINTS, cv2.Rodrigues(side_R)[0], side_t, K, D)
            dataset.append({
                'pair_index': i + 1,
                'object_points': calib.BOARD_POINTS.copy(),
                'center_points': center_pts.reshape(-1, 2),
                'side_points': side_pts.reshape(-1, 2),
            })
        K_before, D_before = ref.K.copy(), ref.D.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            result, active = calib.robust_stereo_calibrate('mid', dataset, K, D, reference=ref)
        np.testing.assert_allclose(result['R'], expected[:3, :3], atol=2e-6)
        np.testing.assert_allclose(result['T'], expected[:3, 3], atol=2e-6)
        self.assertLess(result['rms'], 1e-3)
        self.assertEqual(len(active), len(dataset))
        calib.calculate_final_metrics(dataset, result, K, D, reference=ref)
        for item in dataset:
            self.assertLess(item['final_cross_reproj_px'], 1e-3)
            self.assertLess(item['final_epipolar_px'], 1e-3)
        np.testing.assert_array_equal(ref.K, K_before)
        np.testing.assert_array_equal(ref.D, D_before)

    def test_rig_yaml_front_reference_and_legacy_mid_transforms(self):
        mid = self.transform([0.1, 0.2, 0.03], [0.1, 0.0, 0.05])
        left = self.transform([0.04, 0.4, 0.0], [-0.2, 0.0, 0.01])
        right = self.transform([0.02, -0.4, 0.01], [0.2, 0.0, 0.02])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(calib, 'OUTPUT_DIR', Path(directory)), contextlib.redirect_stdout(io.StringIO()):
                path = calib.save_front_rig(mid, left, right)
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            try:
                self.assertEqual(fs.getNode('reference_camera').string(), 'front')
                np.testing.assert_allclose(fs.getNode('H_front_from_front').mat(), np.eye(4))
                np.testing.assert_allclose(fs.getNode('H_left_from_center').mat(), left)
                np.testing.assert_allclose(fs.getNode('H_right_from_center').mat(), right)
                for name, H in [('mid', mid), ('left', left @ mid), ('right', right @ mid)]:
                    np.testing.assert_allclose(fs.getNode(f'H_{name}_from_front').mat(), H)
                    np.testing.assert_allclose(fs.getNode(f'H_front_from_{name}').mat(), np.linalg.inv(H))
                    np.testing.assert_allclose(fs.getNode(f'{name}_camera_center_in_front_m').mat().ravel(), np.linalg.inv(H)[:3, 3])
            finally:
                fs.release()
            for name in ('left', 'right'):
                fs = cv2.FileStorage(str(Path(directory) / f'front_to_{name}.yaml'), cv2.FILE_STORAGE_READ)
                try:
                    self.assertEqual(fs.getNode('calibration_method').string(), 'composed_via_mid')
                    self.assertTrue(fs.getNode('stereo_rms_px').empty())
                finally:
                    fs.release()


if __name__ == '__main__':
    unittest.main()
