# 四相机外参标定

运行 `src/calibrate_rig_manual.py`，以前相机 front 为统一基准，以中相机 mid 为桥梁。当前流程直接使用 `images/` 中的图像，分别标定 front → mid、mid → left、mid → right，再组合成统一外参。

## 输入与运行

| 直接标定 | 参考相机图像 | 目标相机图像 |
| --- | --- | --- |
| front → mid | `images/front` | `images/front-mid` |
| mid → left | `images/mid-left` | `images/left` |
| mid → right | `images/mid-right` | `images/right` |

使用 `src/` 目录中的 `front_camera_intrinsic.yaml`、`mid_camera_intrinsic.yaml`、`left_camera_intrinsic.yaml`、`right_camera_intrinsic.yaml`。文件名中的时间戳用于一一配对，最大时间差为 100 ms。

在项目根目录运行（需要 NumPy 和包含 aruco 的 OpenCV，已用 OpenCV 4.10.0 验证）：

```powershell
python .\src\calibrate_rig_manual.py
python -m unittest discover -s src -p test_calibrate_rig_manual.py -v
```

本次已在项目 `.venv` 中准备验证环境，也可直接运行：

```powershell
.\.venv\Scripts\python.exe .\src\calibrate_rig_manual.py
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_calibrate_rig_manual.py -v
```

前相机按原图检测，`FLIP_FRONT = False`；中、左、右相机保持水平翻转。当前图像检测支持这一设置，内参也必须对应相同的图像方向。
各相机图像尺寸分别与自身内参文件核对，不缩放图像。标定使用 `CALIB_FIX_INTRINSIC` 固定内参；按 [OpenCV 文档](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html)，`imageSize` 仅用于初始化内参，因此这里可以使用不同分辨率的两台相机。

ChArUco 参数沿用原设置：7×5、方格边长 0.04665 m、标记边长 0.03265 m、DICT_5X5_100。最少共同角点 8 个，最终至少保留 10 组，沿用原有稳健筛选规则。

## 输出与变换方向

结果在 `manual_rig_calibration/`：

- `front_to_mid.yaml`：直接标定的前、中相机外参，以及 RMS、保留图像对数量。
- `center_to_left.yaml`、`center_to_right.yaml`：保留原来的中相机基准外参；`center` 是 `mid` 的旧名称。
- `front_to_left.yaml`、`front_to_right.yaml`：经中相机组合得到的外参，`calibration_method = composed_via_mid`。它们没有直接双目标定 RMS。
- `rig_extrinsics.yaml`：统一结果，`reference_camera = front`，含全部相机的正向、逆向变换及中相机基准的原有字段。
- `front_mid_pair_metrics.csv`、`center_left_pair_metrics.csv`、`center_right_pair_metrics.csv`：每对图像的检测、筛选与误差指标。`dt_ms` 为参考相机时间戳减目标相机时间戳。

`H_A_from_B` 将 B 相机坐标变换为 A 相机坐标，使用列向量和齐次坐标，平移单位为米：

```text
X_A = H_A_from_B @ X_B
H_left_from_front  = H_left_from_mid  @ H_mid_from_front
H_right_from_front = H_right_from_mid @ H_mid_from_front
```

`R_front_to_left`、`T_front_to_left_m` 是把前相机坐标转换到左相机坐标的投影外参。
若需要左相机在前相机坐标系下的位姿，使用 `H_front_from_left`，它是 `H_left_from_front` 的逆矩阵；右相机、中相机同理。
`*_camera_center_in_front_m` 是相机光心在前相机坐标系下的位置，不等同于正向外参的平移向量。
前相机自身的 `H_front_from_front` 为单位矩阵。

## 验证与诊断

以本次运行生成的 YAML 中的 `stereo_rms_px`、`num_final_pairs` 和对应 CSV 为准。CSV 记录可用图像对的 PnP 误差、双向交叉重投影误差、对称极线误差及筛选结果。左右相机相对于 front 的外参由两段变换组合，其精度受到两段标定误差的共同影响。

`test_calibrate_rig_manual.py` 覆盖非交换旋转下的变换组合与求逆、不同分辨率/内参的已知外参恢复及双向误差评估、YAML 输出的前相机基准与旧字段兼容性。

完成主标定后，可在项目根目录按需运行以下辅助工具：

| 脚本 | 用途 | 输出目录 |
| --- | --- | --- |
| `src/diagnose_pair_images.py` | 比较 PnP 求解、畸变及镜像内参，绘制残差 | `manual_rig_calibration/image_diagnostics/` |
| `src/diagnose_stereo_ablation.py` | 对比原畸变、中相机零畸变及双相机零畸变；先运行图像诊断工具创建输出目录 | `manual_rig_calibration/image_diagnostics/` |
| `src/calibrate_rig_mid_no_distortion.py` | 固定主标定保留的图像对，将中相机畸变设为零，重新标定并比较完整变换链 | `manual_rig_calibration_mid_no_distortion/` |

例如：

```powershell
.\.venv\Scripts\python.exe .\src\diagnose_pair_images.py
.\.venv\Scripts\python.exe .\src\diagnose_stereo_ablation.py
.\.venv\Scripts\python.exe .\src\calibrate_rig_mid_no_distortion.py
```

这些工具复用主标定模块，仅向诊断或对照目录写入结果。
