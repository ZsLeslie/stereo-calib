# 四相机 Open3D 预览

`preview_rig_open3d.py` 读取同目录的 `rig_extrinsics.yaml`，以前相机 `front` 为参考坐标系。

## 运行

本机已有 Open3D 的 `lidar` 环境，在 PowerShell 中运行：

```powershell
& 'D:\Software\miniconda\envs\lidar\python.exe' .\preview_rig_open3d.py
```

其他 Python 环境所需依赖：

```shell
python -m pip install numpy opencv-python open3d
python preview_rig_open3d.py
```

支持指定文件、调整坐标轴长度（米）和无窗口检查：

```shell
python preview_rig_open3d.py rig_extrinsics.yaml --axis-length 0.04
python preview_rig_open3d.py --check
```

## 查看方式

- 鼠标左键拖动旋转，Ctrl + 左键拖动平移，滚轮缩放。
- 按 `1` 恢复三维视角，`2` 切换 XZ 视角，`3` 切换 XY 视角。
- 按 `L` 显示/隐藏名称，`Q` 或 Esc 退出；按 `H` 在终端输出 Open3D 操作帮助。
- 光心球体、相机线框和长光轴箭头按相机着色：front 蓝灰、mid 橙黄、left 青色、right 紫色。
- 每个相机的局部坐标轴为 X 红、Y 绿、Z 蓝；相机名称标签随视角转动，保持朝向观察者。终端输出光心坐标和距前相机的距离，单位为毫米。
- 灰线连接光心；XZ 网格位于前相机 Y=0 平面，网格间隔 50 mm。

场景数据保持文件原有的米制坐标，不对 Y 轴或相机姿态做额外翻转。仅将观察视角的向上方向设置为 −Y，以符合常用 OpenCV 相机坐标约定。线框尺寸仅用于示意，不代表真实相机大小或标定视场角。

程序使用 `VisualizerWithKeyCallback` 的传统 OpenGL 窗口。本机 Open3D 0.20.0 的现代 GUI 后端在退出时出现异常，因此选用此实现；相关上游问题：[Open3D #7593](https://github.com/isl-org/Open3D/issues/7593)。

## 外参解释

文件定义 `X_camera = H_camera_from_front * X_front`。程序计算逆变换：

```text
R_front_from_camera = R_camera_from_front.T
C_front = -R_camera_from_front.T @ t_camera_from_front
```

使用该逆变换将每个相机的局部几何体放到前相机参考系。光轴方向是逆变换旋转部分的第三列。读取时检查旋转矩阵、齐次矩阵末行，并与文件内已有的逆变换和光心坐标核对。

GUI 自动检查并导出当前三维画面：

```shell
python preview_rig_open3d.py --smoke-test rig_visualization/open3d_preview.png
```
