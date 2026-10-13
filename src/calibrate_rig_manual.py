import cv2
import csv
import math
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# 1. 路径
# ============================================================

FRONT_DIR = Path("./images/front")
FRONT_MID_DIR = Path("./images/front-mid")

LEFT_DIR = Path("./images/left")
CENTER_LEFT_DIR = Path("./images/mid-left")

RIGHT_DIR = Path("./images/right")
CENTER_RIGHT_DIR = Path("./images/mid-right")


FRONT_INTRINSIC = Path("./front_camera_intrinsic.yaml")
LEFT_INTRINSIC = Path("./left_camera_intrinsic.yaml")
CENTER_INTRINSIC = Path("./mid_camera_intrinsic.yaml")
RIGHT_INTRINSIC = Path("./right_camera_intrinsic.yaml")


OUTPUT_DIR = Path("./manual_rig_calibration")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. ChArUco
# ============================================================

SQUARES_X = 7
SQUARES_Y = 5

SQUARE_LENGTH = 0.04665      # m
MARKER_LENGTH = 0.03265      # m

DICTIONARY_ID = cv2.aruco.DICT_5X5_100


# ============================================================
# 3. 图像处理
#
# 必须与内参标定时一致
# ============================================================

# 当前 front 图像无需镜像；须与 front_camera_intrinsic.yaml 的标定图像一致。
FLIP_FRONT = False
FLIP_LEFT = True
FLIP_CENTER = True
FLIP_RIGHT = True


# ============================================================
# 4. 数据阈值
# ============================================================

MIN_COMMON_CORNERS = 8

# 手动同步拍摄后，正常应该远小于这个值
MAX_PAIR_TIME_DIFF_MS = 100.0


# ============================================================
# 5. Robust参数
#
# 数据只有约15~17组，因此保持保守。
# ============================================================

ENABLE_ROBUST_FILTER = True

MAX_ROBUST_ITERATIONS = 3

# 每一轮最多删2组
MAX_REMOVE_PER_ITER = 2

# 最多删除原始数据25%
MAX_TOTAL_REJECT_FRACTION = 0.25

# 至少保留10组
MIN_FINAL_PAIRS = 10

MAD_K = 3.0

# robust阈值下限
MIN_CROSS_REPROJ_THRESHOLD_PX = 1.5
MIN_EPIPOLAR_THRESHOLD_PX = 0.8


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 6. 读取内参
# ============================================================

def load_intrinsics(path: Path):

    if not path.exists():
        raise RuntimeError(
            f"Intrinsic file does not exist: {path}"
        )

    fs = cv2.FileStorage(
        str(path),
        cv2.FILE_STORAGE_READ
    )

    if not fs.isOpened():
        raise RuntimeError(
            f"Cannot open intrinsic file: {path}"
        )

    K = fs.getNode(
        "camera_matrix"
    ).mat()

    D = fs.getNode(
        "distortion_coefficients"
    ).mat()

    width = int(
        fs.getNode(
            "image_width"
        ).real()
    )

    height = int(
        fs.getNode(
            "image_height"
        ).real()
    )

    fs.release()

    if K is None or D is None:
        raise RuntimeError(
            f"Invalid intrinsic file: {path}"
        )

    return (
        np.asarray(
            K,
            dtype=np.float64
        ),

        np.asarray(
            D,
            dtype=np.float64
        ),

        (width, height),
    )


K_FRONT, D_FRONT, SIZE_FRONT = load_intrinsics(FRONT_INTRINSIC)

K_LEFT, D_LEFT, SIZE_LEFT = load_intrinsics(
    LEFT_INTRINSIC
)

K_CENTER, D_CENTER, SIZE_CENTER = load_intrinsics(
    CENTER_INTRINSIC
)

K_RIGHT, D_RIGHT, SIZE_RIGHT = load_intrinsics(
    RIGHT_INTRINSIC
)


@dataclass(frozen=True)
class Camera:
    name: str
    K: np.ndarray
    D: np.ndarray
    size: tuple
    flip: bool


# center 是原有输出中 mid 的别名，保留以兼容 center_to_left/right.yaml。
CENTER_CAMERA = Camera("center", K_CENTER, D_CENTER, SIZE_CENTER, FLIP_CENTER)
FRONT_CAMERA = Camera("front", K_FRONT, D_FRONT, SIZE_FRONT, FLIP_FRONT)


# ============================================================
# 7. ChArUco Board
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    DICTIONARY_ID
)


board = cv2.aruco.CharucoBoard(
    (
        SQUARES_X,
        SQUARES_Y
    ),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)


detector_params = cv2.aruco.DetectorParameters()

detector_params.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)

detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 53
detector_params.adaptiveThreshWinSizeStep = 10

detector_params.minMarkerPerimeterRate = 0.003
detector_params.maxMarkerPerimeterRate = 4.0


detector = cv2.aruco.CharucoDetector(
    board,
    detectorParams=detector_params
)


BOARD_POINTS = np.asarray(
    board.getChessboardCorners(),
    dtype=np.float64
).reshape(
    -1,
    3
)


# ============================================================
# 8. 文件名时间戳
#
# left_1787562330659.jpg
#
# 13位时间戳 -> ms
#
# 如果以后变成16位us，也自动支持。
# ============================================================

def parse_timestamp_ms(
    path: Path
):

    numbers = re.findall(
        r"\d+",
        path.stem
    )

    if not numbers:
        raise ValueError(
            f"Cannot find timestamp: {path.name}"
        )

    # 使用文件名中最后一个数字串
    raw = int(
        numbers[-1]
    )

    digits = len(
        str(raw)
    )

    # 典型epoch微秒
    if digits >= 15:
        return raw / 1000.0

    # 典型epoch毫秒
    return float(raw)


# ============================================================
# 9. 获取目录中所有图片
# ============================================================

def list_images(
    directory: Path
):

    if not directory.exists():
        raise RuntimeError(
            f"Directory not found: {directory}"
        )

    paths = sorted(
        p
        for p in directory.iterdir()

        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_SUFFIXES
        )
    )

    return paths


# ============================================================
# 10. 根据时间戳一一匹配
#
# 不依赖文件排序。
#
# 对每张Side寻找尚未使用的最近Center图。
# ============================================================

def match_images_by_timestamp(
    side_dir,
    center_dir,
    side_name,
    reference_name="center"
):

    side_paths = list_images(
        side_dir
    )

    center_paths = list_images(
        center_dir
    )


    side_records = []

    for p in side_paths:

        side_records.append({
            "path":
                p,

            "time_ms":
                parse_timestamp_ms(
                    p
                )
        })


    center_records = []

    for p in center_paths:

        center_records.append({
            "path":
                p,

            "time_ms":
                parse_timestamp_ms(
                    p
                )
        })


    used_center = set()

    pairs = []


    for side in side_records:

        candidates = []

        for index, center in enumerate(
            center_records
        ):

            if index in used_center:
                continue

            dt_ms = (
                center[
                    "time_ms"
                ]
                -
                side[
                    "time_ms"
                ]
            )

            candidates.append(
                (
                    abs(dt_ms),
                    dt_ms,
                    index,
                    center
                )
            )


        if not candidates:
            continue


        candidates.sort(
            key=lambda x:
            x[0]
        )


        abs_dt, \
        dt_ms, \
        center_index, \
        center = candidates[0]


        if (
            abs_dt
            >
            MAX_PAIR_TIME_DIFF_MS
        ):

            print(
                f"[PAIR WARNING] "
                f"{side['path'].name}: "
                f"nearest {side_name}-{reference_name} "
                f"dt={dt_ms:.3f}ms > "
                f"{MAX_PAIR_TIME_DIFF_MS:.1f}ms"
            )

            continue


        used_center.add(
            center_index
        )


        pairs.append({

            "side_path":
                side[
                    "path"
                ],

            "center_path":
                center[
                    "path"
                ],

            "side_time_ms":
                side[
                    "time_ms"
                ],

            "center_time_ms":
                center[
                    "time_ms"
                ],

            # center - side
            "dt_ms":
                dt_ms
        })


    print()
    print("=" * 90)

    print(
        f"{side_name.upper()} <-> {reference_name.upper()} PAIRING"
    )

    print("=" * 90)


    print(
        "Side images   :",
        len(
            side_paths
        )
    )

    print(
        f"{reference_name} images :",
        len(
            center_paths
        )
    )

    print(
        "Matched pairs :",
        len(
            pairs
        )
    )


    for i, pair in enumerate(
        pairs,
        start=1
    ):

        print(

            f"{i:02d}  "

            f"{pair['side_path'].name}  <->  "
            f"{pair['center_path'].name}  "

            f"dt="
            f"{pair['dt_ms']:+.3f} ms"
        )


    return pairs


# ============================================================
# 11. ChArUco检测
# ============================================================

def detect_charuco(
    image,
    flip_horizontal
):

    if image is None:
        return {}


    if flip_horizontal:

        image = cv2.flip(
            image,
            1
        )


    corners, ids, _, _ = detector.detectBoard(
        image
    )


    point_map = {}


    if (
        corners is None
        or
        ids is None
    ):
        return point_map


    corners = corners.reshape(
        -1,
        2
    )

    ids = ids.reshape(-1)


    for cid, point in zip(
        ids,
        corners
    ):

        cid = int(
            cid
        )

        if (
            0
            <= cid
            <
            len(
                BOARD_POINTS
            )
        ):

            point_map[
                cid
            ] = np.asarray(
                point,
                dtype=np.float64
            )


    return point_map


# ============================================================
# 12. solvePnP
# ============================================================

def solve_pose(
    object_points,
    image_points,
    K,
    D
):

    obj = np.ascontiguousarray(
        object_points,
        dtype=np.float64
    )

    img = np.ascontiguousarray(
        image_points,
        dtype=np.float64
    )


    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )


    if not ok:
        return None


    try:

        rvec, tvec = cv2.solvePnPRefineLM(
            obj,
            img,
            K,
            D,
            rvec,
            tvec
        )

    except cv2.error:
        pass


    R, _ = cv2.Rodrigues(
        rvec
    )


    projected, _ = cv2.projectPoints(
        obj,
        rvec,
        tvec,
        K,
        D
    )


    projected = projected.reshape(
        -1,
        2
    )


    observed = img.reshape(
        -1,
        2
    )


    diff = (
        projected
        -
        observed
    )


    rmse = math.sqrt(
        float(
            np.mean(
                np.sum(
                    diff ** 2,
                    axis=1
                )
            )
        )
    )


    return {

        "R":
            R,

        "t":
            tvec.reshape(3),

        "rmse":
            float(
                rmse
            )
    }


# ============================================================
# 13. 两次PnP推导Center -> Side
# ============================================================

def relative_pose(
    center_pose,
    side_pose
):

    Rc = center_pose[
        "R"
    ]

    tc = center_pose[
        "t"
    ]


    Rs = side_pose[
        "R"
    ]

    ts = side_pose[
        "t"
    ]


    R_sc = (
        Rs
        @
        Rc.T
    )


    T_sc = (
        ts
        -
        R_sc
        @
        tc
    )


    return (
        R_sc,
        T_sc
    )


# ============================================================
# 14. 旋转角
# ============================================================

def rotation_angle_deg(
    R
):

    cos_angle = (
        np.trace(
            R
        )
        -
        1.0
    ) / 2.0


    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )


    return float(
        np.degrees(
            np.arccos(
                cos_angle
            )
        )
    )


# ============================================================
# 15. 构建Stereo数据
# ============================================================

def build_dataset(
    side_name,
    pairs,
    K_side,
    D_side,
    flip_side,
    reference=CENTER_CAMERA,
    side_size=SIZE_CENTER
):

    dataset = []


    print()
    print("=" * 90)

    print(
        f"BUILD {reference.name.upper()} -> "
        f"{side_name.upper()} DATASET"
    )

    print("=" * 90)


    for pair_index, pair in enumerate(
        pairs,
        start=1
    ):


        center_image = cv2.imread(
            str(
                pair[
                    "center_path"
                ]
            )
        )


        side_image = cv2.imread(
            str(
                pair[
                    "side_path"
                ]
            )
        )


        if (
            center_image is None
            or
            side_image is None
        ):

            print(
                f"[READ FAIL] "
                f"pair={pair_index:02d}"
            )

            continue


        center_size = (
            center_image.shape[1],
            center_image.shape[0]
        )

        actual_side_size = (
            side_image.shape[1],
            side_image.shape[0]
        )


        if (
            center_size
            !=
            reference.size
            or
            actual_side_size
            !=
            side_size
        ):

            print(
                f"[SIZE FAIL] "
                f"pair={pair_index:02d} "
                f"center={center_size} "
                f"side={actual_side_size}; expected={reference.size}/{side_size}"
            )

            continue


        center_map = detect_charuco(
            center_image,
            reference.flip
        )


        side_map = detect_charuco(
            side_image,
            flip_side
        )


        common_ids = sorted(

            set(
                center_map.keys()
            )

            &

            set(
                side_map.keys()
            )
        )


        if (
            len(
                common_ids
            )
            <
            MIN_COMMON_CORNERS
        ):

            print(

                f"[SKIP] pair={pair_index:02d} "
                f"common="
                f"{len(common_ids)}"
            )

            continue


        object_points = np.asarray(
            [
                BOARD_POINTS[cid]
                for cid
                in common_ids
            ],
            dtype=np.float64
        )


        center_points = np.asarray(
            [
                center_map[cid]
                for cid
                in common_ids
            ],
            dtype=np.float64
        )


        side_points = np.asarray(
            [
                side_map[cid]
                for cid
                in common_ids
            ],
            dtype=np.float64
        )


        center_pose = solve_pose(
            object_points,
            center_points,
            reference.K,
            reference.D
        )


        side_pose = solve_pose(
            object_points,
            side_points,
            K_side,
            D_side
        )


        if (
            center_pose is None
            or
            side_pose is None
        ):

            print(
                f"[PnP FAIL] "
                f"pair={pair_index:02d}"
            )

            continue


        R_pair, T_pair = relative_pose(
            center_pose,
            side_pose
        )


        baseline_mm = (
            np.linalg.norm(
                T_pair
            )
            *
            1000.0
        )


        rot_deg = rotation_angle_deg(
            R_pair
        )


        dataset.append({

            "pair_index":
                pair_index,

            "center_filename":
                pair[
                    "center_path"
                ].name,

            "side_filename":
                pair[
                    "side_path"
                ].name,

            "dt_ms":
                pair[
                    "dt_ms"
                ],

            "common_count":
                len(
                    common_ids
                ),

            "object_points":
                object_points,

            "center_points":
                center_points,

            "side_points":
                side_points,

            "center_pnp_rmse":
                center_pose[
                    "rmse"
                ],

            "side_pnp_rmse":
                side_pose[
                    "rmse"
                ],

            "pair_baseline_mm":
                float(
                    baseline_mm
                ),

            "pair_rotation_deg":
                float(
                    rot_deg
                ),

            "rejected_iteration":
                0
        })


        print(

            f"pair={pair_index:02d} "

            f"dt="
            f"{pair['dt_ms']:+7.3f}ms "

            f"common="
            f"{len(common_ids):2d} "

            f"PnP="
            f"{center_pose['rmse']:.3f}/"
            f"{side_pose['rmse']:.3f}px "

            f"baseline="
            f"{baseline_mm:7.2f}mm "

            f"rot="
            f"{rot_deg:7.3f}deg"
        )


    print()

    print(
        "Usable pairs:",
        len(
            dataset
        )
    )


    return dataset



# ============================================================
# 16. stereoCalibrate
#
# Camera1 = reference（front-mid 使用 Front，其余使用 Center/Mid）
# Camera2 = Side
#
# 得到：
#
# X_side = R * X_reference + T
# 固定各自内参时 imageSize 不用于初始化 K；允许两台相机使用不同分辨率。
# ============================================================

def stereo_calibrate(
    dataset,
    indices,
    K_side,
    D_side,
    reference=CENTER_CAMERA
):

    object_points = []
    center_points = []
    side_points = []


    for idx in indices:

        item = dataset[idx]


        # OpenCV 4.10 stereoCalibrate
        # objectPoints需要float32

        obj = np.ascontiguousarray(
            np.asarray(
                item[
                    "object_points"
                ],
                dtype=np.float32
            ).reshape(
                -1,
                1,
                3
            )
        )


        cp = np.ascontiguousarray(
            np.asarray(
                item[
                    "center_points"
                ],
                dtype=np.float32
            ).reshape(
                -1,
                1,
                2
            )
        )


        sp = np.ascontiguousarray(
            np.asarray(
                item[
                    "side_points"
                ],
                dtype=np.float32
            ).reshape(
                -1,
                1,
                2
            )
        )


        object_points.append(
            obj
        )

        center_points.append(
            cp
        )

        side_points.append(
            sp
        )


    criteria = (

        cv2.TERM_CRITERIA_EPS
        +
        cv2.TERM_CRITERIA_MAX_ITER,

        300,

        1e-10
    )


    (
        rms,

        _,
        _,

        _,
        _,

        R,
        T,

        E,
        F

    ) = cv2.stereoCalibrate(

        object_points,

        center_points,
        side_points,

        reference.K.copy(),
        reference.D.copy(),

        K_side.copy(),
        D_side.copy(),

        reference.size,

        criteria=criteria,

        flags=cv2.CALIB_FIX_INTRINSIC
    )


    return {

        "rms":
            float(
                rms
            ),

        "R":
            np.asarray(
                R,
                dtype=np.float64
            ),

        "T":
            np.asarray(
                T,
                dtype=np.float64
            ).reshape(3),

        "E":
            np.asarray(
                E,
                dtype=np.float64
            ),

        "F":
            np.asarray(
                F,
                dtype=np.float64
            )
    }



# ============================================================
# 17. 点RMSE
# ============================================================

def point_rmse(
    predicted,
    observed
):

    predicted = predicted.reshape(
        -1,
        2
    )

    observed = observed.reshape(
        -1,
        2
    )


    diff = (
        predicted
        -
        observed
    )


    return math.sqrt(
        float(
            np.mean(
                np.sum(
                    diff ** 2,
                    axis=1
                )
            )
        )
    )


# ============================================================
# 18. 固定R/T后的双向Cross Reprojection
# ============================================================

def cross_reprojection_error(
    item,
    result,
    K_side,
    D_side,
    reference=CENTER_CAMERA
):

    obj = item[
        "object_points"
    ]

    center_pts = item[
        "center_points"
    ]

    side_pts = item[
        "side_points"
    ]


    center_pose = solve_pose(
        obj,
        center_pts,
        reference.K,
        reference.D
    )


    side_pose = solve_pose(
        obj,
        side_pts,
        K_side,
        D_side
    )


    if (
        center_pose is None
        or
        side_pose is None
    ):
        return float("inf")


    R_sc = result[
        "R"
    ]

    T_sc = result[
        "T"
    ]


    # ========================================================
    # Center board pose -> predict Side
    # ========================================================

    R_cb = center_pose[
        "R"
    ]

    t_cb = center_pose[
        "t"
    ]


    R_sb_pred = (
        R_sc
        @
        R_cb
    )


    t_sb_pred = (
        R_sc
        @
        t_cb
        +
        T_sc
    )


    rvec_side, _ = cv2.Rodrigues(
        R_sb_pred
    )


    predicted_side, _ = cv2.projectPoints(
        obj,
        rvec_side,
        t_sb_pred,
        K_side,
        D_side
    )


    side_error = point_rmse(
        predicted_side,
        side_pts
    )


    # ========================================================
    # Side board pose -> predict Center
    # ========================================================

    R_sb = side_pose[
        "R"
    ]

    t_sb = side_pose[
        "t"
    ]


    R_cs = R_sc.T


    T_cs = (
        -
        R_sc.T
        @
        T_sc
    )


    R_cb_pred = (
        R_cs
        @
        R_sb
    )


    t_cb_pred = (
        R_cs
        @
        t_sb
        +
        T_cs
    )


    rvec_center, _ = cv2.Rodrigues(
        R_cb_pred
    )


    predicted_center, _ = cv2.projectPoints(
        obj,
        rvec_center,
        t_cb_pred,
        reference.K,
        reference.D
    )


    center_error = point_rmse(
        predicted_center,
        center_pts
    )


    combined = math.sqrt(

        (
            side_error ** 2
            +
            center_error ** 2
        )

        /
        2.0
    )


    return float(
        combined
    )



# ============================================================
# 19. 根据R/T重新构造Fundamental Matrix
#
# F = K2^-T [T]x R K1^-1
# ============================================================

def fundamental_from_rt(
    R,
    T,
    K1,
    K2
):

    tx, ty, tz = T


    T_cross = np.array(

        [
            [0.0, -tz, ty],
            [tz, 0.0, -tx],
            [-ty, tx, 0.0]
        ],

        dtype=np.float64
    )


    E = (
        T_cross
        @
        R
    )


    F = (

        np.linalg.inv(
            K2
        ).T

        @
        E

        @
        np.linalg.inv(
            K1
        )
    )


    return F


# ============================================================
# 20. 对称极线误差
# ============================================================

def symmetric_epipolar_error(
    item,
    result,
    K_side,
    D_side,
    reference=CENTER_CAMERA
):

    c = item[
        "center_points"
    ].reshape(
        -1,
        1,
        2
    )


    s = item[
        "side_points"
    ].reshape(
        -1,
        1,
        2
    )


    # 去畸变，但仍保持像素坐标
    c_ud = cv2.undistortPoints(
        c,
        reference.K,
        reference.D,
        P=reference.K
    ).reshape(
        -1,
        2
    )


    s_ud = cv2.undistortPoints(
        s,
        K_side,
        D_side,
        P=K_side
    ).reshape(
        -1,
        2
    )


    n = len(
        c_ud
    )


    x1 = np.hstack(
        (
            c_ud,
            np.ones(
                (
                    n,
                    1
                ),
                dtype=np.float64
            )
        )
    )


    x2 = np.hstack(
        (
            s_ud,
            np.ones(
                (
                    n,
                    1
                ),
                dtype=np.float64
            )
        )
    )


    F = fundamental_from_rt(

        result[
            "R"
        ],

        result[
            "T"
        ],

        reference.K,

        K_side
    )


    # Center -> Side line
    l2 = (
        F
        @
        x1.T
    ).T


    # Side -> Center line
    l1 = (
        F.T
        @
        x2.T
    ).T


    algebraic = np.abs(
        np.sum(
            x2
            *
            l2,
            axis=1
        )
    )


    d1 = (

        algebraic

        /

        np.maximum(
            np.sqrt(
                l1[:, 0] ** 2
                +
                l1[:, 1] ** 2
            ),
            1e-12
        )
    )


    d2 = (

        algebraic

        /

        np.maximum(
            np.sqrt(
                l2[:, 0] ** 2
                +
                l2[:, 1] ** 2
            ),
            1e-12
        )
    )


    return math.sqrt(

        float(

            np.mean(

                (
                    d1 ** 2
                    +
                    d2 ** 2
                )

                /
                2.0
            )
        )
    )



# ============================================================
# 21. MAD robust threshold
# ============================================================

def robust_threshold(
    values,
    minimum
):

    v = np.asarray(
        values,
        dtype=np.float64
    )


    median = float(
        np.median(
            v
        )
    )


    mad = float(
        np.median(
            np.abs(
                v
                -
                median
            )
        )
    )


    sigma = (
        1.4826
        *
        mad
    )


    threshold = (

        median
        +
        MAD_K
        *
        sigma
    )


    return (
        median,
        sigma,
        max(
            threshold,
            minimum
        )
    )


# ============================================================
# 22. Robust Stereo
# ============================================================

def robust_stereo_calibrate(
    side_name,
    dataset,
    K_side,
    D_side,
    reference=CENTER_CAMERA
):

    if (
        len(
            dataset
        )
        <
        MIN_FINAL_PAIRS
    ):

        raise RuntimeError(
            f"{side_name}: too few usable pairs."
        )


    active = list(
        range(
            len(
                dataset
            )
        )
    )


    initial_count = len(
        active
    )


    max_total_remove = int(
        math.floor(
            initial_count
            *
            MAX_TOTAL_REJECT_FRACTION
        )
    )


    total_removed = 0


    # ========================================================
    # 初始结果
    # ========================================================

    initial_result = stereo_calibrate(
        dataset,
        active,
        K_side,
        D_side,
        reference=reference
    )


    print()
    print("=" * 100)

    print(
        f"INITIAL {reference.name.upper()} -> "
        f"{side_name.upper()}"
    )

    print("=" * 100)


    print(
        "Pairs:",
        len(
            active
        )
    )


    print(
        "Initial Stereo RMS:",
        f"{initial_result['rms']:.6f} px"
    )


    if not ENABLE_ROBUST_FILTER:

        return (
            initial_result,
            set(
                active
            )
        )


    # ========================================================
    # Robust iterations
    # ========================================================

    for iteration in range(
        1,
        MAX_ROBUST_ITERATIONS + 1
    ):


        result = stereo_calibrate(
            dataset,
            active,
            K_side,
            D_side,
            reference=reference
        )


        metrics = []


        for idx in active:


            reproj = cross_reprojection_error(
                dataset[idx],
                result,
                K_side,
                D_side,
                reference=reference
            )


            epi = symmetric_epipolar_error(
                dataset[idx],
                result,
                K_side,
                D_side,
                reference=reference
            )


            metrics.append({
                "index":
                    idx,

                "reproj":
                    reproj,

                "epi":
                    epi
            })


        reproj_values = [
            m[
                "reproj"
            ]
            for m
            in metrics
        ]


        epi_values = [
            m[
                "epi"
            ]
            for m
            in metrics
        ]


        (
            reproj_med,
            reproj_sigma,
            reproj_threshold

        ) = robust_threshold(

            reproj_values,
            MIN_CROSS_REPROJ_THRESHOLD_PX
        )


        (
            epi_med,
            epi_sigma,
            epi_threshold

        ) = robust_threshold(

            epi_values,
            MIN_EPIPOLAR_THRESHOLD_PX
        )


        print()
        print("-" * 100)

        print(
            f"Iteration {iteration}"
        )

        print("-" * 100)


        print(
            "Pairs:",
            len(
                active
            )
        )


        print(
            "Stereo RMS:",
            f"{result['rms']:.6f} px"
        )


        print(
            "Cross reprojection:"
        )

        print(
            f"  median    = "
            f"{reproj_med:.3f} px"
        )

        print(
            f"  sigma     = "
            f"{reproj_sigma:.3f} px"
        )

        print(
            f"  threshold = "
            f"{reproj_threshold:.3f} px"
        )


        print(
            "Epipolar:"
        )

        print(
            f"  median    = "
            f"{epi_med:.3f} px"
        )

        print(
            f"  sigma     = "
            f"{epi_sigma:.3f} px"
        )

        print(
            f"  threshold = "
            f"{epi_threshold:.3f} px"
        )


        # ====================================================
        # 异常候选
        # ====================================================

        outliers = []


        for m in metrics:


            reproj_ratio = (
                m[
                    "reproj"
                ]
                /
                reproj_threshold
            )


            epi_ratio = (
                m[
                    "epi"
                ]
                /
                epi_threshold
            )


            score = max(
                reproj_ratio,
                epi_ratio
            )


            if score > 1.0:

                outliers.append(
                    (
                        score,
                        m
                    )
                )


        if not outliers:

            print()

            print(
                "No obvious outlier."
            )

            break


        outliers.sort(
            key=lambda x:
            x[0],
            reverse=True
        )


        remaining_total = (
            max_total_remove
            -
            total_removed
        )


        remaining_pair_budget = (
            len(active)
            -
            MIN_FINAL_PAIRS
        )


        remove_count = min(

            len(
                outliers
            ),

            MAX_REMOVE_PER_ITER,

            remaining_total,

            remaining_pair_budget
        )


        if remove_count <= 0:

            print()

            print(
                "Stop: rejection budget reached."
            )

            break


        # ====================================================
        # 如果超过25%的pair同时被判异常，
        # 不要激进过滤，说明可能整体数据有问题。
        # ====================================================

        if (
            len(
                outliers
            )
            >
            len(
                active
            )
            *
            0.40
        ):

            print()

            print(
                "[WARNING] Too many pairs are classified "
                "as outliers."
            )

            print(
                "Stop robust filtering instead of "
                "destroying the dataset."
            )

            break


        remove_indices = set()


        print()

        print(
            f"Reject {remove_count} pair(s):"
        )


        for score, m in outliers[
            :remove_count
        ]:


            idx = m[
                "index"
            ]


            item = dataset[idx]


            remove_indices.add(
                idx
            )


            item[
                "rejected_iteration"
            ] = iteration


            print(

                f"  pair="
                f"{item['pair_index']:02d} "

                f"dt="
                f"{item['dt_ms']:+.3f}ms "

                f"reproj="
                f"{m['reproj']:.3f}px "

                f"epi="
                f"{m['epi']:.3f}px"
            )


        active = [
            idx
            for idx
            in active
            if idx
            not in remove_indices
        ]


        total_removed += (
            remove_count
        )


    # ========================================================
    # Final result
    # ========================================================

    final_result = stereo_calibrate(
        dataset,
        active,
        K_side,
        D_side,
        reference=reference
    )


    return (
        final_result,
        set(
            active
        )
    )



# ============================================================
# 23. 计算最终所有pair指标
# ============================================================

def calculate_final_metrics(
    dataset,
    result,
    K_side,
    D_side,
    reference=CENTER_CAMERA
):

    for item in dataset:

        item[
            "final_cross_reproj_px"
        ] = cross_reprojection_error(
            item,
            result,
            K_side,
            D_side,
            reference=reference
        )


        item[
            "final_epipolar_px"
        ] = symmetric_epipolar_error(
            item,
            result,
            K_side,
            D_side,
            reference=reference
        )



# ============================================================
# 24. H矩阵
# ============================================================

def make_transform(
    R,
    T
):

    H = np.eye(
        4,
        dtype=np.float64
    )


    H[
        :3,
        :3
    ] = R


    H[
        :3,
        3
    ] = T


    return H


# ============================================================
# 25. 输出结果
# ============================================================

def save_result(
    side_name,
    dataset,
    result,
    active_set,
    K_side,
    D_side,
    reference=CENTER_CAMERA
):

    calculate_final_metrics(
        dataset,
        result,
        K_side,
        D_side,
        reference=reference
    )


    R = result[
        "R"
    ]

    T = result[
        "T"
    ]


    H = make_transform(
        R,
        T
    )


    baseline_m = float(
        np.linalg.norm(
            T
        )
    )


    camera_center = (
        -
        R.T
        @
        T
    )


    # ========================================================
    # YAML
    # ========================================================

    yaml_path = (

        OUTPUT_DIR
        /
        f"{reference.name}_to_{side_name}.yaml"
    )


    fs = cv2.FileStorage(
        str(yaml_path),
        cv2.FILE_STORAGE_WRITE
    )


    fs.write(

        "transform_definition",

        f"X_{side_name} = "
        f"R_{reference.name}_to_{side_name} * "
        f"X_{reference.name} + "
        f"T_{reference.name}_to_{side_name}_m"
    )


    fs.write(
        f"R_{reference.name}_to_{side_name}",
        R
    )


    fs.write(
        f"T_{reference.name}_to_{side_name}_m",
        T.reshape(
            3,
            1
        )
    )


    fs.write(
        f"H_{side_name}_from_{reference.name}",
        H
    )


    fs.write(
        "baseline_m",
        baseline_m
    )


    fs.write(
        "stereo_rms_px",
        result[
            "rms"
        ]
    )


    fs.write(
        "num_final_pairs",
        len(
            active_set
        )
    )


    fs.write(
        f"{side_name}_camera_center_in_{reference.name}_m",
        camera_center.reshape(
            3,
            1
        )
    )


    fs.write("reference_camera", reference.name)
    fs.write("target_camera", side_name)
    fs.write("translation_unit", "m")
    fs.write(f"H_{reference.name}_from_{side_name}", np.linalg.inv(H))
    fs.release()


    # ========================================================
    # CSV
    # ========================================================

    csv_path = (

        OUTPUT_DIR
        /
        f"{reference.name}_{side_name}_pair_metrics.csv"
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:


        writer = csv.writer(
            f
        )


        writer.writerow([

            "pair_index",

            "final_inlier",

            "rejected_iteration",

            f"{reference.name}_filename",

            f"{side_name}_filename",

            "dt_ms",

            "common_count",

            f"{reference.name}_pnp_rmse_px",

            f"{side_name}_pnp_rmse_px",

            "independent_baseline_mm",

            "independent_rotation_deg",

            "final_cross_reprojection_px",

            "final_epipolar_px"
        ])


        for idx, item in enumerate(
            dataset
        ):


            writer.writerow([

                item[
                    "pair_index"
                ],

                int(
                    idx
                    in
                    active_set
                ),

                item[
                    "rejected_iteration"
                ],

                item[
                    "center_filename"
                ],

                item[
                    "side_filename"
                ],

                item[
                    "dt_ms"
                ],

                item[
                    "common_count"
                ],

                item[
                    "center_pnp_rmse"
                ],

                item[
                    "side_pnp_rmse"
                ],

                item[
                    "pair_baseline_mm"
                ],

                item[
                    "pair_rotation_deg"
                ],

                item[
                    "final_cross_reproj_px"
                ],

                item[
                    "final_epipolar_px"
                ]
            ])


    return (
        yaml_path,
        csv_path,
        H
    )



# ============================================================
# 26. 打印最终外参
# ============================================================

def print_final_result(
    side_name,
    result,
    final_count,
    reference=CENTER_CAMERA
):

    R = result[
        "R"
    ]

    T = result[
        "T"
    ]


    baseline_mm = (
        np.linalg.norm(
            T
        )
        *
        1000.0
    )


    angle = rotation_angle_deg(
        R
    )


    camera_center = (
        -
        R.T
        @
        T
    )


    print()
    print("=" * 100)

    print(
        f"FINAL {reference.name.upper()} -> "
        f"{side_name.upper()}"
    )

    print("=" * 100)


    print(
        "Stereo RMS:",
        f"{result['rms']:.6f} px"
    )


    print(
        "Final pairs:",
        final_count
    )


    print()

    print(
        "R:"
    )

    print(
        R
    )


    print()

    print(
        "T [m]:"
    )

    print(
        T
    )


    print()

    print(
        "Baseline:",
        f"{baseline_mm:.3f} mm"
    )


    print(
        "Rotation angle:",
        f"{angle:.3f} deg"
    )


    print()

    print(
        f"{side_name.capitalize()} camera center "
        f"in {reference.name} coordinates [m]:"
    )

    print(
        camera_center
    )



# ============================================================
# 27. Main
# ============================================================

def compose_front_transforms(H_mid_from_front, H_left_from_mid, H_right_from_mid):
    """H_A_from_B 将 B 坐标转换为 A 坐标；平移单位为米。"""
    return {
        "front": np.eye(4, dtype=np.float64),
        "mid": H_mid_from_front,
        "left": H_left_from_mid @ H_mid_from_front,
        "right": H_right_from_mid @ H_mid_from_front,
    }


def write_front_transform(fs, camera_name, H_camera_from_front):
    """同时保存投影外参（front -> camera）及相机在 front 下的位姿。"""
    H_front_from_camera = np.linalg.inv(H_camera_from_front)
    fs.write(f"H_{camera_name}_from_front", H_camera_from_front)
    if camera_name != "front":
        fs.write(f"H_front_from_{camera_name}", H_front_from_camera)
    fs.write(f"R_front_to_{camera_name}", H_camera_from_front[:3, :3])
    fs.write(f"T_front_to_{camera_name}_m", H_camera_from_front[:3, 3:4])
    fs.write(f"{camera_name}_camera_center_in_front_m", H_front_from_camera[:3, 3:4])


def save_front_rig(H_mid_from_front, H_left_from_mid, H_right_from_mid):
    transforms = compose_front_transforms(H_mid_from_front, H_left_from_mid, H_right_from_mid)
    rig_path = OUTPUT_DIR / "rig_extrinsics.yaml"
    fs = cv2.FileStorage(str(rig_path), cv2.FILE_STORAGE_WRITE)
    fs.write("reference_camera", "front")
    fs.write("center_camera_alias", "mid")
    fs.write("translation_unit", "m")
    fs.write("transform_definition", "X_A = H_A_from_B * X_B (homogeneous coordinates)")
    fs.write("composition_definition", "H_side_from_front = H_side_from_mid * H_mid_from_front")
    for name, H in transforms.items():
        write_front_transform(fs, name, H)
    # 保留原有中相机基准和左右相机之间的变换。
    fs.write("H_left_from_mid", H_left_from_mid)
    fs.write("H_right_from_mid", H_right_from_mid)
    fs.write("H_left_from_center", H_left_from_mid)
    fs.write("H_right_from_center", H_right_from_mid)
    fs.write("H_right_from_left", H_right_from_mid @ np.linalg.inv(H_left_from_mid))
    fs.release()

    for name in ("left", "right"):
        path = OUTPUT_DIR / f"front_to_{name}.yaml"
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        fs.write("reference_camera", "front")
        fs.write("target_camera", name)
        fs.write("translation_unit", "m")
        fs.write("calibration_method", "composed_via_mid")
        fs.write("transform_definition", f"X_{name} = R_front_to_{name} * X_front + T_front_to_{name}_m")
        write_front_transform(fs, name, transforms[name])
        fs.write("baseline_m", float(np.linalg.norm(transforms[name][:3, 3])))
        fs.release()
        print(f"Front -> {name} (via mid): {path}")
        print(f"{name} camera center in front [m]: {np.linalg.inv(transforms[name])[:3, 3]}")
    return rig_path


def main():
    print("=" * 80)
    print("FOUR-CAMERA CALIBRATION: reference = front, bridge = mid")
    print("Front:", SIZE_FRONT, "Mid:", SIZE_CENTER, "Left:", SIZE_LEFT, "Right:", SIZE_RIGHT)
    print("Square length:", SQUARE_LENGTH * 1000, "mm")
    print("Marker length:", MARKER_LENGTH * 1000, "mm")

    # Camera1 为 reference，Camera2 为 target。front-mid 目录存放中相机图像。
    jobs = [
        (FRONT_CAMERA, "mid", FRONT_MID_DIR, FRONT_DIR, K_CENTER, D_CENTER, SIZE_CENTER, FLIP_CENTER),
        (CENTER_CAMERA, "left", LEFT_DIR, CENTER_LEFT_DIR, K_LEFT, D_LEFT, SIZE_LEFT, FLIP_LEFT),
        (CENTER_CAMERA, "right", RIGHT_DIR, CENTER_RIGHT_DIR, K_RIGHT, D_RIGHT, SIZE_RIGHT, FLIP_RIGHT),
    ]
    calibrated = {}
    for reference, name, target_dir, reference_dir, K, D, size, flip in jobs:
        pairs = match_images_by_timestamp(target_dir, reference_dir, name, reference.name)
        dataset = build_dataset(name, pairs, K, D, flip, reference=reference, side_size=size)
        result, active = robust_stereo_calibrate(name, dataset, K, D, reference=reference)
        print_final_result(name, result, len(active), reference=reference)
        yaml_path, csv_path, H = save_result(name, dataset, result, active, K, D, reference=reference)
        calibrated[name] = H
        print("Extrinsics:", yaml_path)
        print("Metrics:", csv_path)

    rig_path = save_front_rig(calibrated["mid"], calibrated["left"], calibrated["right"])
    print("Rig extrinsics:", rig_path)
    print("Finished.")


if __name__ == "__main__":
    main()
