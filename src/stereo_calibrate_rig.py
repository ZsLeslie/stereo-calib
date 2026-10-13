import cv2
import csv
import math
import numpy as np
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

LEFT_DIR = Path("./frames/left")
RIGHT_DIR = Path("./frames/right")

MATCHED_CENTER_LEFT_DIR = Path(
    "./matched/center_for_left"
)

MATCHED_CENTER_RIGHT_DIR = Path(
    "./matched/center_for_right"
)

LEFT_PAIR_CSV = Path(
    "./matched/left_center_pairs.csv"
)

RIGHT_PAIR_CSV = Path(
    "./matched/right_center_pairs.csv"
)


LEFT_INTRINSIC = Path(
    "./left_camera_intrinsic.yaml"
)

CENTER_INTRINSIC = Path(
    "./mid_camera_intrinsic.yaml"
)

RIGHT_INTRINSIC = Path(
    "./right_camera_intrinsic.yaml"
)


OUTPUT_DIR = Path(
    "./matched_stereo_calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. ChArUco 参数
# ============================================================

SQUARES_X = 7
SQUARES_Y = 5

SQUARE_LENGTH = 0.04665
MARKER_LENGTH = 0.03265

DICTIONARY_ID = cv2.aruco.DICT_5X5_100


# ============================================================
# 3. 图像水平翻转
#
# 必须与三台相机内参标定时保持一致
# ============================================================

FLIP_LEFT = True
FLIP_CENTER = True
FLIP_RIGHT = True


# ============================================================
# 4. 最少共同 ChArUco 角点
# ============================================================

MIN_COMMON_CORNERS = 8


# ============================================================
# 5. 是否跳过 CSV 中 status=REJECT 的行
# ============================================================

SKIP_REJECT_ROWS = True


# ============================================================
# 6. 读取内参
# ============================================================

def load_intrinsics(path: Path):

    if not path.exists():

        raise RuntimeError(
            f"Intrinsic file not found: {path}"
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


    if K is None:

        raise RuntimeError(
            f"camera_matrix missing: {path}"
        )


    if D is None:

        raise RuntimeError(
            f"distortion_coefficients missing: {path}"
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

        (
            width,
            height
        )
    )


# ============================================================
# 7. 加载三台相机内参
# ============================================================

K_LEFT, D_LEFT, SIZE_LEFT = load_intrinsics(
    LEFT_INTRINSIC
)

K_CENTER, D_CENTER, SIZE_CENTER = load_intrinsics(
    CENTER_INTRINSIC
)

K_RIGHT, D_RIGHT, SIZE_RIGHT = load_intrinsics(
    RIGHT_INTRINSIC
)


if not (
    SIZE_LEFT
    ==
    SIZE_CENTER
    ==
    SIZE_RIGHT
):

    raise RuntimeError(
        "Camera image sizes are inconsistent."
    )


IMAGE_SIZE = SIZE_CENTER


print("=" * 80)
print("INTRINSICS")
print("=" * 80)

print(
    "Left   :",
    SIZE_LEFT
)

print(
    "Center :",
    SIZE_CENTER
)

print(
    "Right  :",
    SIZE_RIGHT
)

print()

print(
    "K_CENTER:"
)

print(
    K_CENTER
)

print()

print(
    "D_CENTER:"
)

print(
    D_CENTER.reshape(-1)
)


# ============================================================
# 8. 创建 ChArUco Board
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


print()

print(
    "Board corners:",
    len(BOARD_POINTS)
)

print(
    "Square length:",
    SQUARE_LENGTH * 1000.0,
    "mm"
)

print(
    "Marker length:",
    MARKER_LENGTH * 1000.0,
    "mm"
)


# ============================================================
# 9. ChArUco 检测
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


    corners, ids, \
    marker_corners, marker_ids = detector.detectBoard(
        image
    )


    result = {}


    if (
        corners is None
        or
        ids is None
    ):

        return result


    corners = corners.reshape(
        -1,
        2
    )


    ids = ids.reshape(
        -1
    )


    for cid, point in zip(
        ids,
        corners
    ):

        cid = int(
            cid
        )


        if (
            0
            <=
            cid
            <
            len(
                BOARD_POINTS
            )
        ):

            result[
                cid
            ] = np.asarray(
                point,
                dtype=np.float64
            )


    return result


# ============================================================
# 10. 辅助：CSV 中获取第一个存在的字段
# ============================================================

def get_first(
    row,
    candidates
):

    for key in candidates:

        value = row.get(
            key,
            ""
        )


        if (
            value is not None
            and
            str(value).strip()
            !=
            ""
        ):

            return str(
                value
            ).strip()


    return None


# ============================================================
# 11. 解析 Side 图像路径
# ============================================================

def resolve_side_image_path(
    base_dir,
    filename
):

    filename = Path(
        filename
    ).name


    candidate = (
        base_dir
        /
        filename
    )


    if candidate.exists():

        return candidate


    # --------------------------------------------------------
    # 兜底：
    # 根据文件名后缀搜索
    # --------------------------------------------------------

    matches = list(
        base_dir.glob(
            f"*{filename}"
        )
    )


    if len(matches) == 1:

        return matches[0]


    return candidate


# ============================================================
# 12. 解析 matched Center 图像
#
# CSV：
#
# center_000000017693244.png
#
# 实际目录：
#
# pair_0001_center_000000017693244.png
# ============================================================

def resolve_matched_center_path(
    base_dir,
    center_filename,
    pair_index
):

    center_filename = Path(
        center_filename
    ).name


    # ========================================================
    # 情况1：
    #
    # 目录里就是原始文件名
    # ========================================================

    candidate = (
        base_dir
        /
        center_filename
    )


    if candidate.exists():

        return candidate


    # ========================================================
    # 情况2：
    #
    # pair_0001_center_xxx.png
    # ========================================================

    candidate = (

        base_dir
        /
        f"pair_{pair_index:04d}_{center_filename}"
    )


    if candidate.exists():

        return candidate


    # ========================================================
    # 情况3：
    #
    # 根据Center原始文件名后缀匹配
    # ========================================================

    matches = list(

        base_dir.glob(
            f"pair_*_{center_filename}"
        )
    )


    if len(matches) == 1:

        return matches[0]


    if len(matches) > 1:

        print()

        print(
            "[WARNING] Multiple Center matches:"
        )


        for path in matches:

            print(
                "   ",
                path
            )


        return matches[0]


    # ========================================================
    # 情况4：
    #
    # 根据timestamp部分搜索
    # ========================================================

    stem = Path(
        center_filename
    ).stem


    if stem.startswith(
        "center_"
    ):

        timestamp = stem[
            len(
                "center_"
            ):
        ]


        matches = list(

            base_dir.glob(
                f"*{timestamp}.*"
            )
        )


        if len(matches) == 1:

            return matches[0]


    # 返回预计路径，后面会触发READ FAIL
    return candidate


# ============================================================
# 13. 读取 matched CSV
# ============================================================

def load_pair_rows(
    csv_path,
    side_name
):

    if not csv_path.exists():

        raise RuntimeError(
            f"Pair CSV not found: {csv_path}"
        )


    rows = []


    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:


        reader = csv.DictReader(
            f
        )


        print()

        print(
            f"{csv_path.name} columns:"
        )

        print(
            reader.fieldnames
        )


        for row_number, row in enumerate(
            reader,
            start=1
        ):


            status = str(
                row.get(
                    "status",
                    ""
                )
            ).strip().upper()


            if (
                SKIP_REJECT_ROWS
                and
                status
                ==
                "REJECT"
            ):

                continue


            side_filename = get_first(
                row,
                [
                    f"{side_name}_filename",
                    f"{side_name}_file",
                    "side_filename",
                    side_name
                ]
            )


            center_filename = get_first(
                row,
                [
                    "center_filename",
                    "mid_filename",
                    "center_file",
                    "mid_file"
                ]
            )


            if (
                side_filename is None
                or
                center_filename is None
            ):

                print(
                    f"[SKIP CSV] row={row_number}: "
                    f"missing filename"
                )

                continue


            # =================================================
            # CSV 本身已有 index
            # =================================================

            index_text = row.get(
                "index",
                ""
            )


            try:

                pair_index = int(
                    index_text
                )

            except (
                ValueError,
                TypeError
            ):

                pair_index = row_number


            # 防止某些CSV index从0开始
            if pair_index <= 0:

                pair_index = row_number


            time_diff_ms = row.get(
                "time_diff_ms",
                ""
            )


            rows.append({

                "pair_index":
                    pair_index,

                "status":
                    status,

                "side_filename":
                    side_filename,

                "center_filename":
                    center_filename,

                "time_diff_ms":
                    time_diff_ms
            })


    return rows


# ============================================================
# 14. 单相机 PnP
# ============================================================

def solve_pose(
    object_points,
    image_points,
    K,
    D
):

    object_points = np.ascontiguousarray(
        object_points,
        dtype=np.float64
    )


    image_points = np.ascontiguousarray(
        image_points,
        dtype=np.float64
    )


    if len(
        object_points
    ) < 4:

        return None


    try:

        ok, rvec, tvec = cv2.solvePnP(

            object_points,
            image_points,

            K,
            D,

            flags=cv2.SOLVEPNP_ITERATIVE
        )

    except cv2.error:

        return None


    if not ok:

        return None


    try:

        rvec, tvec = cv2.solvePnPRefineLM(

            object_points,
            image_points,

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

        object_points,

        rvec,
        tvec,

        K,
        D
    )


    projected = projected.reshape(
        -1,
        2
    )


    observed = image_points.reshape(
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
            tvec.reshape(
                3
            ),

        "rmse":
            float(
                rmse
            )
    }


# ============================================================
# 15. 根据两个独立PnP计算 Center -> Side
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


    # Board -> Center
    #
    # Xc = Rc Xb + tc
    #
    # Board -> Side
    #
    # Xs = Rs Xb + ts
    #
    # Center -> Side
    #
    # Xs = Rsc Xc + tsc

    R_sc = (
        Rs
        @
        Rc.T
    )


    t_sc = (

        ts

        -

        R_sc
        @
        tc
    )


    return (
        R_sc,
        t_sc
    )


# ============================================================
# 16. Rotation angle
# ============================================================

def rotation_angle_deg(
    R
):

    cos_value = (

        np.trace(
            R
        )
        -
        1.0

    ) / 2.0


    cos_value = np.clip(
        cos_value,
        -1.0,
        1.0
    )


    return float(

        np.degrees(

            np.arccos(
                cos_value
            )
        )
    )


# ============================================================
# 17. 构建 matched 数据集
# ============================================================

def build_matched_dataset(

    side_name,

    side_dir,

    center_dir,

    csv_path,

    K_side,
    D_side,

    flip_side

):

    rows = load_pair_rows(
        csv_path,
        side_name
    )


    print()
    print("=" * 90)

    print(
        f"BUILD MATCHED CENTER -> "
        f"{side_name.upper()}"
    )

    print("=" * 90)


    print(
        "CSV pairs:",
        len(rows)
    )


    dataset = []


    for row_number, row in enumerate(
        rows,
        start=1
    ):


        pair_index = row[
            "pair_index"
        ]


        side_path = resolve_side_image_path(

            side_dir,

            row[
                "side_filename"
            ]
        )


        center_path = resolve_matched_center_path(

            center_dir,

            row[
                "center_filename"
            ],

            pair_index
        )


        # ====================================================
        # 前3对打印真实路径，确认命名对应
        # ====================================================

        if row_number <= 3:

            print()

            print(
                f"[PAIR {pair_index:04d}]"
            )

            print(
                "Side   :",
                side_path
            )

            print(
                "Center :",
                center_path
            )


        side_image = cv2.imread(
            str(
                side_path
            )
        )


        center_image = cv2.imread(
            str(
                center_path
            )
        )


        if side_image is None:

            print(

                f"[READ FAIL] "
                f"side: {side_path}"
            )

            continue


        if center_image is None:

            print(

                f"[READ FAIL] "
                f"center: {center_path}"
            )

            continue


        # ====================================================
        # 分辨率检查
        # ====================================================

        side_size = (

            side_image.shape[1],

            side_image.shape[0]
        )


        center_size = (

            center_image.shape[1],

            center_image.shape[0]
        )


        if (
            side_size
            !=
            IMAGE_SIZE
        ):

            print(

                f"[SIZE FAIL] "
                f"{side_path.name}: "
                f"{side_size}"
            )

            continue


        if (
            center_size
            !=
            IMAGE_SIZE
        ):

            print(

                f"[SIZE FAIL] "
                f"{center_path.name}: "
                f"{center_size}"
            )

            continue


        # ====================================================
        # ChArUco
        # ====================================================

        center_map = detect_charuco(

            center_image,

            FLIP_CENTER
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


        common_count = len(
            common_ids
        )


        if (
            common_count
            <
            MIN_COMMON_CORNERS
        ):

            print(

                f"[SKIP] "
                f"pair={pair_index:03d} "
                f"common={common_count}"
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


        # ====================================================
        # 独立PnP作为诊断
        # ====================================================

        center_pose = solve_pose(

            object_points,

            center_points,

            K_CENTER,
            D_CENTER
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
                f"pair={pair_index:03d}"
            )

            continue


        R_pair, \
        T_pair = relative_pose(

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


        rotation_deg = rotation_angle_deg(
            R_pair
        )


        dataset.append({

            "pair_index":
                pair_index,

            "status":
                row[
                    "status"
                ],

            "side_filename":
                side_path.name,

            "center_filename":
                center_path.name,

            "center_original_filename":
                row[
                    "center_filename"
                ],

            "time_diff_ms":
                row[
                    "time_diff_ms"
                ],

            "common_ids":
                common_ids,

            "common_count":
                common_count,

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

            "R_pair":
                R_pair,

            "T_pair":
                T_pair,

            "baseline_mm":
                float(
                    baseline_mm
                ),

            "rotation_deg":
                float(
                    rotation_deg
                )
        })


        print(

            f"pair={pair_index:03d} "

            f"common={common_count:2d} "

            f"PnP="
            f"{center_pose['rmse']:.3f}/"
            f"{side_pose['rmse']:.3f}px "

            f"baseline="
            f"{baseline_mm:8.2f} mm "

            f"rot="
            f"{rotation_deg:7.3f} deg"
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
# 18. 直接 stereoCalibrate
#
# Camera1 = Center
# Camera2 = Side
#
# OpenCV 返回：
#
# X_side = R * X_center + T
# ============================================================

def run_stereo_calibrate(

    dataset,

    K_side,
    D_side

):

    if (
        len(
            dataset
        )
        <
        3
    ):

        raise RuntimeError(
            "Too few stereo pairs."
        )


    object_points = []
    center_points = []
    side_points = []


    for item in dataset:


        # ====================================================
        # OpenCV 4.10 stereoCalibrate：
        #
        # Point3f / Point2f
        #
        # 必须使用 float32
        # ====================================================

        obj = np.asarray(

            item[
                "object_points"
            ],

            dtype=np.float32

        ).reshape(
            -1,
            1,
            3
        )


        center = np.asarray(

            item[
                "center_points"
            ],

            dtype=np.float32

        ).reshape(
            -1,
            1,
            2
        )


        side = np.asarray(

            item[
                "side_points"
            ],

            dtype=np.float32

        ).reshape(
            -1,
            1,
            2
        )


        obj = np.ascontiguousarray(
            obj
        )

        center = np.ascontiguousarray(
            center
        )

        side = np.ascontiguousarray(
            side
        )


        object_points.append(
            obj
        )

        center_points.append(
            center
        )

        side_points.append(
            side
        )


    print()

    print(
        "Stereo input pairs:",
        len(
            object_points
        )
    )


    print(
        "objectPoints:",
        object_points[0].shape,
        object_points[0].dtype
    )


    print(
        "centerPoints:",
        center_points[0].shape,
        center_points[0].dtype
    )


    print(
        "sidePoints  :",
        side_points[0].shape,
        side_points[0].dtype
    )


    criteria = (

        cv2.TERM_CRITERIA_EPS
        +
        cv2.TERM_CRITERIA_MAX_ITER,

        300,

        1e-10
    )


    rms, \
    Kc, Dc, \
    Ks, Ds, \
    R, T, E, F = cv2.stereoCalibrate(

        object_points,

        center_points,

        side_points,

        K_CENTER.copy(),

        D_CENTER.copy(),

        K_side.copy(),

        D_side.copy(),

        IMAGE_SIZE,

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
            ).reshape(
                3
            ),

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
# 19. 每一pair相对于最终Stereo外参的偏差
# ============================================================

def compute_pair_deviations(

    dataset,
    result

):

    R_final = result[
        "R"
    ]


    T_final = result[
        "T"
    ]


    for item in dataset:


        R_pair = item[
            "R_pair"
        ]


        T_pair = item[
            "T_pair"
        ]


        R_error = (

            R_pair

            @

            R_final.T
        )


        rotation_dev_deg = rotation_angle_deg(
            R_error
        )


        translation_dev_mm = (

            np.linalg.norm(

                T_pair

                -

                T_final
            )

            *
            1000.0
        )


        item[
            "rotation_deviation_deg"
        ] = float(
            rotation_dev_deg
        )


        item[
            "translation_deviation_mm"
        ] = float(
            translation_dev_mm
        )


# ============================================================
# 20. 4x4变换矩阵
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
    ] = T.reshape(
        3
    )


    return H


# ============================================================
# 21. 保存结果
# ============================================================

def save_result(

    side_name,

    dataset,

    result

):

    compute_pair_deviations(
        dataset,
        result
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


    # ========================================================
    # Side相机光心在Center坐标系中的坐标
    #
    # C_side_center = -R^T T
    # ========================================================

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

        f"center_to_{side_name}.yaml"
    )


    fs = cv2.FileStorage(

        str(
            yaml_path
        ),

        cv2.FILE_STORAGE_WRITE
    )


    fs.write(

        "transform_definition",

        f"X_{side_name} = "
        f"R_center_to_{side_name} * "
        f"X_center + T_center_to_{side_name}"
    )


    fs.write(

        f"R_center_to_{side_name}",

        R
    )


    fs.write(

        f"T_center_to_{side_name}_m",

        T.reshape(
            3,
            1
        )
    )


    fs.write(

        f"H_center_to_{side_name}",

        H
    )


    fs.write(

        "baseline_m",

        baseline_m
    )


    fs.write(

        "stereo_rms_px",

        float(
            result[
                "rms"
            ]
        )
    )


    fs.write(

        f"{side_name}_camera_center_in_center_m",

        camera_center.reshape(
            3,
            1
        )
    )


    fs.write(
        "E",
        result[
            "E"
        ]
    )


    fs.write(
        "F",
        result[
            "F"
        ]
    )


    fs.release()


    # ========================================================
    # 每对统计CSV
    # ========================================================

    csv_path = (

        OUTPUT_DIR

        /

        f"center_{side_name}_pair_metrics.csv"
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

            "status",

            "center_filename",

            f"{side_name}_filename",

            "time_diff_ms",

            "common_count",

            "center_pnp_rmse_px",

            f"{side_name}_pnp_rmse_px",

            "pair_baseline_mm",

            "pair_rotation_deg",

            "rotation_deviation_from_final_deg",

            "translation_deviation_from_final_mm"
        ])


        for item in dataset:


            writer.writerow([

                item[
                    "pair_index"
                ],

                item[
                    "status"
                ],

                item[
                    "center_filename"
                ],

                item[
                    "side_filename"
                ],

                item[
                    "time_diff_ms"
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
                    "baseline_mm"
                ],

                item[
                    "rotation_deg"
                ],

                item[
                    "rotation_deviation_deg"
                ],

                item[
                    "translation_deviation_mm"
                ]
            ])


    return (
        yaml_path,
        csv_path,
        H
    )


# ============================================================
# 22. 打印最终结果
# ============================================================

def print_final_result(

    side_name,

    dataset,

    result

):

    compute_pair_deviations(
        dataset,
        result
    )


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


    rotation_deg = rotation_angle_deg(
        R
    )


    camera_center = (

        -
        R.T

        @

        T
    )


    print()
    print("=" * 90)

    print(
        f"FINAL MATCHED CENTER -> "
        f"{side_name.upper()}"
    )

    print("=" * 90)


    print(
        "Pairs:",
        len(
            dataset
        )
    )


    print(
        "Stereo RMS:",
        f"{result['rms']:.6f} px"
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
        f"{rotation_deg:.3f} deg"
    )


    print()

    print(
        f"{side_name} camera center "
        f"in Center coordinates [m]:"
    )

    print(
        camera_center
    )


    # ========================================================
    # 独立PnP统计
    # ========================================================

    baselines = np.asarray(
        [
            x[
                "baseline_mm"
            ]

            for x
            in dataset
        ],
        dtype=np.float64
    )


    rotations = np.asarray(
        [
            x[
                "rotation_deg"
            ]

            for x
            in dataset
        ],
        dtype=np.float64
    )


    center_errors = np.asarray(
        [
            x[
                "center_pnp_rmse"
            ]

            for x
            in dataset
        ],
        dtype=np.float64
    )


    side_errors = np.asarray(
        [
            x[
                "side_pnp_rmse"
            ]

            for x
            in dataset
        ],
        dtype=np.float64
    )


    rot_devs = np.asarray(
        [
            x[
                "rotation_deviation_deg"
            ]

            for x
            in dataset
        ],
        dtype=np.float64
    )


    trans_devs = np.asarray(
        [
            x[
                "translation_deviation_mm"
            ]

            for x
            in dataset
        ],
        dtype=np.float64
    )


    print()
    print(
        "Independent PnP statistics:"
    )


    print(
        "  Center PnP median:",
        f"{np.median(center_errors):.3f} px"
    )


    print(
        f"  {side_name.capitalize()} PnP median:",
        f"{np.median(side_errors):.3f} px"
    )


    print()

    print(
        "  baseline median:",
        f"{np.median(baselines):.3f} mm"
    )


    print(
        "  baseline mean  :",
        f"{np.mean(baselines):.3f} mm"
    )


    print(
        "  baseline std   :",
        f"{np.std(baselines):.3f} mm"
    )


    print()

    print(
        "  rotation median:",
        f"{np.median(rotations):.3f} deg"
    )


    print(
        "  rotation mean  :",
        f"{np.mean(rotations):.3f} deg"
    )


    print(
        "  rotation std   :",
        f"{np.std(rotations):.3f} deg"
    )


    print()

    print(
        "  final R deviation median:",
        f"{np.median(rot_devs):.3f} deg"
    )


    print(
        "  final T deviation median:",
        f"{np.median(trans_devs):.3f} mm"
    )


# ============================================================
# 23. Main
# ============================================================

def main():


    # ========================================================
    # Center -> Left
    # ========================================================

    left_dataset = build_matched_dataset(

        side_name="left",

        side_dir=LEFT_DIR,

        center_dir=
            MATCHED_CENTER_LEFT_DIR,

        csv_path=
            LEFT_PAIR_CSV,

        K_side=
            K_LEFT,

        D_side=
            D_LEFT,

        flip_side=
            FLIP_LEFT
    )


    if len(
        left_dataset
    ) < 3:

        raise RuntimeError(
            "Too few usable Center-Left pairs."
        )


    left_result = run_stereo_calibrate(

        left_dataset,

        K_LEFT,
        D_LEFT
    )


    print_final_result(

        "left",

        left_dataset,

        left_result
    )


    (
        left_yaml,
        left_csv,
        H_left_from_center

    ) = save_result(

        "left",

        left_dataset,

        left_result
    )


    # ========================================================
    # Center -> Right
    # ========================================================

    right_dataset = build_matched_dataset(

        side_name="right",

        side_dir=RIGHT_DIR,

        center_dir=
            MATCHED_CENTER_RIGHT_DIR,

        csv_path=
            RIGHT_PAIR_CSV,

        K_side=
            K_RIGHT,

        D_side=
            D_RIGHT,

        flip_side=
            FLIP_RIGHT
    )


    if len(
        right_dataset
    ) < 3:

        raise RuntimeError(
            "Too few usable Center-Right pairs."
        )


    right_result = run_stereo_calibrate(

        right_dataset,

        K_RIGHT,
        D_RIGHT
    )


    print_final_result(

        "right",

        right_dataset,

        right_result
    )


    (
        right_yaml,
        right_csv,
        H_right_from_center

    ) = save_result(

        "right",

        right_dataset,

        right_result
    )


    # ========================================================
    # Left -> Right
    #
    # H_RC = Center -> Right
    # H_LC = Center -> Left
    #
    # Left -> Right:
    #
    # H_RL = H_RC * inv(H_LC)
    # ========================================================

    H_right_from_left = (

        H_right_from_center

        @

        np.linalg.inv(
            H_left_from_center
        )
    )


    # ========================================================
    # 保存完整 Rig
    # ========================================================

    rig_yaml = (

        OUTPUT_DIR

        /

        "rig_extrinsics.yaml"
    )


    fs = cv2.FileStorage(

        str(
            rig_yaml
        ),

        cv2.FILE_STORAGE_WRITE
    )


    fs.write(
        "reference_camera",
        "center"
    )


    fs.write(

        "H_left_from_center",

        H_left_from_center
    )


    fs.write(

        "H_right_from_center",

        H_right_from_center
    )


    fs.write(

        "H_right_from_left",

        H_right_from_left
    )


    fs.release()


    # ========================================================
    # 输出
    # ========================================================

    print()
    print("=" * 90)
    print("OUTPUT")
    print("=" * 90)


    print(
        "Center -> Left :",
        left_yaml
    )


    print(
        "Center -> Right:",
        right_yaml
    )


    print(
        "Rig extrinsic  :",
        rig_yaml
    )


    print(
        "Left metrics   :",
        left_csv
    )


    print(
        "Right metrics  :",
        right_csv
    )


    print()
    print(
        "Finished."
    )


if __name__ == "__main__":

    main()