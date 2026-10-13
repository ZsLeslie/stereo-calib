import cv2
import csv
import math
import shutil
import numpy as np
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

LEFT_DIR = Path("./frames/left")
CENTER_DIR = Path("./frames/center")
RIGHT_DIR = Path("./frames/right")

LEFT_CENTER_CSV = Path(
    "./common_charuco_check/center_left_common_ids.csv"
)

RIGHT_CENTER_CSV = Path(
    "./common_charuco_check/center_right_common_ids.csv"
)


LEFT_INTRINSIC = Path(
    "./left_camera_intrinsic.yaml"
)

# 修改成你现在实际使用的新中相机内参
CENTER_INTRINSIC = Path(
    "./mid_camera_intrinsic.yaml"
)

RIGHT_INTRINSIC = Path(
    "./right_camera_intrinsic.yaml"
)


OUTPUT_ROOT = Path(
    "./pose_consistency"
)


# ============================================================
# 2. 三台相机都需要水平翻转
#
# 必须和内参标定时完全一致
# ============================================================

FLIP_LEFT = True
FLIP_CENTER = True
FLIP_RIGHT = True


# ============================================================
# 3. ChArUco 参数
#
# 与内参标定完全一致
# ============================================================

SQUARES_X = 7
SQUARES_Y = 5

SQUARE_LENGTH = 0.04665
MARKER_LENGTH = 0.03265

DICTIONARY_ID = cv2.aruco.DICT_5X5_100


# ============================================================
# 4. 基本筛选参数
# ============================================================

MIN_COMMON_IDS = 8

MAX_PNP_RMSE_PX = 1.0


# ============================================================
# 5. 重投影可视化配置
# ============================================================

# 是否保存所有Center的PnP可视化
SAVE_ALL_CENTER_REPROJECTION = True

# JPEG质量
JPEG_QUALITY = 95


# ------------------------------------------------------------
# 自动寻找最接近这4个RMSE的代表帧
# ------------------------------------------------------------

TARGET_RMSE_VALUES = [
    0.6,
    5.0,
    15.0,
    50.0
]


# ============================================================
# 创建输出目录
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


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
# 加载三台内参
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
print("=" * 80)
print("CENTER INTRINSIC")
print("=" * 80)

print(
    "File:"
)

print(
    CENTER_INTRINSIC.resolve()
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

print("=" * 80)


# ============================================================
# 7. 创建 ChArUco Board
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


parameters = cv2.aruco.DetectorParameters()

parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 53
parameters.adaptiveThreshWinSizeStep = 10

parameters.minMarkerPerimeterRate = 0.003
parameters.maxMarkerPerimeterRate = 4.0

parameters.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)


detector = cv2.aruco.CharucoDetector(
    board,
    detectorParams=parameters
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
    "Square length:",
    SQUARE_LENGTH * 1000.0,
    "mm"
)

print(
    "Marker length:",
    MARKER_LENGTH * 1000.0,
    "mm"
)

print(
    "Charuco corner count:",
    len(BOARD_POINTS)
)

print()


# ============================================================
# 8. ChArUco检测
# ============================================================

def detect_charuco(
    image,
    flip_horizontal
):

    if image is None:

        return (
            None,
            {}
        )


    if flip_horizontal:

        image = cv2.flip(
            image,
            1
        )


    charuco_corners, \
    charuco_ids, \
    marker_corners, \
    marker_ids = detector.detectBoard(
        image
    )


    point_map = {}


    if (
        charuco_ids is None
        or
        charuco_corners is None
    ):

        return (
            image,
            point_map
        )


    ids = charuco_ids.reshape(
        -1
    )

    corners = charuco_corners.reshape(
        -1,
        2
    )


    for cid, point in zip(
        ids,
        corners
    ):

        point_map[
            int(cid)
        ] = np.asarray(
            point,
            dtype=np.float64
        )


    return (
        image,
        point_map
    )


# ============================================================
# 9. 共同 ChArUco ID
# ============================================================

def build_common_points(
    center_map,
    side_map
):

    common_ids = sorted(
        set(
            center_map.keys()
        )
        &
        set(
            side_map.keys()
        )
    )


    object_points = []
    center_points = []
    side_points = []

    valid_ids = []


    for cid in common_ids:

        if (
            cid < 0
            or
            cid >= len(BOARD_POINTS)
        ):

            continue


        valid_ids.append(
            cid
        )


        object_points.append(
            BOARD_POINTS[cid]
        )


        center_points.append(
            center_map[cid]
        )


        side_points.append(
            side_map[cid]
        )


    return (

        valid_ids,

        np.asarray(
            object_points,
            dtype=np.float64
        ),

        np.asarray(
            center_points,
            dtype=np.float64
        ),

        np.asarray(
            side_points,
            dtype=np.float64
        )
    )


# ============================================================
# 10. solvePnP
# ============================================================

def solve_camera_pose(
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


    # --------------------------------------------------------
    # 初始PnP
    # --------------------------------------------------------

    ok, rvec, tvec = cv2.solvePnP(

        object_points,
        image_points,

        K,
        D,

        flags=cv2.SOLVEPNP_ITERATIVE
    )


    if not ok:

        return None


    # --------------------------------------------------------
    # LM精修
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    R, _ = cv2.Rodrigues(
        rvec
    )


    # --------------------------------------------------------
    # PnP重投影
    # --------------------------------------------------------

    projected_points, _ = cv2.projectPoints(

        object_points,

        rvec,
        tvec,

        K,
        D
    )


    projected_points = projected_points.reshape(
        -1,
        2
    )


    observed_points = image_points.reshape(
        -1,
        2
    )


    error_vectors = (
        projected_points
        -
        observed_points
    )


    per_point_errors = np.linalg.norm(
        error_vectors,
        axis=1
    )


    rmse = math.sqrt(

        float(

            np.mean(

                np.sum(
                    error_vectors ** 2,
                    axis=1
                )
            )
        )
    )


    # --------------------------------------------------------
    # 深度检查
    # --------------------------------------------------------

    camera_points = (

        R
        @
        object_points.T

        +

        tvec.reshape(
            3,
            1
        )
    )


    positive_depth_ratio = float(

        np.mean(

            camera_points[
                2,
                :
            ] > 0
        )
    )


    return {

        "R":
            R,

        "t":
            tvec.reshape(
                3
            ),

        "rvec":
            rvec.reshape(
                3
            ),

        "reproj_rmse":
            rmse,

        "positive_depth_ratio":
            positive_depth_ratio,

        # 新增
        "observed_points":
            observed_points.copy(),

        "projected_points":
            projected_points.copy(),

        "per_point_errors":
            per_point_errors.copy()
    }


# ============================================================
# 11. Center -> Side
# ============================================================

def calculate_relative_pose(
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
# 12. 旋转角
# ============================================================

def rotation_angle_deg(
    R
):

    rvec, _ = cv2.Rodrigues(
        R
    )


    return float(

        np.linalg.norm(
            rvec
        )

        *
        180.0
        /
        math.pi
    )


# ============================================================
# 13. 根据RMSE创建分类目录名
# ============================================================

def get_rmse_category(
    rmse
):

    if rmse < 1.0:

        return "00_rmse_lt_1"


    elif rmse < 5.0:

        return "01_rmse_1_to_5"


    elif rmse < 15.0:

        return "02_rmse_5_to_15"


    elif rmse < 30.0:

        return "03_rmse_15_to_30"


    else:

        return "04_rmse_ge_30"


# ============================================================
# 14. PnP重投影可视化
# ============================================================

def draw_reprojection_visualization(

    image,

    observed_points,
    projected_points,
    per_point_errors,

    common_ids,

    rmse,

    pair_index,
    side_name,

    time_diff_ms=""
):

    vis = image.copy()


    # ========================================================
    # 每一个ChArUco角点
    # ========================================================

    for (
        cid,
        observed,
        projected,
        error
    ) in zip(

        common_ids,
        observed_points,
        projected_points,
        per_point_errors
    ):


        ox = int(
            round(
                observed[0]
            )
        )

        oy = int(
            round(
                observed[1]
            )
        )


        px = int(
            round(
                projected[0]
            )
        )

        py = int(
            round(
                projected[1]
            )
        )


        # ----------------------------------------------------
        # 蓝线：
        #
        # 实测 -> PnP预测
        # ----------------------------------------------------

        cv2.line(

            vis,

            (
                ox,
                oy
            ),

            (
                px,
                py
            ),

            (
                255,
                0,
                0
            ),

            2,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # 绿色：
        #
        # 实际检测的ChArUco角点
        # ----------------------------------------------------

        cv2.circle(

            vis,

            (
                ox,
                oy
            ),

            7,

            (
                0,
                255,
                0
            ),

            -1,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # 红色十字：
        #
        # PnP重投影点
        # ----------------------------------------------------

        cv2.drawMarker(

            vis,

            (
                px,
                py
            ),

            (
                0,
                0,
                255
            ),

            markerType=cv2.MARKER_CROSS,

            markerSize=18,

            thickness=3,

            line_type=cv2.LINE_AA
        )


        # ----------------------------------------------------
        # ID + 单点误差
        # ----------------------------------------------------

        text = (
            f"{cid}:{error:.1f}"
        )


        cv2.putText(

            vis,

            text,

            (
                ox + 10,
                oy - 10
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                255,
                255,
                0
            ),

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # 顶部信息区
    # ========================================================

    cv2.rectangle(

        vis,

        (
            0,
            0
        ),

        (
            vis.shape[1],
            135
        ),

        (
            0,
            0,
            0
        ),

        -1
    )


    line1 = (

        f"CENTER PnP  "
        f"{side_name.upper()} pair={pair_index:03d}"
    )


    line2 = (

        f"RMSE = {rmse:.3f} px    "
        f"Common = {len(common_ids)}"
    )


    line3 = (

        f"Green=observed   "
        f"Red=projected   "
        f"Blue=error vector"
    )


    if time_diff_ms != "":

        line2 += (
            f"    dt={time_diff_ms} ms"
        )


    cv2.putText(

        vis,

        line1,

        (
            30,
            38
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        vis,

        line2,

        (
            30,
            78
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (
            0,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        vis,

        line3,

        (
            30,
            116
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    return vis


# ============================================================
# 15. 保存Center重投影图
# ============================================================

def save_center_reprojection(

    side_name,

    pair_index,

    center_filename,

    center_image,

    center_pose,

    common_ids,

    time_diff_ms
):

    root_dir = (

        OUTPUT_ROOT
        /
        "center_reprojection"
        /
        side_name
    )


    category = get_rmse_category(

        center_pose[
            "reproj_rmse"
        ]
    )


    category_dir = (
        root_dir
        /
        category
    )


    category_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    vis = draw_reprojection_visualization(

        center_image,

        center_pose[
            "observed_points"
        ],

        center_pose[
            "projected_points"
        ],

        center_pose[
            "per_point_errors"
        ],

        common_ids,

        center_pose[
            "reproj_rmse"
        ],

        pair_index,

        side_name,

        time_diff_ms
    )


    filename = (

        f"{side_name}_"
        f"pair_{pair_index:03d}_"
        f"rmse_{center_pose['reproj_rmse']:.3f}_"
        f"{Path(center_filename).stem}.jpg"
    )


    output_path = (
        category_dir
        /
        filename
    )


    cv2.imwrite(

        str(
            output_path
        ),

        vis,

        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY
        ]
    )


    return output_path


# ============================================================
# 16. 读取pair CSV
# ============================================================

def load_pairs(
    csv_path
):

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


        for row in reader:

            if row.get(
                "status",
                ""
            ) == "REJECT":

                continue


            rows.append(
                row
            )


    return rows


# ============================================================
# 17. 处理一组 Center-Side
# ============================================================

def process_pair_group(

    side_name,

    side_dir,

    csv_path,

    K_side,
    D_side,

    flip_side
):

    print()
    print("=" * 90)

    print(
        f"CENTER -> {side_name.upper()}"
    )

    print("=" * 90)


    rows = load_pairs(
        csv_path
    )


    print(
        "Input pairs:",
        len(rows)
    )


    results = []


    for pair_index, row in enumerate(
        rows,
        start=1
    ):


        center_filename = row[
            "center_filename"
        ]


        side_filename = row[
            f"{side_name}_filename"
        ]


        center_path = (

            CENTER_DIR
            /
            center_filename
        )


        side_path = (

            side_dir
            /
            side_filename
        )


        center_image = cv2.imread(
            str(center_path)
        )


        side_image = cv2.imread(
            str(side_path)
        )


        if (
            center_image is None
            or
            side_image is None
        ):

            print(
                f"[READ FAIL] pair={pair_index:03d}"
            )

            continue


        # ====================================================
        # ChArUco检测
        # ====================================================

        center_image, \
        center_map = detect_charuco(

            center_image,
            FLIP_CENTER
        )


        side_image, \
        side_map = detect_charuco(

            side_image,
            flip_side
        )


        common_ids, \
        object_points, \
        center_points, \
        side_points = build_common_points(

            center_map,
            side_map
        )


        common_count = len(
            common_ids
        )


        if common_count < MIN_COMMON_IDS:

            print(

                f"[SKIP] "
                f"pair={pair_index:03d} "
                f"common={common_count}"
            )

            continue


        # ====================================================
        # Center PnP
        # ====================================================

        center_pose = solve_camera_pose(

            object_points,
            center_points,

            K_CENTER,
            D_CENTER
        )


        # ====================================================
        # Side PnP
        # ====================================================

        side_pose = solve_camera_pose(

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


        # ====================================================
        # Center -> Side
        # ====================================================

        R_rel, \
        t_rel = calculate_relative_pose(

            center_pose,
            side_pose
        )


        baseline = float(

            np.linalg.norm(
                t_rel
            )
        )


        rot_deg = rotation_angle_deg(
            R_rel
        )


        time_diff_ms = row.get(
            "time_diff_ms",
            ""
        )


        # ====================================================
        # 保存Center重投影图
        # ====================================================

        reprojection_path = None


        if SAVE_ALL_CENTER_REPROJECTION:

            reprojection_path = save_center_reprojection(

                side_name,

                pair_index,

                center_filename,

                center_image,

                center_pose,

                common_ids,

                time_diff_ms
            )


        # ====================================================
        # 保存结果
        # ====================================================

        result = {

            "side_name":
                side_name,

            "pair_index":
                pair_index,

            "center_filename":
                center_filename,

            "side_filename":
                side_filename,

            "common_count":
                common_count,

            "common_ids":
                common_ids,

            "center_pnp_rmse":
                center_pose[
                    "reproj_rmse"
                ],

            "side_pnp_rmse":
                side_pose[
                    "reproj_rmse"
                ],

            "baseline":
                baseline,

            "rotation_deg":
                rot_deg,

            "time_diff_ms":
                time_diff_ms,

            "reprojection_path":
                reprojection_path,

            "center_image":
                center_image,

            "center_pose":
                center_pose
        }


        results.append(
            result
        )


        print(

            f"pair={pair_index:03d} "

            f"common={common_count:2d} "

            f"baseline="
            f"{baseline * 1000.0:8.2f} mm "

            f"rot="
            f"{rot_deg:7.3f} deg "

            f"PnP="
            f"{center_pose['reproj_rmse']:.3f}/"
            f"{side_pose['reproj_rmse']:.3f}px"
        )


    return results


# ============================================================
# 18. 保存所有结果CSV
# ============================================================

def save_metrics_csv(
    side_name,
    results
):

    output_dir = (
        OUTPUT_ROOT
        /
        f"center_{side_name}"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_csv = (
        output_dir
        /
        "pair_pose_metrics.csv"
    )


    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow([

            "pair_index",

            "center_filename",

            f"{side_name}_filename",

            "common_count",

            "center_pnp_rmse_px",

            f"{side_name}_pnp_rmse_px",

            "baseline_mm",

            "rotation_deg",

            "time_diff_ms",

            "reprojection_path"
        ])


        for r in results:

            writer.writerow([

                r[
                    "pair_index"
                ],

                r[
                    "center_filename"
                ],

                r[
                    "side_filename"
                ],

                r[
                    "common_count"
                ],

                r[
                    "center_pnp_rmse"
                ],

                r[
                    "side_pnp_rmse"
                ],

                r[
                    "baseline"
                ]
                *
                1000.0,

                r[
                    "rotation_deg"
                ],

                r[
                    "time_diff_ms"
                ],

                str(
                    r[
                        "reprojection_path"
                    ]
                )
            ])


    return output_csv


# ============================================================
# 19. 找最接近目标RMSE的代表图
# ============================================================

def save_representative_samples(
    all_results
):

    representative_dir = (

        OUTPUT_ROOT
        /
        "center_reprojection"
        /
        "representative_samples"
    )


    representative_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    csv_path = (

        representative_dir
        /
        "representative_samples.csv"
    )


    rows = []


    print()
    print("=" * 90)
    print("CENTER PnP REPRESENTATIVE SAMPLES")
    print("=" * 90)


    for target_rmse in TARGET_RMSE_VALUES:


        # ----------------------------------------------------
        # 从全部Center-Left/Right结果里找最接近目标RMSE的
        # ----------------------------------------------------

        sample = min(

            all_results,

            key=lambda r:

                abs(
                    r[
                        "center_pnp_rmse"
                    ]
                    -
                    target_rmse
                )
        )


        actual_rmse = sample[
            "center_pnp_rmse"
        ]


        src_path = sample[
            "reprojection_path"
        ]


        if src_path is None:

            continue


        target_text = (
            str(target_rmse)
            .replace(
                ".",
                "p"
            )
        )


        output_name = (

            f"target_{target_text}px_"
            f"actual_{actual_rmse:.3f}px_"
            f"{sample['side_name']}_"
            f"pair_{sample['pair_index']:03d}.jpg"
        )


        dst_path = (

            representative_dir
            /
            output_name
        )


        shutil.copy2(

            src_path,
            dst_path
        )


        print()

        print(
            f"Target RMSE : "
            f"{target_rmse:.1f}px"
        )

        print(
            f"Actual RMSE : "
            f"{actual_rmse:.3f}px"
        )

        print(
            f"Group       : "
            f"Center-{sample['side_name'].capitalize()}"
        )

        print(
            f"Pair        : "
            f"{sample['pair_index']:03d}"
        )

        print(
            f"Image       : "
            f"{sample['center_filename']}"
        )

        print(
            f"Saved       : "
            f"{dst_path}"
        )


        rows.append([

            target_rmse,

            actual_rmse,

            sample[
                "side_name"
            ],

            sample[
                "pair_index"
            ],

            sample[
                "center_filename"
            ],

            sample[
                "side_filename"
            ],

            sample[
                "common_count"
            ],

            sample[
                "time_diff_ms"
            ],

            str(
                dst_path
            )
        ])


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

            "target_rmse_px",

            "actual_rmse_px",

            "group",

            "pair_index",

            "center_filename",

            "side_filename",

            "common_count",

            "time_diff_ms",

            "image_path"
        ])


        writer.writerows(
            rows
        )


    print()
    print(
        "Representative CSV:",
        csv_path
    )


# ============================================================
# 20. 简单RMSE统计
# ============================================================

def print_rmse_statistics(
    side_name,
    results
):

    values = np.asarray(

        [
            r[
                "center_pnp_rmse"
            ]

            for r in results
        ],

        dtype=np.float64
    )


    if len(
        values
    ) == 0:

        return


    print()
    print(
        "-" * 80
    )

    print(
        f"CENTER-{side_name.upper()} "
        f"Center PnP RMSE statistics"
    )

    print(
        "-" * 80
    )


    print(
        "Count :",
        len(values)
    )

    print(
        "Min   :",
        f"{np.min(values):.3f}px"
    )

    print(
        "Median:",
        f"{np.median(values):.3f}px"
    )

    print(
        "Mean  :",
        f"{np.mean(values):.3f}px"
    )

    print(
        "Max   :",
        f"{np.max(values):.3f}px"
    )


    print(
        "< 1px :",
        int(
            np.sum(
                values < 1.0
            )
        )
    )

    print(
        "1-5px :",
        int(
            np.sum(
                (
                    values >= 1.0
                )
                &
                (
                    values < 5.0
                )
            )
        )
    )

    print(
        "5-15px:",
        int(
            np.sum(
                (
                    values >= 5.0
                )
                &
                (
                    values < 15.0
                )
            )
        )
    )

    print(
        "15-30px:",
        int(
            np.sum(
                (
                    values >= 15.0
                )
                &
                (
                    values < 30.0
                )
            )
        )
    )

    print(
        ">=30px:",
        int(
            np.sum(
                values >= 30.0
            )
        )
    )


# ============================================================
# Main
# ============================================================

def main():


    # ========================================================
    # Center -> Left
    # ========================================================

    left_results = process_pair_group(

        side_name="left",

        side_dir=LEFT_DIR,

        csv_path=LEFT_CENTER_CSV,

        K_side=K_LEFT,

        D_side=D_LEFT,

        flip_side=FLIP_LEFT
    )


    save_metrics_csv(

        "left",
        left_results
    )


    print_rmse_statistics(

        "left",
        left_results
    )


    # ========================================================
    # Center -> Right
    # ========================================================

    right_results = process_pair_group(

        side_name="right",

        side_dir=RIGHT_DIR,

        csv_path=RIGHT_CENTER_CSV,

        K_side=K_RIGHT,

        D_side=D_RIGHT,

        flip_side=FLIP_RIGHT
    )


    save_metrics_csv(

        "right",
        right_results
    )


    print_rmse_statistics(

        "right",
        right_results
    )


    # ========================================================
    # 汇总全部结果
    # ========================================================

    all_results = (

        left_results
        +
        right_results
    )


    if len(
        all_results
    ) > 0:

        save_representative_samples(
            all_results
        )


    print()
    print("=" * 90)

    print(
        "Finished"
    )

    print("=" * 90)


if __name__ == "__main__":

    main()