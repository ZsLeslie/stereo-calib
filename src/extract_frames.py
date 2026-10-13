import cv2
import csv
from pathlib import Path


# ============================================================
# 用户配置
# ============================================================

VIDEOS = {
    "left": Path("./videos/left.mp4"),
    "center": Path("./videos/mid.mp4"),
    "right": Path("./videos/right.mp4"),
}

OUTPUT_ROOT = Path("./frames")

IMAGE_SUFFIX = ".png"

# 每隔多少帧保存一张
FRAME_STEP = 30


# ============================================================
# 抽取一个视频
# ============================================================

def extract_video(camera_name: str, video_path: Path):

    print()
    print("=" * 70)
    print(f"Camera : {camera_name}")
    print(f"Video  : {video_path}")
    print("=" * 70)

    if not video_path.exists():
        print(f"[ERROR] Video does not exist: {video_path}")
        return

    output_dir = OUTPUT_ROOT / camera_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS for {video_path}: {fps}"
        )

    duration = frame_count / fps

    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.6f}")
    print(f"Frames     : {frame_count}")
    print(f"Duration   : {duration:.3f} s")
    print(f"Frame step : {FRAME_STEP}")
    print()

    csv_path = output_dir / "timestamps.csv"

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "camera",
        "frame_index",
        "timestamp_ms",
        "timestamp_sec",
        "filename"
    ])

    frame_index = 0
    saved_count = 0
    previous_timestamp_ms = -1.0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        # ----------------------------------------------------
        # 只保存每隔 FRAME_STEP 帧的数据
        # ----------------------------------------------------

        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        # ----------------------------------------------------
        # 获取当前视频时间戳
        # ----------------------------------------------------

        timestamp_ms = cap.get(
            cv2.CAP_PROP_POS_MSEC
        )

        # 如果POS_MSEC异常，则根据帧号和FPS计算
        if (
            timestamp_ms < 0
            or
            (saved_count > 0 and
             timestamp_ms <= previous_timestamp_ms)
        ):
            timestamp_ms = (
                frame_index / fps * 1000.0
            )

        timestamp_sec = (
            timestamp_ms / 1000.0
        )

        previous_timestamp_ms = timestamp_ms

        # ----------------------------------------------------
        # 文件名
        #
        # 使用视频时间戳微秒命名
        # ----------------------------------------------------

        timestamp_us = int(
            round(timestamp_ms * 1000.0)
        )

        filename = (
            f"{camera_name}_"
            f"{timestamp_us:015d}"
            f"{IMAGE_SUFFIX}"
        )

        output_path = (
            output_dir / filename
        )

        # ----------------------------------------------------
        # 保存PNG
        # ----------------------------------------------------

        success = cv2.imwrite(
            str(output_path),
            frame
        )

        if not success:
            print(
                f"[ERROR] Cannot save: "
                f"{output_path}"
            )
            break

        # ----------------------------------------------------
        # 保存CSV
        # ----------------------------------------------------

        writer.writerow([
            camera_name,
            frame_index,
            f"{timestamp_ms:.6f}",
            f"{timestamp_sec:.9f}",
            filename
        ])

        print(
            f"[{camera_name}] "
            f"frame={frame_index:6d} "
            f"time={timestamp_sec:9.3f}s "
            f"saved={saved_count:5d}"
        )

        saved_count += 1
        frame_index += 1

    csv_file.close()
    cap.release()

    print()
    print(
        f"[DONE] {camera_name}"
    )

    print(
        f"Original frames : {frame_count}"
    )

    print(
        f"Saved frames    : {saved_count}"
    )

    print(
        f"Images          : {output_dir}"
    )

    print(
        f"Timestamp CSV   : {csv_path}"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for camera_name, video_path in VIDEOS.items():

        extract_video(
            camera_name,
            video_path
        )


if __name__ == "__main__":
    main()