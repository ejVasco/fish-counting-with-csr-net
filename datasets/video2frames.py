# datasets/video2frames.py

import os
import sys

import cv2


def extract_frames(video_path: str, num_frames: int):
    if not os.path.exists(video_path):
        print(f"error: video file not found {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: failed to open video: {video_path}")
        return

    # prevent capturing more frames than is in video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames > total_frames:
        print(
            f"warn: requested frame count {num_frames}, when video only has {total_frames} frames"
        )
        num_frames = total_frames

    # distribute the frames
    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        step = (total_frames - 1) / (num_frames - 1)
        indices = [round(i * step) for i in range(num_frames)]

    # mk output foldher
    video_dir = os.path.dirname(os.path.abspath(video_path))
    output_dir = os.path.join(
        video_dir,
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"extracting {num_frames} frames from '{video_path}' into '{output_dir}'")

    extracted = 0
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"warn: could not read frame {i}, skipped")
            continue
        filename = os.path.join(output_dir, f"image_{i:06d}.jpg")
        cv2.imwrite(filename, frame)
        extracted += 1

    cap.release()
    print(f"done. {extracted} frames saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USage: python -m datasets.video2frames <vid_path> <num_frames>")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        num_frames = int(sys.argv[2])
        if num_frames <= 0:
            raise ValueError
    except ValueError:
        print("Error: invalid num_frames, num_frames must be a positive int")
        sys.exit(1)

    num_frames = int(sys.argv[2])
    extract_frames(video_path, num_frames)
