"""Play a local video in an OpenCV window from the terminal.

Keys: Space pauses/resumes; Q or Esc closes the window.
"""

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the MP4 video")
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    delay = max(1, round(1000 / fps))
    window = "Pose2Arm video replay | Space: pause | Q/Esc: close"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    paused = False
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break
                cv2.imshow(window, frame)
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
