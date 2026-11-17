"""
Generate synthetic video files for testing and offline drone demonstration.
Generates moving human-like silhouettes against background zones to test detection, tracking, flow, and density.
"""

import cv2
import numpy as np
from pathlib import Path


def generate_synthetic_crowd_video(
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: float = 20.0,
    duration_sec: int = 5,
    num_people: int = 15,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    total_frames = int(fps * duration_sec)

    # Initialize people positions and velocities
    np.random.seed(42)
    positions = []
    for _ in range(num_people):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-2, 2)
        color = (
            int(np.random.randint(30, 80)),
            int(np.random.randint(30, 100)),
            int(np.random.randint(120, 220)),
        )
        positions.append({"x": x, "y": y, "vx": vx, "vy": vy, "color": color})

    for f in range(total_frames):
        # Draw background (pavement / plaza texture)
        frame = np.full((height, width, 3), 190, dtype=np.uint8)

        # Draw simulated walkways / bottleneck lines
        cv2.rectangle(frame, (100, 80), (1180, 640), (170, 170, 170), -1)
        # Narrow passage / bottleneck guide
        cv2.line(frame, (600, 80), (600, 280), (140, 140, 140), 4)
        cv2.line(frame, (680, 80), (680, 280), (140, 140, 140), 4)

        # Draw people (head, torso, legs to form humanoid shape recognizable by detectors or simple tracking)
        for p in positions:
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Bounce off walls
            if p["x"] < 120 or p["x"] > width - 120:
                p["vx"] *= -1
            if p["y"] < 120 or p["y"] > height - 120:
                p["vy"] *= -1

            cx, cy = int(p["x"]), int(p["y"])

            # Head
            cv2.circle(frame, (cx, cy - 25), 10, (50, 50, 50), -1)
            # Torso
            cv2.rectangle(frame, (cx - 12, cy - 15), (cx + 12, cy + 20), p["color"], -1)
            # Legs
            cv2.line(frame, (cx - 6, cy + 20), (cx - 8, cy + 45), (40, 40, 40), 4)
            cv2.line(frame, (cx + 6, cy + 20), (cx + 8, cy + 45), (40, 40, 40), 4)

        # Add timestamp & simulated drone OSD text
        cv2.putText(
            frame,
            f"SENTINEL AERIAL SIM - FRAME: {f:04d} - ALT: 25.4m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
        )

        out.write(frame)

    out.release()
    return str(path)


if __name__ == "__main__":
    out_file = "data/sample_videos/demo_drone_crowd.mp4"
    generate_synthetic_crowd_video(out_file)
    print(f"Generated sample video at {out_file}")

