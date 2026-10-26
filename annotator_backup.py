
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "images"
ANNOTATION_DIR = PROJECT_DIR / "annotations"

MODEL_NAME = "yolo26n-pose.pt"
CONFIDENCE_THRESHOLD = 0.25

KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# COCO skeleton connections: (first keypoint, second keypoint)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# Visibility: 0 = not determinable,
#             1 = occluded but estimable,
#             2 = visible


class PoseAnnotator:
    def __init__(self, image_path, model):
        self.image_path = Path(image_path)
        self.model = model

        self.image = plt.imread(str(self.image_path))
        self.height, self.width = self.image.shape[:2]

        result = model.predict(
            source=str(self.image_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        self.people = []

        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()

            for person_points in keypoints:
                points = person_points[:, :2].astype(float)
                confidences = person_points[:, 2].astype(float)

                visibility = [
                    2 if confidence >= 0.5 else 0
                    for confidence in confidences
                ]

                self.people.append({
                    "keypoints": points,
                    "confidence": confidences,
                    "visibility": visibility,
                })

        self.person_index = 0
        self.selected_keypoint = 0

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )
        self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key
        )

        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image)

        if not self.people:
            self.ax.set_title(
                "No people detected. Close the window to exit."
            )
            self.fig.canvas.draw_idle()
            return

        person = self.people[self.person_index]
        points = person["keypoints"]
        visibility = person["visibility"]

        # Draw skeleton lines.
        for first, second in SKELETON:
            x1, y1 = points[first]
            x2, y2 = points[second]

            if visibility[first] != 0 and visibility[second] != 0:
                self.ax.plot(
                    [x1, x2], [y1, y2],
                    linewidth=2,
                )

        # Draw keypoints and labels.
        for index, (x, y) in enumerate(points):
            if visibility[index] == 0:
                marker = "x"
            elif visibility[index] == 1:
                marker = "s"
            else:
                marker = "o"

            size = 11 if index == self.selected_keypoint else 7

            self.ax.scatter(
                x, y,
                marker=marker,
                s=size ** 2,
            )

            self.ax.text(
                x + 4,
                y - 4,
                str(index),
                fontsize=8,
            )

        self.ax.set_title(
            f"{self.image_path.name} | "
            f"Person {self.person_index + 1}/{len(self.people)} | "
            f"Selected: {KEYPOINT_NAMES[self.selected_keypoint]} | "
            f"Visibility: {visibility[self.selected_keypoint]}\n"
            "Click near a point to select it; click elsewhere to move it. "
            "Keys: 0/1/2 visibility, N/P person, S save, Q quit"
        )

        self.ax.axis("off")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or not self.people:
            return

        if event.xdata is None or event.ydata is None:
            return

        person = self.people[self.person_index]
        points = person["keypoints"]

        click_x, click_y = event.xdata, event.ydata

        distances = np.sqrt(
            (points[:, 0] - click_x) ** 2
            + (points[:, 1] - click_y) ** 2
        )

        nearest = int(np.argmin(distances))

        # Click near an existing keypoint to select it.
        if distances[nearest] < 25:
            self.selected_keypoint = nearest
        else:
            # Otherwise, move the selected keypoint.
            self.people[self.person_index]["keypoints"][
                self.selected_keypoint
            ] = [click_x, click_y]

        self.redraw()

    def on_key(self, event):
        if not self.people:
            if event.key == "q":
                plt.close(self.fig)
            return

        if event.key in {"0", "1", "2"}:
            self.people[self.person_index]["visibility"][
                self.selected_keypoint
            ] = int(event.key)

        elif event.key == "n":
            self.person_index = (
                self.person_index + 1
            ) % len(self.people)

        elif event.key == "p":
            self.person_index = (
                self.person_index - 1
            ) % len(self.people)

        elif event.key == "s":
            self.save()

        elif event.key == "q":
            plt.close(self.fig)
            return

        self.redraw()

    def save(self):
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

        output_path = (
            ANNOTATION_DIR
            / f"{self.image_path.stem}_annotations.json"
        )

        annotations = {
            "image": self.image_path.name,
            "image_width": self.width,
            "image_height": self.height,
            "keypoint_format": "COCO-17",
            "people": [],
        }

        for person in self.people:
            annotations["people"].append({
                "keypoints": [
                    {
                        "name": KEYPOINT_NAMES[index],
                        "x": round(float(point[0]), 2),
                        "y": round(float(point[1]), 2),
                        "confidence": round(
                            float(person["confidence"][index]), 4
                        ),
                        "visibility": int(
                            person["visibility"][index]
                        ),
                    }
                    for index, point in enumerate(person["keypoints"])
                ]
            })

        output_path.write_text(
            json.dumps(annotations, indent=2),
            encoding="utf-8",
        )

        print(f"Annotations saved to: {output_path}")

    def show(self):
        plt.show()


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    )

    if not images:
        print(f"No images found in {IMAGE_DIR}")
        print("Add a photo to the images folder and run again.")
        return

    print("Available images:")

    for index, path in enumerate(images, start=1):
        print(f"{index}. {path.name}")

    choice = input(
        "\nSelect an image number: "
    ).strip()

    try:
        image_index = int(choice) - 1
        image_path = images[image_index]

        if not 0 <= image_index < len(images):
            raise ValueError

    except (ValueError, IndexError):
        print("Invalid image selection.")
        return

    print(f"\nLoading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    annotator = PoseAnnotator(image_path, model)

    if not annotator.people:
        print("No people detected in this image.")

    annotator.show()


if __name__ == "__main__":
    main()
