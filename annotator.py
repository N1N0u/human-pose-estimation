import json
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
ANNOTATION_DIR = BASE_DIR / "annotations"

MODEL_NAME = "yolo26n-pose.pt"
CONFIDENCE_THRESHOLD = 0.25

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# COCO-17 skeleton connections
SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ============================================================
# ANNOTATOR
# ============================================================

class PoseAnnotator:

    def __init__(self):
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

        self.images = sorted(
            path for path in IMAGE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_FORMATS
        )

        if not self.images:
            raise RuntimeError(
                f"No images found in: {IMAGE_DIR}\n"
                "Add JPG, PNG, or other supported photos."
            )

        print("=" * 55)
        print("HUMAN POSE ANNOTATOR")
        print("=" * 55)
        print(f"Images found: {len(self.images)}")
        print(f"Annotations: {ANNOTATION_DIR}")

        print(f"\nLoading model: {MODEL_NAME}")
        self.model = YOLO(MODEL_NAME)

        self.image_index = 0
        self.person_index = 0
        self.selected_keypoint = 0

        self.people = []
        self.image = None
        self.image_width = 0
        self.image_height = 0

        # Clicking and dragging moves a keypoint.
        self.dragging = False
        self.drag_person = None
        self.drag_keypoint = None

        self.fig, self.ax = plt.subplots(figsize=(14, 9))

        self.fig.canvas.mpl_connect(
            "button_press_event", self.on_mouse_press
        )
        self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_mouse_move
        )
        self.fig.canvas.mpl_connect(
            "button_release_event", self.on_mouse_release
        )
        self.fig.canvas.mpl_connect(
            "key_press_event", self.on_key
        )

        self.load_image()

    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    def annotation_path(self):
        image_path = self.images[self.image_index]

        return (
            ANNOTATION_DIR
            / f"{image_path.stem}_annotations.json"
        )

    # --------------------------------------------------------
    # LOAD IMAGE AND ANNOTATIONS
    # --------------------------------------------------------

    def load_image(self):
        image_path = self.images[self.image_index]

        self.image = plt.imread(str(image_path))
        self.image_height, self.image_width = self.image.shape[:2]

        self.person_index = 0
        self.selected_keypoint = 0
        self.dragging = False

        saved_file = self.annotation_path()

        if saved_file.exists():
            try:
                with open(saved_file, "r", encoding="utf-8") as file:
                    data = json.load(file)

                self.people = []

                for person in data.get("people", []):
                    points = None
                    visibility = None

                    # Current format:
                    # {"points": [[x, y], ...],
                    #  "visibility": [...]}
                    if "points" in person:
                        points = np.asarray(
                            person["points"],
                            dtype=float,
                        )

                        visibility = person.get(
                            "visibility",
                            [None] * 17,
                        )

                    # Previous format:
                    # {"keypoints": [
                    #   {"name": ..., "x": ..., "y": ...,
                    #    "visibility": ...}
                    # ]}
                    elif "keypoints" in person:
                        keypoints = person["keypoints"]

                        try:
                            points = np.asarray(
                                [
                                    [kp["x"], kp["y"]]
                                    for kp in keypoints
                                ],
                                dtype=float,
                            )

                            visibility = [
                                kp.get("visibility")
                                for kp in keypoints
                            ]

                        except (KeyError, TypeError):
                            print(
                                "Warning: invalid keypoint "
                                f"data in {saved_file.name}"
                            )
                            continue

                    if points is None or points.shape != (17, 2):
                        print(
                            "Warning: skipping an invalid person "
                            f"annotation in {saved_file.name}"
                        )
                        continue

                    if not isinstance(visibility, list):
                        visibility = [None] * 17

                    if len(visibility) != 17:
                        visibility = [None] * 17

                    # Only accept supported visibility labels.
                    visibility = [
                        value if value in (0, 1, 2) else None
                        for value in visibility
                    ]

                    self.people.append({
                        "points": points,
                        "visibility": visibility,
                    })

                print(f"\nLoaded saved annotations: {saved_file.name}")
                print(f"People in annotation file: {len(self.people)}")

            except (OSError, json.JSONDecodeError) as error:
                print(f"Could not read annotations: {error}")
                print("Generating fresh model predictions.")
                self.detect_people(image_path)

        else:
            self.detect_people(image_path)

        self.redraw()

    def detect_people(self, image_path):
        """Generate initial pose predictions using YOLO Pose."""

        result = self.model.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        self.people = []

        if result.keypoints is not None:
            for person in result.keypoints.data.cpu().numpy():
                points = person[:, :2].astype(float)

                # Visibility must be checked manually.
                visibility = [None] * 17

                self.people.append({
                    "points": points,
                    "visibility": visibility,
                })

        print(
            f"\n{image_path.name}: "
            f"{len(self.people)} people detected."
        )

    # --------------------------------------------------------
    # DRAW INTERFACE
    # --------------------------------------------------------

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image)

        if not self.people:
            self.ax.set_title(
                f"Image {self.image_index + 1}/{len(self.images)}\n"
                "No people detected | D: next image | "
                "A: previous image | Q: quit"
            )

            self.ax.axis("off")
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            return

        self.person_index = min(
            self.person_index,
            len(self.people) - 1,
        )

        person = self.people[self.person_index]
        points = person["points"]
        visibility = person["visibility"]

        # Draw skeleton lines.
        for first, second in SKELETON:
            if (
                visibility[first] == 0
                or visibility[second] == 0
            ):
                continue

            x1, y1 = points[first]
            x2, y2 = points[second]

            self.ax.plot(
                [x1, x2],
                [y1, y2],
                color="cyan",
                linewidth=1.8,
                zorder=1,
            )

        # Draw keypoints.
        for index, (x, y) in enumerate(points):
            state = visibility[index]

            if state == 0:
                color = "red"
                marker = "x"
            elif state == 1:
                color = "orange"
                marker = "s"
            elif state == 2:
                color = "lime"
                marker = "o"
            else:
                color = "yellow"
                marker = "o"

            selected = index == self.selected_keypoint
            size = 120 if selected else 55

            self.ax.scatter(
                x,
                y,
                color=color,
                marker=marker,
                s=size,
                edgecolors="black" if marker != "x" else "none",
                linewidths=0.8,
                zorder=3,
            )

            self.ax.text(
                x + 5,
                y - 5,
                str(index),
                fontsize=9,
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.65,
                    "pad": 1,
                },
                zorder=4,
            )

        reviewed = sum(
            value in (0, 1, 2)
            for value in visibility
        )

        selected_visibility = visibility[self.selected_keypoint]

        if selected_visibility is None:
            visibility_text = "Unreviewed"
        else:
            visibility_text = str(selected_visibility)

        self.ax.set_title(
            f"Image {self.image_index + 1}/{len(self.images)}: "
            f"{self.images[self.image_index].name}\n"
            f"Person {self.person_index + 1}/{len(self.people)} | "
            f"Reviewed: {reviewed}/17 | "
            f"Selected: {KEYPOINT_NAMES[self.selected_keypoint]} "
            f"(ID {self.selected_keypoint}) | "
            f"Visibility: {visibility_text}\n"
            "Drag point: move | Click point: select | "
            "0/1/2: visibility | N/P: person | "
            "D/A: image | Arrows: select point | S: save | Q: quit"
        )

        self.ax.axis("off")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # MOUSE INTERACTION
    # --------------------------------------------------------

    def on_mouse_press(self, event):
        if event.inaxes != self.ax or not self.people:
            return

        if event.xdata is None or event.ydata is None:
            return

        if event.button != 1:
            return

        person = self.people[self.person_index]
        points = person["points"]

        clicked = np.array([event.xdata, event.ydata])

        distances = np.linalg.norm(points - clicked, axis=1)
        nearest = int(np.argmin(distances))

        # Select a point only if the click is sufficiently close.
        if distances[nearest] < 25:
            self.selected_keypoint = nearest
            self.dragging = True
            self.drag_person = self.person_index
            self.drag_keypoint = nearest
            self.redraw()

    def on_mouse_move(self, event):
        if not self.dragging or not self.people:
            return

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        person = self.people[self.drag_person]
        point_index = self.drag_keypoint

        # Keep the point inside the image boundaries.
        x = float(np.clip(event.xdata, 0, self.image_width - 1))
        y = float(np.clip(event.ydata, 0, self.image_height - 1))

        person["points"][point_index] = [x, y]

        self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        if self.dragging:
            self.dragging = False
            self.drag_person = None
            self.drag_keypoint = None
            self.redraw()

    # --------------------------------------------------------
    # SAVE ANNOTATIONS
    # --------------------------------------------------------

    def save(self):
        image_path = self.images[self.image_index]

        data = {
            "image": image_path.name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "format": "COCO-17",
            "keypoint_names": KEYPOINT_NAMES,
            "people": [],
        }

        for person in self.people:
            data["people"].append({
                "points": [
                    [float(x), float(y)]
                    for x, y in person["points"]
                ],
                "visibility": person["visibility"],
            })

        output_path = self.annotation_path()

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        print(f"Saved annotations: {output_path}")

    # --------------------------------------------------------
    # IMAGE NAVIGATION
    # --------------------------------------------------------

    def change_image(self, direction):
        new_index = self.image_index + direction

        if not 0 <= new_index < len(self.images):
            print("No more images in that direction.")
            return

        self.save()
        self.image_index = new_index
        self.load_image()

    # --------------------------------------------------------
    # KEYBOARD CONTROLS
    # --------------------------------------------------------

    def on_key(self, event):
        key = event.key

        # Image navigation
        if key in ("d", "right"):
            self.change_image(1)
            return

        if key in ("a", "left"):
            self.change_image(-1)
            return

        # Save and quit
        if key == "s":
            self.save()
            return

        if key == "q":
            self.save()
            plt.close(self.fig)
            return

        # No person-specific controls when no people are detected
        if not self.people:
            return

        # Person navigation
        if key == "n":
            self.person_index = (
                self.person_index + 1
            ) % len(self.people)
            self.selected_keypoint = 0

        elif key == "p":
            self.person_index = (
                self.person_index - 1
            ) % len(self.people)
            self.selected_keypoint = 0

        # Visibility labels
        elif key in ("0", "1", "2"):
            self.people[self.person_index]["visibility"][
                self.selected_keypoint
            ] = int(key)

        # Keypoint navigation
        elif key in ("up",):
            self.selected_keypoint = (
                self.selected_keypoint + 1
            ) % len(KEYPOINT_NAMES)

        elif key in ("down",):
            self.selected_keypoint = (
                self.selected_keypoint - 1
            ) % len(KEYPOINT_NAMES)

        self.redraw()

    # --------------------------------------------------------
    # START APPLICATION
    # --------------------------------------------------------

    def run(self):
        plt.show()


if __name__ == "__main__":
    try:
        annotator = PoseAnnotator()
        annotator.run()

    except KeyboardInterrupt:
        print("\nAnnotation cancelled.")

    except Exception as error:
        print(f"\nError: {error}")
        raise
