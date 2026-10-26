import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_NAME = "yolo26n-pose.pt"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25


# COCO 17 keypoint names
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


def find_default_image():
    """Find the first supported image in the images directory."""

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not IMAGE_DIR.exists():
        return None

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    return images[0] if images else None


def process_image(image_path, model):
    """Detect people and estimate their body keypoints."""

    print("\n" + "=" * 55)
    print("HUMAN POSE ESTIMATION")
    print("=" * 55)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    # Load the original image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    height, width = image.shape[:2]

    print(f"Input image: {image_path.name}")
    print(f"Image resolution: {width} x {height}")

    # Run pose estimation
    start_time = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    elapsed_time = time.perf_counter() - start_time

    result = results[0]

    # Draw bounding boxes, keypoints, and skeletons
    annotated_image = result.plot(
        boxes=True,
        labels=True,
        conf=True,
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save annotated image
    output_path = OUTPUT_DIR / f"{image_path.stem}_pose.jpg"

    success = cv2.imwrite(str(output_path), annotated_image)

    if not success:
        print("ERROR: Could not save the output image.")
        return

    # Count detected people
    person_count = (
        len(result.keypoints)
        if result.keypoints is not None
        else 0
    )

    print("\nRESULTS")
    print("-" * 55)
    print(f"People detected: {person_count}")
    print(f"Total inference time: {elapsed_time:.3f} seconds")

    if elapsed_time > 0:
        print(f"Processing speed: {1 / elapsed_time:.2f} images/second")

    # Print keypoint information for every detected person
    if result.keypoints is not None:
        for person_index, keypoints in enumerate(
            result.keypoints.data
        ):
            print(f"\nPerson {person_index + 1}")
            print("-" * 35)

            points = keypoints.cpu().numpy()

            for index, point in enumerate(points):
                x = float(point[0])
                y = float(point[1])

                # Depending on the model output, the third value
                # may be a confidence score rather than a human-
                # annotated visibility label.
                confidence = (
                    float(point[2])
                    if len(point) > 2
                    else None
                )

                if confidence is not None:
                    print(
                        f"{KEYPOINT_NAMES[index]:<16} "
                        f"x={x:7.1f}, y={y:7.1f}, "
                        f"confidence={confidence:.3f}"
                    )
                else:
                    print(
                        f"{KEYPOINT_NAMES[index]:<16} "
                        f"x={x:7.1f}, y={y:7.1f}"
                    )

    print("\n" + "=" * 55)
    print("Processing completed successfully!")
    print(f"Annotated image saved to:\n{output_path}")
    print("=" * 55)

    # Display the result in a window
    cv2.imshow("Human Pose Estimation", annotated_image)

    print("\nPress any key while the image window is active to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Human Pose Estimation using YOLO and OpenCV"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the input image",
    )

    args = parser.parse_args()

    # Select input image
    if args.image:
        image_path = Path(args.image).expanduser().resolve()
    else:
        image_path = find_default_image()

    if image_path is None:
        print("No image found in the images directory.")
        print(f"Please add an image to: {IMAGE_DIR}")
        print("Alternatively, use: python main.py --image /path/to/image.jpg")
        return

    # Load pretrained model
    print(f"Loading model: {MODEL_NAME}")
    print("The model will download automatically if necessary.")

    model = YOLO(MODEL_NAME)

    process_image(image_path, model)


if __name__ == "__main__":
    main()