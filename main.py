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

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_images():
    """Find supported images in the input directory."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_FORMATS
    )


# ============================================================
# POSE ESTIMATION
# ============================================================

def process_image(image_path, model):
    """Run pose estimation and save the annotated image."""

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Cannot read image: {image_path.name}")
        return None

    start_time = time.perf_counter()

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    inference_time = time.perf_counter() - start_time

    result = results[0]

    # Draw bounding boxes, keypoints, and skeletons.
    annotated_image = result.plot(
        boxes=True,
        labels=True,
        conf=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{image_path.stem}_pose.jpg"

    if not cv2.imwrite(str(output_path), annotated_image):
        print(f"[ERROR] Could not save: {output_path.name}")
        return None

    person_count = (
        0 if result.keypoints is None
        else len(result.keypoints)
    )

    print(f"\nImage: {image_path.name}")
    print(f"People detected: {person_count}")
    print(f"Inference time: {inference_time * 1000:.2f} ms")
    print(f"Saved to: {output_path}")

    return {
        "image": image_path.name,
        "people": person_count,
        "inference_time": inference_time,
    }


# ============================================================
# BENCHMARK
# ============================================================

def print_benchmark(results, total_processing_time):
    """Display aggregate performance measurements."""

    if not results:
        print("\nNo images were processed successfully.")
        return

    successful_images = len(results)

    total_inference_time = sum(
        item["inference_time"] for item in results
    )

    average_inference_time = (
        total_inference_time / successful_images
    )

    total_people = sum(
        item["people"] for item in results
    )

    throughput = (
        successful_images / total_processing_time
        if total_processing_time > 0 else 0
    )

    print("\n" + "=" * 55)
    print("PERFORMANCE BENCHMARK")
    print("=" * 55)

    print(f"Images processed:       {successful_images}")
    print(f"People detected:        {total_people}")
    print(f"Total inference time:   {total_inference_time:.3f} s")
    print(
        f"Average inference:      "
        f"{average_inference_time * 1000:.2f} ms/image"
    )
    print(f"Total processing time:  {total_processing_time:.3f} s")
    print(f"Processing throughput:  {throughput:.2f} images/s")

    print("=" * 55)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 55)
    print("HUMAN POSE ESTIMATION")
    print("=" * 55)

    images = get_images()

    if not images:
        print(f"No images found in: {IMAGE_DIR}")
        print("Add some photos to the images directory.")
        return

    print(f"Images found: {len(images)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    print("\nLoading model...")

    model = YOLO(MODEL_NAME)

    results = []

    # Measure the complete processing loop, excluding model loading.
    processing_start = time.perf_counter()

    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] Processing...")

        result = process_image(image_path, model)

        if result is not None:
            results.append(result)

    total_processing_time = (
        time.perf_counter() - processing_start
    )

    print_benchmark(results, total_processing_time)

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()