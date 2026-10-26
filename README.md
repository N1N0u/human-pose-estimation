# 🧍 Human Pose Estimation & Annotation Tool

> An interactive computer vision pipeline combining AI-powered human pose estimation with manual keypoint correction and visibility annotation.

---

## 📌 Overview

**Human Pose Estimation & Annotation Tool** is a Python-based computer vision project designed to detect human body keypoints, visualize skeletal structures, and refine AI-generated predictions through an interactive annotation interface.

Built with **YOLO Pose and Python**, the system provides a practical workflow for processing images, reviewing pose predictions, correcting keypoint coordinates, and exporting structured annotations in JSON format.

Unlike basic pose estimation demos, this project focuses on **annotation workflows, data quality, and practical computer vision applications**.

---

## 🚀 Highlights

- ⚡ Automated human pose estimation using YOLO Pose
- 🧍 Detection and visualization of human body keypoints
- 🎯 Interactive keypoint selection and position correction
- 👁️ Visibility annotation for individual keypoints
- 👥 Support for multiple people in an image
- 🔄 Navigation across image datasets
- 💾 Automatic loading and saving of annotation files
- 📄 Structured JSON annotation export
- ⌨️ Keyboard shortcuts for efficient annotation
- 🖥️ Local execution without requiring a hosted service

---

## 🔄 System Pipeline

1. **Image Input** — Load images from a local dataset.
2. **Pose Estimation** — Detect human body keypoints using a pretrained YOLO Pose model.
3. **Pose Visualization** — Display keypoints and skeletal connections.
4. **Manual Correction** — Adjust inaccurate keypoint positions.
5. **Visibility Annotation** — Assign visibility labels to each keypoint.
6. **Annotation Export** — Save corrected annotations in JSON format.

---

## 🖼️ Application Workflow

### 📸 Step 1: Image Input

Load images from the local `images/` directory.

The application supports common image formats, including JPEG, PNG, BMP, and WebP.

### 🧠 Step 2: AI Pose Estimation

A pretrained YOLO Pose model generates initial body keypoint predictions.

The predicted keypoints and skeletal connections are displayed for manual review.

### 🎯 Step 3: Interactive Keypoint Correction

Review the detected pose and drag keypoints to correct their positions.

This helps refine model predictions when body parts are difficult to detect accurately.

### 👁️ Step 4: Visibility Annotation

Assign a visibility label to each keypoint:

- `0` — Not determinable
- `1` — Occluded but estimable
- `2` — Visible

Visibility labels are assigned manually during annotation.

### 💾 Step 5: Save Annotations

Save the annotations as JSON files for later editing and further dataset processing.

> Add screenshots or GIF recordings of your actual application to demonstrate these steps.

---

## ⚙️ Interactive Annotation System

The application combines AI-generated predictions with human review.

- Automatically generates initial pose predictions when no saved annotations exist.
- Loads existing annotations to continue previous work.
- Allows manual correction of keypoint coordinates.
- Supports annotation of multiple detected people.
- Saves annotation progress in JSON files.
- Provides keyboard shortcuts for navigation and visibility labeling.

---

## 📂 Project Structure

```text
human-pose-estimation/
│
├── images/
│   ├── 1.jpeg
│   ├── 2.jpeg
│   └── ...
│
├── annotations/
│   ├── 1_annotations.json
│   ├── 2_annotations.json
│   └── ...
│
├── main.py
├── annotator.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Directory overview:**

- `images/` — Input images for pose estimation and annotation.
- `annotations/` — Saved keypoint coordinates and visibility labels.
- `main.py` — Batch pose estimation script.
- `annotator.py` — Interactive pose annotation interface.
- `requirements.txt` — Python dependencies.

---

## ▶️ Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/N1N0u/human-pose-estimation/
cd human-pose-estimation
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

On Ubuntu, install Tkinter if needed:

```bash
sudo apt install python3-tk -y
```

### 4. Launch the annotation tool

```bash
python annotator.py
```

The application loads the images, generates initial pose predictions when necessary, and opens an interactive window for reviewing and correcting keypoints.

---

## ⌨️ Keyboard Controls

| Key | Action |
|---|---|
| `D` / `Right Arrow` | Next image |
| `A` / `Left Arrow` | Previous image |
| `N` | Next person |
| `P` | Previous person |
| `Up Arrow` | Select next keypoint |
| `Down Arrow` | Select previous keypoint |
| `0` | Mark keypoint as not determinable |
| `1` | Mark keypoint as occluded but estimable |
| `2` | Mark keypoint as visible |
| `S` | Save annotations |
| `Q` | Save and quit |

Keypoints can also be selected and repositioned interactively with the mouse.

---

## 🔬 Technical Details

- **Pose Estimation:** Ultralytics YOLO Pose
- **Deep Learning Framework:** PyTorch, through Ultralytics
- **Image Processing:** NumPy and image-reading utilities
- **Visualization:** Matplotlib
- **Annotation Format:** JSON
- **Keypoint Schema:** COCO-17
- **Interaction:** Mouse events and keyboard shortcuts

### Keypoint Representation

The current implementation uses 17 standard COCO body keypoints:

- Nose
- Left and right eyes
- Left and right ears
- Left and right shoulders
- Left and right elbows
- Left and right wrists
- Left and right hips
- Left and right knees
- Left and right ankles

Each person is represented by 17 keypoint coordinates and 17 corresponding visibility labels.

---

## ⚡ Design Goals

- **Practical Computer Vision:** Combine automated inference with manual correction.
- **Annotation Quality:** Allow users to refine inaccurate keypoint predictions.
- **Efficient Workflow:** Simplify reviewing multiple images and people.
- **Persistent Data:** Save and reload annotations in structured JSON files.
- **Local Execution:** Run the tool on a personal computer without requiring a cloud service.

---

## 🎯 Use Cases

- Human pose estimation experiments
- Sports image annotation
- Computer vision dataset preparation
- Manual correction of AI-generated keypoints
- Human movement analysis research
- Training-data review and quality control

---

## 🔮 Future Improvements

- Support custom keypoint schemas, including 21-keypoint datasets.
- Add dataset-specific annotation export formats.
- Improve annotation validation and consistency checks.
- Add annotation progress statistics.
- Introduce confidence-based review and quality-control tools.
- Integrate with external annotation platforms.

---

## 👤 Author

**ATEF Aliat**

Machine Learning & Computer Vision Enthusiast

- Focus: Computer Vision, AI Inference, and Practical Machine Learning Systems
- Interested in real-time AI applications and dataset preparation
- Building hands-on projects focused on applied AI and deployment

---

## ⭐ Final Statement

This project demonstrates the integration of:

- AI-powered human pose estimation
- Interactive keypoint correction
- Visibility labeling and annotation workflows
- Structured dataset generation
- Practical Python computer vision development

> **Not just pose detection — an interactive workflow for reviewing, correcting, and annotating AI-generated human poses.**
