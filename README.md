# Project Structure
```
jetson_yolo_object_counter/
├── README.md                               # You're reading the future! Step-by-step guide.
├── src/
│   └── inference_jetson.py                 # Our main Python script for detection and counting.
├── model_files/                            # You'll put your model and data files here.
│   ├── PUTC_YOUR_YOLOv8_PT_MODEL_HERE.txt  # Placeholder: e.g., best_yolov8n.pt
│   ├── PUTC_YOUR_DATA_YAML_HERE.txt        # Placeholder: e.g., dataset.yaml for INT8 calibration
│   └── PUTC_YOUR_CALIBRATION_IMAGES_HERE/  # Placeholder: And the images folder (e.g., 'dataset/images/')
│       └── (images_go_here_as_per_data_yaml).txt
├── .gitignore                              # Tells Git which files to ignore.
└── requirements.txt                        # Python dependencies (minimal, Docker handles most).
```
---

# YOLO Object Detection & Counting on Jetson Nano with Docker

This project enables you to deploy a custom YOLO (You Only Look Once) model on an NVIDIA Jetson Nano for real-time object detection and counting using a USB webcam. It leverages Docker for a clean and reproducible environment and uses the Ultralytics framework.

**The core idea is to take your trained PyTorch model (`.pt` file) and convert it to a TensorRT engine *directly on the Jetson Nano* to ensure compatibility and optimal performance.**

## Features

* Real-time object detection from USB webcam.
* Object counting for specified classes.
* Visualization of bounding boxes, class labels, confidence scores, and counts.
* Uses Docker for easy dependency management.
* Optimized for Jetson Nano.

## Prerequisites

1.  **NVIDIA Jetson Nano:** Flashed with a recent NVIDIA JetPack (e.g., JetPack 4.6.x or JetPack 5.x).
2.  **Docker Installed on Jetson Nano:**
    * Check with `docker --version`.
    * If not installed, follow NVIDIA's official documentation to install Docker.
3.  **NVIDIA Container Toolkit Installed on Jetson Nano:**
    * This allows Docker containers to access the Jetson's GPU.
    * Often installed with JetPack if Docker is selected. Verify its presence.
4.  **USB Webcam:** Connected to the Jetson Nano.
5.  **Cloned This Repository:**
    ```bash
    git clone <your_github_repo_url>
    cd jetson_yolo_object_counter
    ```
6.  **(For INT8 Export) Your trained YOLOv8 `.pt` model file, your `data.yaml` file used for training/calibration, and the corresponding calibration images.**
7.  **(For FP16 Export) Your trained YOLO `.pt` model file.**

## Project Setup & Step-by-Step Instructions

This is like preparing your summoning jutsu – follow each step carefully!

**Step 1: Prepare Your Model and Data Files**

1.  Navigate to the `model_files/` directory in this repository on your Jetson Nano.
2.  **Place your trained YOLO `.pt` model file** here. For example, if your model is `best.pt`, place it in `model_files/best.pt`.
3.  **If you are planning to export to INT8 (recommended for YOLOv8 on Nano for performance):**
    * Place your `data.yaml` file (the one that defines your dataset, classes, and paths to images, used during training/calibration) in `model_files/data.yaml`.
    * Place your calibration image dataset (e.g., a `dataset/` folder containing `images/` and `labels/` subdirectories, as referenced by your `data.yaml`) inside the `model_files/` directory. Ensure the paths within your `data.yaml` are relative to its location or update them to be correct *when accessed from inside the Docker container* (e.g., if `data.yaml` expects `../dataset/images`, make sure that structure exists relative to where `data.yaml` will be in the container, like `/app/model_files/`).
        * **Example structure inside `model_files/` for INT8:**
            ```
            model_files/
            ├── my_yolov8_model.pt
            ├── my_dataset_config.yaml
            └── hardware_dataset/          # Or whatever your dataset folder is named
                ├── images/
                │   ├── train/
                │   └── valid/
                └── labels/
                    ├── train/
                    └── valid/
            ```
            And `data.yaml` might have paths like:
            ```yaml
            path: ../hardware_dataset  # Relative path to the dataset root from data.yaml
            train: images/train
            val: images/valid
            names:
              0: black
              1: defect
              # ... your other classes
            ```

**Step 2: Choose Your JetPack Version for Docker Image**

Ultralytics provides Docker images tagged for different JetPack versions. Identify your JetPack version.
* Common for Jetson Nano: JetPack 4.6.x (use `jetpack4` tag)
* Newer Jetson Nano setups: JetPack 5.x (use `jetpack5` tag)

Let's assume `JETPACK_TAG=jetpack4` for these instructions. Adjust if needed.

**Step 3: Pull the Ultralytics Docker Image**

Open a terminal on your Jetson Nano and pull the image:

```bash
# Adjust jetpack4 to jetpack5 or jetpack6 if needed
# For JetPack 4.x:
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4

# For JetPack 5.x:
# sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack5
```

**Step 4: Run the Docker Container**

This command starts the Docker container, mounts your project directory, and gives the container access to the Jetson's GPU, webcam, and display.

```bash
# Make sure you are in the root of this cloned repository (jetson_yolo_object_counter/)
# Adjust JETPACK_TAG if you used a different one above.
JETPACK_TAG=jetpack4 # Or jetpack5, etc.

# Allow Docker to access your X server for GUI display from the container
xhost +

sudo docker run -it --ipc=host --runtime nvidia \
    -v $(pwd)/src:/app/src \
    -v $(pwd)/model_files:/app/model_files \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ultralytics/ultralytics:latest-jetson-$JETPACK_TAG
```

* `-it`: Interactive terminal.
* `--ipc=host`: Shares host's inter-process communication.
* `--runtime nvidia`: Enables GPU access.
* `-v $(pwd)/src:/app/src`: Mounts your `src` directory into `/app/src` in the container.
* `-v $(pwd)/model_files:/app/model_files`: Mounts your `model_files` directory into `/app/model_files`. **This is crucial for accessing your `.pt` file and calibration data.**
* `--device /dev/video0`: Gives access to the first USB webcam. If your webcam is different (e.g., `/dev/video1`), change this.
* `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`: Allows GUI windows (like OpenCV's `imshow`) from the container to appear on your Jetson's display.

You should now be inside the Docker container's terminal, likely in the `/` directory. Navigate to your app directory:
```bash
cd /app
```

**Step 5: Inside Docker - Export Your Model to TensorRT Engine**

This is where you convert your `.pt` model to a Jetson-compatible TensorRT engine.

* **Navigate to where your model files are:**
    ```bash
    cd /app/model_files
    ```

* **Command for YOLOv8n INT8 Export (Recommended):**
    Replace `my_yolov8_model.pt` with your actual `.pt` filename and `my_dataset_config.yaml` with your actual `data.yaml` filename.
    ```bash
    yolo export \
        model=my_yolov8_model.pt \
        format=engine \
        imgsz=640 \
        int8=True \
        data=my_dataset_config.yaml \
        device=0 \
        workspace=4 # Optional: GPU workspace in GB, adjust if needed for Nano
    ```
    This will create `my_yolov8_model.engine` (or similar) in the `/app/model_files/` directory (inside the container, which is your `model_files` on the host).

* **Alternative: Command for FP16 Export (e.g., for YOLOv11n or if you don't want INT8):**
    Replace `my_yolo_model.pt` with your actual `.pt` filename.
    ```bash
    yolo export \
        model=my_yolo_model.pt \
        format=engine \
        imgsz=640 \
        half=True \
        device=0 \
        workspace=4 # Optional
    ```
    This will create `my_yolo_model.engine` in FP16.

**Wait for the export to complete. It might take several minutes.** Once done, you'll have a `.engine` file in your `model_files/` directory.

**Step 6: Inside Docker - Run the Inference Script!**

1.  Now, make sure the `inference_jetson.py` script knows which engine file to use. The script is configured to look for an engine file based on the `.pt` file name you specify (or you can hardcode the path).
    *Open `/app/src/inference_jetson.py` using a text editor available in the container (like `nano` or `vi`), or edit it on your host Jetson system in `src/inference_jetson.py` (changes will reflect in the container due to the volume mount).*
    *Ensure the `MODEL_PT_NAME` variable in the script matches the base name of your `.pt` file, or update `ENGINE_PATH` directly.*

2.  Run the script:
    ```bash
    cd /app/src
    python3 inference_jetson.py
    ```

You should see a window pop up showing your webcam feed with objects detected, bounding boxes, and counts for your specified classes! Press 'q' to quit.

## Usage

Once the initial setup and model export are done, to run the application again:

1.  Start the Docker container (Step 4 above).
2.  Navigate to `/app/src` inside the container.
3.  Run `python3 inference_jetson.py`. (The `.engine` file is already built and will be reused).

## Customization

* **Classes to Count:** Modify the `CLASSES_TO_COUNT` list in `src/inference_jetson.py`.
* **Confidence Threshold:** Adjust `CONFIDENCE_THRESHOLD` in the script.
* **Webcam Index/Resolution:** Change `WEBCAM_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT` in the script.
* **Model Path:** Update `MODEL_PT_NAME` or `ENGINE_PATH` in the script if you change your model file.

## Troubleshooting

* **`Error: Could not open webcam`**:
    * Ensure your webcam is connected.
    * Verify the correct device ID (e.g., `/dev/video0` or `/dev/video1`) in the `docker run` command and `WEBCAM_INDEX` in the script.
    * Try simpler OpenCV webcam test scripts to isolate the issue.
* **Display Errors (e.g., `cannot open display :0`)**:
    * Ensure you ran `xhost +` on the Jetson host *before* `docker run`.
    * Verify the `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` flags in your `docker run` command.
* **TensorRT Engine Creation Fails**:
    * Check for sufficient disk space and memory on the Jetson Nano.
    * Ensure the `workspace` parameter in `yolo export` is reasonable (e.g., 2 or 4 GB for Nano).
    * Verify your `.pt` model and calibration data (for INT8) are correct and accessible.
    * Check the Jetson's system logs (`dmesg`) for any hardware-related errors.
* **Incorrect Class Names or No Detections for Specific Classes:**
    * The class names in `CLASSES_TO_COUNT` in the Python script *must exactly match* the class names your model was trained with (and as present in `model.names` after loading). The script prints `model.names` on load; verify them.
