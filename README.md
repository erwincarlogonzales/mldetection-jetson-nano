# YOLO Object Detection & Counting on Jetson Nano with Docker (Simple ONNX Method)

This project guides you to deploy a custom YOLO ONNX model on an NVIDIA Jetson Nano for real-time object detection and counting from a USB webcam. We're using Docker for a hassle-free setup with the Ultralytics framework.

**The core idea is to use your pre-exported `.onnx` model file directly for the quickest and easiest deployment.**

## Features

* Real-time object detection from a USB webcam using your ONNX model.
* Customizable object counting.
* Visualizations (bounding boxes, labels, counts) via `supervision`.
* Docker for easy environment setup.

## Project Structure

```
jetson_yolo_object_counter/
├── README.md                     # This guide!
├── src/
│   └── app.py                    # Python script for detection & counting using ONNX.
├── model_files/
│   └── your_model.onnx           # << YOUR ONNX MODEL GOES HERE (e.g., yolov11n.onnx)
├── .gitignore                    # Tells Git which files to ignore.
└── requirements.txt              # Python dependencies for the script.
```

## Prerequisites

1.  **NVIDIA Jetson Nano:** Flashed with a recent NVIDIA JetPack (e.g., JetPack 4.6.x or 5.x).
2.  **Docker Installed on Jetson Nano:** Check with `docker --version`. Install if needed via NVIDIA's docs.
3.  **NVIDIA Container Toolkit Installed on Jetson Nano:** Allows Docker to use the GPU. Usually part of JetPack.
4.  **USB Webcam:** Connected to your Jetson Nano.
5.  **Cloned This Repository:**
    ```bash
    git clone <your_github_repo_url>
    cd jetson_yolo_object_counter
    ```
6.  **Your `.onnx` model file** (e.g., `yolov11n.onnx`).

## Project Setup & Step-by-Step Instructions

Let's get your ONNX model running fast!

**Step 1: Prepare Your ONNX Model File**

1.  Navigate to the `model_files/` directory in this repository on your Jetson Nano.
2.  Place your `.onnx` model file here. For example, if your model is `yolov11n.onnx`, the path will be `model_files/yolov11n.onnx`.
3.  Open `src/app.py` and ensure the `MODEL_FILENAME` variable at the top correctly points to your ONNX file's name.
    ```python
    # src/app.py - Ensure this matches your ONNX file name
    MODEL_FILENAME = "yolov11n.onnx"
    ```

**Step 2: Choose Your JetPack Version for Docker Image**

Identify your Jetson Nano's JetPack version to select the correct Docker image tag.
* JetPack 4.6.x: Use `jetpack4` tag.
* JetPack 5.x: Use `jetpack5` tag.

We'll use `JETPACK_TAG=jetpack4` as an example. Modify if your version differs.

**Step 3: Pull the Ultralytics Docker Image**

Open a terminal on your Jetson Nano:
```bash
# Adjust 'jetpack4' if using a different JetPack version (e.g., jetpack5)
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4
```

**Step 4: Run the Docker Container**

This command starts the Docker container, mounts your project files, and grants access to the Jetson's GPU, webcam, and display.

```bash
# Ensure you are in the root directory of this cloned repository
JETPACK_TAG=jetpack4 # Or jetpack5, etc.

# Allow Docker to access your X server for displaying GUI windows
xhost +

sudo docker run -it --ipc=host --runtime nvidia \
    -v $(pwd)/src:/app/src \
    -v $(pwd)/model_files:/app/model_files \
    -v $(pwd)/requirements.txt:/app/requirements.txt \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ultralytics/ultralytics:latest-jetson-$JETPACK_TAG
```
* `-v $(pwd)/requirements.txt:/app/requirements.txt`: Mounts your `requirements.txt` file.

You should now be inside the Docker container's terminal. Navigate to your application directory:
```bash
cd /app
```

**Step 5: Inside Docker - Install Script Dependencies**

Your `app.py` uses the `supervision` library. Install it and other dependencies using the mounted `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Step 6: Inside Docker - Run the Inference Script!**

Make sure the `MODEL_FILENAME` in `/app/src/app.py` is correctly set.
```bash
cd /app/src
python3 app.py
```
A window should pop up showing your webcam feed with object detections and counts! Press 'q' in the window to quit.

## Usage

After the first-time setup:

1.  Start the Docker container (Step 4).
2.  Inside Docker, navigate to `/app`. If it's a brand new container instance and you didn't commit changes to an image, you might need to run `pip install -r requirements.txt` again.
3.  Navigate to `/app/src` and run `python3 app.py`.

## Customization (in `src/app.py`)

* **Model File:** Change `MODEL_FILENAME`.
* **Classes to Count:** Modify the `CLASSES_TO_COUNT` list.
* **Confidence Threshold:** Adjust `CONFIDENCE_THRESHOLD`.
* **Webcam Settings:** Modify `WEBCAM_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT`.

## Troubleshooting

* **`Error: Could not open webcam` / `Failed to grab frame`**:
    * Is the webcam connected and powered on?
    * Is `/dev/video0` (in `docker run`) the correct device for your camera? (Check host with `ls /dev/video*`).
    * Is `WEBCAM_INDEX` in `src/app.py` correct?
    * Is another program using the webcam? (Check on host with `sudo lsof /dev/video0`).
* **Display Errors (e.g., `cannot open display :0`)**:
    * Did you run `xhost +` on the Jetson host *before* the `docker run` command?
    * Are the display-related flags (`-e DISPLAY`, `-v /tmp/.X11-unix`) present in your `docker run` command?
* **`ModuleNotFoundError: No module named 'supervision'` (or other modules)**:
    * Did you run `pip install -r requirements.txt` inside the Docker container (Step 5)?
* **Incorrect Class Names or Counts:**
    * The class names in `CLASSES_TO_COUNT` in `src/app.py` *must exactly match* the names your ONNX model was trained with (these are usually accessible via `model.names` after the model loads; the script will print these for verification).

---

## Testing on Your PC (using Poetry)

You can test the `app.py` script on your local PC (Windows/Linux/macOS) to verify its functionality before deploying to the Jetson Nano. This setup uses Poetry for managing the Python environment and dependencies. Inference will typically run on your PC's CPU if you don't have an NVIDIA GPU with ONNX Runtime GPU support configured.

**Prerequisites for PC Testing:**

1.  **Python installed** (version compatible with `requires-python` in `pyproject.toml`, e.g., >=3.8).
2.  **Poetry installed** (see [Poetry's official documentation](https://python-poetry.org/docs/#installation)).
3.  **Git installed** (for cloning the repository).
4.  **A connected webcam.**

**Steps to Run `app.py` on Your PC:**

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <your_github_repo_url>
    cd jetson_yolo_object_counter
    ```

2.  **Initialize Poetry and Install Dependencies:**
    Navigate to the project root directory (`jetson_yolo_object_counter`). If this is your first time setting up with Poetry for this project:
    ```bash
    # This installs dependencies from pyproject.toml and poetry.lock
    poetry install
    ```
    *(If you've been adding dependencies one-by-one with `poetry add ...` and have a `poetry.lock` file, `poetry install` will ensure everything is synced).*

3.  **Activate the Poetry Virtual Environment:**
    ```bash
    poetry shell
    ```
    Your terminal prompt should change, indicating the virtual environment is active.

4.  **Prepare Your ONNX Model:**
    * Place your `.onnx` model file (e.g., `best_int8_dynamic.onnx`) into the `model_files/` directory.
    * Open `src/app.py` and ensure the `MODEL_FILENAME` variable at the top correctly matches the name of your ONNX file.
    * Also, verify that the `CLASSES_TO_COUNT` list in `src/app.py` contains the class names relevant to your model.

5.  **Navigate to the Source Directory:**
    ```bash
    # If you ran 'env activate' from the project root, you're likely still there.
    # If so, navigate into src:
    cd src
    ```

6.  **Run the Script:**
    ```bash
    python app.py
    ```
    *(Since the Poetry shell is active, you can use `python app.py` directly. Alternatively, from the project root, you could use `poetry run python src/app.py`)*

7.  **View Output:**
    * A window should appear showing your webcam feed.
    * Detected objects (from your `CLASSES_TO_COUNT` list) should have bounding boxes and labels.
    * Counts for these classes and the inference speed (in milliseconds) will be displayed on the frame.
    * Check your terminal for any print messages from the script (model loading status, class names, camera resolution).

8.  **To Stop:** Press 'q' in the OpenCV window. To deactivate the Poetry shell later, you can type `exit`.

This setup allows you to debug and refine the core application logic on your PC before moving to the Jetson Nano Docker deployment.