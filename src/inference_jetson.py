import cv2
from ultralytics import YOLO
from collections import defaultdict
import os

# --- Configuration ---
# IMPORTANT: If you change MODEL_PT_NAME, ensure the .engine file generated in Step 5
# of the README also reflects this base name (e.g., if MODEL_PT_NAME is "my_model.pt",
# the script expects "my_model.engine" to be in ../model_files/)
MODEL_PT_NAME = "my_yolov8_model.pt"  # CHANGE THIS to your actual .pt model filename

# Path to the an engine file. It assumes the .engine file is in ../model_files
# and has the same base name as your .pt file, but with a .engine extension.
ENGINE_FILENAME = MODEL_PT_NAME.replace('.pt', '.engine')
ENGINE_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_files', ENGINE_FILENAME)

WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 640 # Matched to common YOLO input size for consistency
CONFIDENCE_THRESHOLD = 0.5

# --- Classes to Count (MUST EXACTLY MATCH YOUR MODEL'S CLASS NAMES) ---
CLASSES_TO_COUNT = [
    "black", "defect", "long_screw", "nail",
    "nut", "rivet", "tek_screw", "washer"
]
# --- End Configuration ---

def main():
    print(f"Attempting to load engine: {ENGINE_PATH}")
    if not os.path.exists(ENGINE_PATH):
        print(f"ERROR: Engine file not found at {ENGINE_PATH}")
        print("Please ensure you have run Step 5 (Model Export) from the README.md")
        print(f"The script is looking for an engine file based on MODEL_PT_NAME: {MODEL_PT_NAME}")
        return

    # 1. Load the exported TensorRT model
    try:
        model = YOLO(ENGINE_PATH, task='detect') # Specify task for clarity
        print(f"TensorRT engine loaded successfully from {ENGINE_PATH}")
        print("Model class names from engine:", model.names)

        # Verification: Check if CLASSES_TO_COUNT are in model.names
        for cls_name_to_count in CLASSES_TO_COUNT:
            found = False
            # model.names is often a dict like {0: 'name1', 1: 'name2'}
            if isinstance(model.names, dict):
                if cls_name_to_count in model.names.values():
                    found = True
            elif isinstance(model.names, list): # Or a list
                 if cls_name_to_count in model.names:
                    found = True
            
            if not found:
                print(f"WARNING: Class '{cls_name_to_count}' in your CLASSES_TO_COUNT list "
                      f"is NOT FOUND in the model's class names. "
                      f"This class will not be counted unless the name matches exactly (case-sensitive).")
                print(f"Available model names: {list(model.names.values()) if isinstance(model.names, dict) else model.names}")


    except Exception as e:
        print(f"Error loading TensorRT model: {e}")
        print("This could be due to an incompatible TensorRT version if the engine was not built on this Jetson,")
        print("or if the engine file is corrupted or not found at the specified path.")
        return

    # 2. Initialize Webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {WEBCAM_INDEX}")
        print("Check if webcam is connected and /dev/video0 (or other index) is correct in docker run.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"Attempting to set webcam to {FRAME_WIDTH}x{FRAME_HEIGHT}")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual webcam resolution: {actual_width}x{actual_height}")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame from webcam.")
            break

        # 4. Perform Inference
        # verbose=False to reduce console spam, stream=True might be good for continuous video
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        object_counts_this_frame = defaultdict(int)
        display_total_count = 0

        # The 'results' object is a list, usually with one item for one image/frame
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                # Extract bounding box (xyxy format), confidence, and class ID
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Get class name using model.names
                class_name = ""
                if isinstance(model.names, dict) and cls_id in model.names:
                    class_name = model.names[cls_id]
                elif isinstance(model.names, list) and cls_id < len(model.names):
                    class_name = model.names[cls_id]
                else:
                    print(f"Warning: class_id {cls_id} not in model.names")
                    continue # Skip if class_id is out of bounds

                if class_name in CLASSES_TO_COUNT:
                    object_counts_this_frame[class_name] += 1
                    display_total_count += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare label text
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display Object Counts on Frame
        y_offset = 30
        for class_name_to_display in CLASSES_TO_COUNT:
            count = object_counts_this_frame.get(class_name_to_display, 0)
            cv2.putText(frame, f"{class_name_to_display}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        cv2.putText(frame, f"Total (Specified): {display_total_count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display Frame
        cv2.imshow("YOLO Jetson Nano Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")

if __name__ == "__main__":
    main()