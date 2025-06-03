import cv2
import supervision as sv
from ultralytics import YOLO
import os
from collections import defaultdict

# --- Configuration ---
MODEL_FILENAME = "best_int8_dynamic.onnx"  # <<< IMPORTANT: Change this to your ONNX model's filename
                                  # This file should be in the 'model_files/' directory.

# Path assumes this script is in 'src/' and models are in '../model_files/'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_files', MODEL_FILENAME)

WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 640 # You can adjust this; 640x640 is also common for YOLO inputs
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence to consider a detection

# --- Classes to Count (MUST EXACTLY MATCH YOUR MODEL'S CLASS NAMES) ---
# Replace these with the actual class names your model detects and you want to count.
# Example: CLASSES_TO_COUNT = ["person", "car"]
CLASSES_TO_COUNT = [
    "black", "defect", "long_screw", "nail",
    "nut", "rivet", "tek_screw", "washer"
]
# --- End Configuration ---

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print(f"Please ensure '{MODEL_FILENAME}' is in the 'model_files' directory.")
        print(f"Current working directory (inside container likely): {os.getcwd()}")
        return

    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
        print("Model class names:", model.names) # Verify your class names

        # Verification for CLASSES_TO_COUNT
        for cls_name_to_count in CLASSES_TO_COUNT:
            found = False
            if isinstance(model.names, dict): # model.names can be {id: 'name'}
                if cls_name_to_count in model.names.values():
                    found = True
            elif isinstance(model.names, list): # or a list ['name1', 'name2']
                 if cls_name_to_count in model.names:
                    found = True
            if not found:
                print(f"WARNING: Class '{cls_name_to_count}' in your CLASSES_TO_COUNT list "
                      f"is NOT FOUND in the model's class names. "
                      f"This class will not be counted unless the name matches exactly (case-sensitive).")
                print(f"Available model names from model: {list(model.names.values()) if isinstance(model.names, dict) else model.names}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Opening camera {WEBCAM_INDEX}...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Could not open camera {WEBCAM_INDEX}")
        return
    print(f"Camera opened successfully! Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    print("Starting detection... Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break

        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0] # verbose=False for less console output
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections for specific classes to count
        counted_detections_this_frame = []
        object_counts_per_class = defaultdict(int)
        
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            # Get class name using model.names
            class_name = ""
            if isinstance(model.names, dict) and class_id in model.names:
                class_name = model.names[class_id]
            elif isinstance(model.names, list) and class_id < len(model.names):
                class_name = model.names[class_id]

            if class_name in CLASSES_TO_COUNT:
                counted_detections_this_frame.append(detections[i])
                object_counts_per_class[class_name] += 1
        
        # Create a new Detections object for annotation if you only want to draw counted objects
        # For simplicity here, we'll annotate all detections above threshold, but count specific ones.
        # If you want to only draw boxes for counted items, create a new sv.Detections object:
        # display_detections = sv.Detections(...) # from counted_detections_this_frame attributes

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections) # Annotate all confident detections
        
        # Prepare labels for all confident detections
        labels = []
        if len(detections) > 0:
            labels = [
                f"{model.names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Display specific class counts
        y_offset = 30
        total_counted_objects = 0
        for class_name in CLASSES_TO_COUNT:
            count = object_counts_per_class[class_name]
            total_counted_objects += count
            cv2.putText(
                annotated_frame,
                f"{class_name}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2 # Cyan color for counts
            )
            y_offset += 25
        
        cv2.putText(
            annotated_frame,
            f"Total Monitored: {total_counted_objects}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 0, 255), 2 # Magenta for total
        )

        cv2.imshow("ONNX Object Detection - Jetson Nano", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped. Resources released.")

if __name__ == "__main__":
    main()