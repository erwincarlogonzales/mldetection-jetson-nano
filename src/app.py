import cv2
import supervision as sv
from ultralytics import YOLO
import os
from collections import defaultdict
import numpy as np

# --- Configuration ---
MODEL_FILENAME = "best_int8_dynamic.onnx"  # Your ONNX model's filename
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_files', MODEL_FILENAME)

WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.5

CLASSES_TO_COUNT = [
    "black", "defect", "long_screw", "nail",
    "nut", "rivet", "tek_screw", "washer"
]
# --- End Configuration ---

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print(f"Please ensure '{MODEL_FILENAME}' is in the 'model_files' directory.")
        return

    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
        print("Model class names:", model.names)

        for cls_name_to_count in CLASSES_TO_COUNT:
            found = False
            if isinstance(model.names, dict):
                if cls_name_to_count in model.names.values():
                    found = True
            elif isinstance(model.names, list):
                 if cls_name_to_count in model.names:
                    found = True
            if not found:
                print(f"WARNING: Class '{cls_name_to_count}' in your CLASSES_TO_COUNT list "
                      f"is NOT FOUND in the model's class names.")
                print(f"Available model names: {list(model.names.values()) if isinstance(model.names, dict) else model.names}")
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
    actual_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened successfully! Actual Resolution: {actual_frame_width}x{actual_frame_height}")

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_padding=3) 

    print("Starting detection... Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break

        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        result = results[0] 
        detections = sv.Detections.from_ultralytics(result)

        filtered_detections_list = []
        if len(detections) > 0 and detections.class_id is not None:
            for i in range(len(detections)):
                class_id = detections.class_id[i]
                class_name = ""
                if isinstance(model.names, dict) and class_id in model.names:
                    class_name = model.names[class_id]
                elif isinstance(model.names, list) and class_id < len(model.names):
                    class_name = model.names[class_id]
                
                if class_name in CLASSES_TO_COUNT:
                    filtered_detections_list.append(detections[i])
        
        if filtered_detections_list:
            xyxy = np.array([d.xyxy[0] for d in filtered_detections_list])
            confidence = np.array([d.confidence[0] for d in filtered_detections_list])
            class_id = np.array([d.class_id[0] for d in filtered_detections_list])
            filtered_sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
        else:
            filtered_sv_detections = sv.Detections.empty()

        object_counts_per_class = defaultdict(int)
        total_counted_objects = 0
        if len(filtered_sv_detections) > 0:
            for i in range(len(filtered_sv_detections)):
                class_id = filtered_sv_detections.class_id[i]
                class_name = model.names[class_id] 
                object_counts_per_class[class_name] += 1
                total_counted_objects +=1
        
        annotated_frame = frame.copy()
        if len(filtered_sv_detections) > 0:
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=filtered_sv_detections)
            labels = [
                f"{model.names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(filtered_sv_detections.class_id, filtered_sv_detections.confidence)
            ]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=filtered_sv_detections, labels=labels)
        
        # --- MODIFIED SECTION FOR DISPLAYING COUNTS ---
        y_offset = 30
        # Iterate through the items in object_counts_per_class.
        # This dictionary now only contains classes from CLASSES_TO_COUNT that were detected in this frame.
        if object_counts_per_class: # Check if the dictionary is not empty
            for class_name, count in object_counts_per_class.items():
                cv2.putText(
                    annotated_frame,
                    f"{class_name}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2 # Cyan color
                )
                y_offset += 25
        else: # If no objects from CLASSES_TO_COUNT were detected, maybe print a "No monitored objects" message
            cv2.putText(
                annotated_frame,
                "No monitored objects detected",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (128, 128, 128), 2 # Gray color
            )
            y_offset +=25


        cv2.putText(
            annotated_frame,
            f"Total: {total_counted_objects}", # This already reflects only counted items
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 0, 255), 2 # Magenta
        )
        # --- END OF MODIFIED SECTION ---

        inference_time_ms = result.speed.get('inference', 0.0)
        cv2.putText(
            annotated_frame,
            f"Inference: {inference_time_ms:.2f} ms",
            (10, y_offset + 30), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 0), 2
        )

        cv2.imshow("ONNX Object Detection - Jetson Nano (PC Test)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped. Resources released.")

if __name__ == "__main__":
    main()