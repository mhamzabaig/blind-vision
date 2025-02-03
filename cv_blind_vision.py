import torch
import cv2
import numpy as np

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MiDas model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

# Ensure models are in evaluation mode
yolo_model.eval()
midas.eval()

def detect_objects_yolo(frame):
    # Convert the frame from BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection with YOLOv5
    results = yolo_model(img_rgb)

    return results

def estimate_depth_midas(frame):
    # Applying MiDas transformations in CPU
    frame_transformed = midas_transforms(frame).to('cpu')

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(frame_transformed)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map

def process_camera_feed():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect objects using YOLOv5
        yolo_results = detect_objects_yolo(frame)

        # Process YOLO detections
        for *box, conf, cls in yolo_results.xyxy[0]:
            x_min, y_min, x_max, y_max = map(int, box)
            label = yolo_model.names[int(cls)]
            score = float(conf)

            # Crop the detected object from the frame
            cropped_object = frame[y_min:y_max, x_min:x_max]

            # Estimate depth using MiDas for the cropped object
            depth_map = estimate_depth_midas(cropped_object)
            avg_depth = np.mean(depth_map)

            # Draw bounding box, label, score, and depth on the original frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}, Depth: {avg_depth:.2f}m'
            cv2.putText(frame, label_text, (x_min+5, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('YOLOv5 + MiDaS', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start processing the camera feed
process_camera_feed()
