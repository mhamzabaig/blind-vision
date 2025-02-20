
import torch
# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to('cuda')
yolo_model.eval()

# Load MiDas model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to('cuda')
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

midas.eval()

def estimate_depth_midas(frame):
    frame_transformed = midas_transforms(frame).to('cuda')
    with torch.no_grad():
        prediction = midas(frame_transformed)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def danger_threshold(distance: float):
    return "DangerAhead" if distance > 100 else "NoDanger"