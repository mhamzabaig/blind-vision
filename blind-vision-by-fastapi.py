import torch
import cv2
import numpy as np
import av
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder


from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")


# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MiDas model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

# Ensure models are in evaluation mode
yolo_model.eval()
midas.eval()



# Serve static files (index.html)
# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# WebRTC Peer Connection
pcs = set()

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()  
        self.track = track  

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format

        # ✅ Object Detection with YOLOv5
        results = yolo_model(img)

        for *box, conf, cls in results.xyxy[0]:
            x_min, y_min, x_max, y_max = map(int, box)
            label = yolo_model.names[int(cls)]
            score = float(conf)

            # Crop detected object
            cropped_object = img[y_min:y_max, x_min:x_max]

            # ✅ Depth Estimation with MiDaS
            depth_map = estimate_depth_midas(cropped_object)
            avg_depth = np.mean(depth_map)

            # ✅ Draw Bounding Box and Label
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_text = f'{label}: {score:.2f}, Depth: {avg_depth:.2f}m'
            cv2.putText(img, label_text, (x_min+5, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ Convert back to Video Frame and Return
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


@app.post("/offer")
async def offer(request: dict):
    params = RTCSessionDescription(sdp=request["sdp"], type=request["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print("🔵 Video track received!")  # Input track is received

        if track.kind == "video":
            transformed_track = VideoTransformTrack(track)
            pc.addTrack(transformed_track)  
            print("🟢 Transformed video track added!")  # Ensure processed track is added

    await pc.setRemoteDescription(params)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print("🔴 Sending SDP response to client!")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}



# Depth Estimation Function
def estimate_depth_midas(frame):
    frame_transformed = midas_transforms(frame).to('cpu')
    with torch.no_grad():
        prediction = midas(frame_transformed)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()
