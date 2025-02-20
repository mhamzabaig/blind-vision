# import torch
# import cv2
# import numpy as np
# import av
# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
# from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # Allowed origins (update this if needed)
# origins = [
#     "http://localhost:3000",  # If running frontend locally
#     "https://f9d2-39-45-56-86.ngrok-free.app/",  # Ngrok tunnel URL
#     # "*"  # Allow all origins (not recommended for production)
# ]

# # Add CORS middleware to allow frontend requests
# app.add_middleware(
#     # CORSMiddleware,
#     # allow_origins=origins,  # Allow only specific origins
#     # allow_credentials=True,
#     # allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
#     # allow_headers=["*"],  # Allow all headers

#     CORSMiddleware,
#     allow_origins=["*"],  # ✅ Allow all origins (for testing)
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app = FastAPI()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# print(torch.cuda.is_available())  # Should return True
# print(torch.cuda.device_count())  # Check the number of GPUs available
# print(torch.cuda.get_device_name(0)) 


# @app.get("/favicon.ico")
# async def favicon():
#     return FileResponse("static/favicon.ico")


# # Load YOLOv5 model
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to('cuda')

# # Load MiDas model for depth estimation
# midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to('cuda')
# midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

# # Ensure models are in evaluation mode
# yolo_model.eval()
# midas.eval()



# # Serve static files (index.html)
# # Serve static files from the "static" directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/")
# async def serve_index():
#     return FileResponse("static/index.html")

# # WebRTC Peer Connection
# pcs = set()

# @app.post("/offer")
# async def offer(request: dict):
#     params = RTCSessionDescription(sdp=request["sdp"], type=request["type"])
#     pc = RTCPeerConnection()
#     pcs.add(pc)

#     @pc.on("track")
#     def on_track(track):
#         print(f"🔵 Video track received! {track.kind}")  # Input track is received

#         if track.kind == "video":
#             transformed_track = VideoTransformTrack(track)
#             pc.addTrack(transformed_track)  
#             print("🟢 Transformed video track added!")  # Ensure processed track is added

#     await pc.setRemoteDescription(params)
#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     print("🔴 Sending SDP response to client!")
#     return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

# class VideoTransformTrack(VideoStreamTrack):
#     def __init__(self, track):
#         super().__init__()  
#         self.track = track  

#     async def recv(self):
#         frame = await self.track.recv()
#         img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format
        
#         # ✅ Object Detection with YOLOv5
#         results = yolo_model(img)

#         for *box, conf, cls in results.xyxy[0]:
#             x_min, y_min, x_max, y_max = map(int, box)
#             label = yolo_model.names[int(cls)]
#             score = float(conf)

#             # Crop detected object
#             cropped_object = img[y_min:y_max, x_min:x_max]

#             # ✅ Depth Estimation with MiDaS
#             depth_map = estimate_depth_midas(cropped_object)
#             avg_depth = np.mean(depth_map)
            
#             status = danger_threshold(avg_depth)


#             # ✅ Draw Bounding Box and Label
#             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             # label_text = f'{label}: \n{score:.2f}, Depth: \n{avg_depth:.2f}m, \nstatus:{status}'
#             label_text = f'{label} Depth: \n{avg_depth-7500:.2f}m, status:{status}'
#             cv2.putText(img, label_text, (x_min+5, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # ✅ Convert back to Video Frame and Return
#         new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
#         new_frame.pts = frame.pts
#         new_frame.time_base = frame.time_base
        
#         return new_frame






# # Depth Estimation Function
# def estimate_depth_midas(frame):
#     frame_transformed = midas_transforms(frame).to('cuda')
#     with torch.no_grad():
#         prediction = midas(frame_transformed)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=frame.shape[:2],
#             mode='bicubic',
#             align_corners=False,
#         ).squeeze()
#         print(prediction)
#     return prediction.cpu().numpy()

# def danger_threshold(distance: float):
#         if( distance> 100):
#             return "DangerAhead"
#         return "NoDanger"

import torch
import cv2
import numpy as np
import av
from model_utils import yolo_model,danger_threshold,estimate_depth_midas 
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
import json,os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#starting app Server
app = FastAPI()
# Get the absolute path to the "static" folder in the project root
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))

# Mount the static folder with the corrected path
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(f"{static_path}/index.html")


pcs = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔵 WebSocket connected")

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print(f"🔵 Video track received: {track.kind}")
        if track.kind == "video":
            transformed_track = VideoTransformTrack(track)
            pc.addTrack(transformed_track)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "offer":
                offer = RTCSessionDescription(sdp=message["sdp"], type="offer")
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_text(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
                print("🟢 Answer sent")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
    finally:
        await pc.close()
        pcs.remove(pc)
        print("🔴 WebSocket closed")

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        results = yolo_model(img)
        
        for *box, conf, cls in results.xyxy[0]:
            x_min, y_min, x_max, y_max = map(int, box)
            #class Labels
            label = yolo_model.names[int(cls)]
            
            #To find out the specific object distance
            cropped_object = img[y_min:y_max, x_min:x_max]
            depth_map = estimate_depth_midas(cropped_object)
            avg_depth = np.mean(depth_map)
            
            #Desired Output wither person in danger or not
            status = danger_threshold(avg_depth)
            
            #displaying identified objects
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_text = f'{label} Depth: {avg_depth:.2f}m, Status: {status}'
            cv2.putText(img, label_text, (x_min+5, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame





