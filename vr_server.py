# server.py

import asyncio
import cv2
import fractions
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiohttp import web
import numpy as np
from stereo_camera import StereoCamera

# -----------------------------
# Custom Video Track for WebRTC
# -----------------------------
class CameraTrack(VideoStreamTrack):
    _shared_timestamp = 0
    _timestamp = 0

    def __init__(self, stereo_cam, side="left"):
        super().__init__()
        self.stereo_cam = stereo_cam
        self.side = side
        self.frame_count = 0
        self._start = None
        print(f"CameraTrack created: {side}")

    async def recv(self):
        self.frame_count += 1
        if self.frame_count == 1:
            print(f"{self.side} track: recv() called for first time!")

        await asyncio.sleep(1 / 30)

        try:
            # Left eye triggers a fresh stereo capture
            if self.side == "left":
                self.stereo_cam.clear_cache()
                frameL, frameR = self.stereo_cam.get_frames_once()

                # Shared timestamp for this stereo pair
                CameraTrack._shared_timestamp = CameraTrack._timestamp
                CameraTrack._timestamp += int(90000 / 30)
            else:
                # Right eye reuses the same stereo pair
                frameL, frameR = self.stereo_cam.get_frames_once()

            frame = frameL if self.side == "left" else frameR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"CameraTrack frame capture error ({self.side}):", e)
            import traceback
            traceback.print_exc()
            raise

        try:
            from av import VideoFrame
            video_frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            video_frame.pts = CameraTrack._shared_timestamp
            video_frame.time_base = fractions.Fraction(1, 90000)
            return video_frame

        except Exception as e:
            print(f"CameraTrack VideoFrame creation error ({self.side}):", e)
            import traceback
            traceback.print_exc()
            raise


# -----------------------------
# WebRTC Signaling Server
# -----------------------------
pcs = set()
stereo_cam = StereoCamera(size=(1280, 720))

# Store the most recent peer connection for answer endpoint
_current_pc = None

# TCP connection to Pi 4 servo controller
servo_writer = None
servo_lock = asyncio.Lock()

async def connect_to_servo_controller(host='192.168.1.79', port=9090):
    """Connect to Pi 4 servo controller (non-blocking, optional)"""
    global servo_writer
    try:
        print(f"Attempting to connect to servo controller at {host}:{port}...")
        reader, writer = await asyncio.open_connection(host, port)
        servo_writer = writer
        print(f"✓ Connected to servo controller at {host}:{port}")
        return True
    except Exception as e:
        print(f"✗ Could not connect to servo controller at {host}:{port}")
        print(f"  Error: {type(e).__name__}: {e}")
        print("  Continuing without servo control (pose data will be logged only)")
        return False

async def send_pose_to_servo(pose_json):
    """Send pose data to Pi 4 servo controller (non-blocking)"""
    global servo_writer
    if not servo_writer:
        
        return  # Silently skip if not connected
    
    async with servo_lock:
        try:
            # Send JSON with newline delimiter
            servo_writer.write((pose_json + '\n').encode('utf-8'))
            await servo_writer.drain()
        except Exception as e:
            print(f"Error sending to servo controller: {e}")
            servo_writer = None  # Reset on error

async def offer(request):
    global _current_pc
    # CORS header for actual POST response
    headers = {"Access-Control-Allow-Origin": "*"}

    params = await request.json()
    client_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    _current_pc = pc

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print(f"ICE connection state: {pc.iceConnectionState}")

    # Server creates data channel (since server creates offer)
    data_channel = pc.createDataChannel("poseData", ordered=True, maxRetransmits=0)
    print(f"[DATA CHANNEL] Created server-side channel: {data_channel.label}")
    
    @data_channel.on("message")
    def on_data_message(message):
        # Log first few characters to debug format
        msg_preview = message[:100] if len(message) > 100 else message
        print(f"[DATA CHANNEL] Received: {msg_preview}")
        
        # Forward to servo controller immediately (non-blocking)
        asyncio.create_task(send_pose_to_servo(message))
    
    @data_channel.on("open")
    def on_data_open():
        print(f"[DATA CHANNEL] '{data_channel.label}' opened - ready to receive pose data!")
        try:
            data_channel.send('{"type":"ack","message":"Server ready to receive pose data"}')
            print(f"[DATA CHANNEL] Sent ACK to client")
        except Exception as e:
            print(f"[DATA CHANNEL] Failed to send ACK: {e}")
    
    @data_channel.on("close")
    def on_data_close():
        print(f"[DATA CHANNEL] '{data_channel.label}' closed")

    # Server sends video tracks
    left_track = CameraTrack(stereo_cam, side="left")
    right_track = CameraTrack(stereo_cam, side="right")

    # Add transceivers explicitly with sendonly direction
    left_sender = pc.addTransceiver(left_track, direction="sendonly")
    right_sender = pc.addTransceiver(right_track, direction="sendonly")

    print(f"Added tracks via transceivers: left={left_track.kind}, right={right_track.kind}")

    try:
        await pc.setRemoteDescription(client_offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("Answer created, returning to client...")
    except Exception as e:
        print("Answer creation error:", e)
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, headers=headers)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }, headers=headers)

async def answer(request):
    global _current_pc
    headers = {"Access-Control-Allow-Origin": "*"}
    
    if not _current_pc:
        return web.json_response({"error": "No active peer connection"}, headers=headers)
    
    params = await request.json()
    answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    try:
        await _current_pc.setRemoteDescription(answer)
        print("WebRTC connection established successfully")
    except Exception as e:
        print("Answer processing error:", e)
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, headers=headers)
    
    return web.json_response({"status": "ok"}, headers=headers)


async def calibration(request):
    headers = {"Access-Control-Allow-Origin": "*"}
    try:
        calib = stereo_cam.get_calibration()
        return web.json_response({
            "k_left": calib.k_left,
            "k_right": calib.k_right,
            "baseline_m": calib.baseline_m,
        }, headers=headers)
    except Exception as e:
        print("Calibration error:", e)
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, headers=headers)


async def on_startup(app):
    # Connect to Pi 4 servo controller (optional, non-blocking)
    await connect_to_servo_controller(host='192.168.1.138', port=9090)

async def on_shutdown(app):
    global servo_writer
    # Close servo connection if open
    if servo_writer:
        servo_writer.close()
        await servo_writer.wait_closed()
    
    for pc in pcs:
        await pc.close()
    stereo_cam.stop()
# Add this BEFORE creating routes
@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

app = web.Application(middlewares=[cors_middleware])

app.router.add_post("/offer", offer)
app.router.add_post("/answer", answer)
app.router.add_get("/calibration", calibration)

# Serve the HTML file at root
app.router.add_get("/", lambda request: web.FileResponse("./index.html"))
app.router.add_static("/", "./", show_index=False)

app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

# Use HTTP - Cloudflare Tunnel provides HTTPS
print("Starting HTTP server on port 8080 (Cloudflare Tunnel provides HTTPS)")
web.run_app(app, port=8080)