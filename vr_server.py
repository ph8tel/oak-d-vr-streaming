# server.py

import asyncio
import cv2
import fractions
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiohttp import web
import numpy as np
from stereo_camera import StereoCamera

# -----------------------------
# Custom Video Track for WebRTC
# -----------------------------
class CameraTrack(VideoStreamTrack):
    """Sends one eye of the stereo pair with clock-based pacing.

    Both left and right tracks share the same frame cache so
    they always deliver the same stereo pair (no eye desync).
    The left track drives the capture; the right track reuses it.
    """
    _FPS = 30
    _INTERVAL = 1 / _FPS
    _TIME_BASE = fractions.Fraction(1, 90000)
    _PTS_STEP = int(90000 / _FPS)

    # Shared across both tracks
    _current_pts = 0
    _cached_left = None
    _cached_right = None
    _cache_lock = None  # set in first recv()

    def __init__(self, stereo_cam, side="left"):
        super().__init__()
        self.stereo_cam = stereo_cam
        self.side = side
        self._start_time = None
        self._frame_count = 0
        print(f"CameraTrack created: {side}")

    async def recv(self):
        from av import VideoFrame

        # Initialise shared lock once
        if CameraTrack._cache_lock is None:
            CameraTrack._cache_lock = asyncio.Lock()

        # --- Clock-based pacing (wall-clock, not fixed sleep) ---
        if self._start_time is None:
            self._start_time = asyncio.get_event_loop().time()

        self._frame_count += 1
        target_time = self._start_time + self._frame_count * self._INTERVAL
        now = asyncio.get_event_loop().time()
        wait = target_time - now
        if wait > 0:
            await asyncio.sleep(wait)

        if self._frame_count == 1:
            print(f"{self.side} track: recv() called for first time!")

        try:
            async with CameraTrack._cache_lock:
                # Left eye triggers a fresh capture (both tracks share the result)
                if self.side == "left":
                    self.stereo_cam.clear_cache()
                    frameL, frameR = self.stereo_cam.get_frames_once()
                    CameraTrack._cached_left = cv2.cvtColor(frameL, cv2.COLOR_RGB2BGR)
                    CameraTrack._cached_right = cv2.cvtColor(frameR, cv2.COLOR_RGB2BGR)
                    CameraTrack._current_pts += self._PTS_STEP

            # Pick the correct eye
            frame_bgr = CameraTrack._cached_left if self.side == "left" else CameraTrack._cached_right

            # Fallback if right track runs before first left capture
            if frame_bgr is None:
                frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

            video_frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            video_frame.pts = CameraTrack._current_pts
            video_frame.time_base = self._TIME_BASE
            return video_frame

        except Exception as e:
            print(f"CameraTrack error ({self.side}): {e}")
            import traceback
            traceback.print_exc()
            raise


# -----------------------------
# WebRTC Signaling Server
# -----------------------------
pcs = set()
# Defer creating the StereoCamera until startup so the server can run
# even if DepthAI hardware or driver APIs are missing during import.
stereo_cam = None

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

    ice_config = RTCConfiguration(iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302",
                          "stun:stun1.l.google.com:19302"])
    ])
    pc = RTCPeerConnection(configuration=ice_config)
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
        print("calibration: ", calib.k_left, calib.k_right, calib.baseline_m   )
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
    # Initialize stereo camera here so import-time failures don't kill the server
    global stereo_cam
    if stereo_cam is None:
        try:
            stereo_cam = StereoCamera(size=(1280, 720))
            print("✓ StereoCamera initialized")
        except Exception as e:
            print("Warning: StereoCamera initialization failed:", e)
            import traceback
            traceback.print_exc()
            stereo_cam = None

async def on_shutdown(app):
    global servo_writer
    # Close servo connection if open
    if servo_writer:
        servo_writer.close()
        await servo_writer.wait_closed()
    
    for pc in pcs:
        await pc.close()
    if stereo_cam:
        try:
            stereo_cam.stop()
        except Exception:
            pass
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
# Serve static assets from the ./static directory under the /static/ URL path
app.router.add_static("/static/", "./static", show_index=False)

app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

# Use HTTP - Cloudflare Tunnel provides HTTPS
print("Starting HTTP server on port 8080 (Cloudflare Tunnel provides HTTPS)")
web.run_app(app, port=8080)