# vr-streaming-from-OAK-D-Pro

Local WebRTC server that streams a rectified stereo pair from an OAK-D (DepthAI) camera to a WebXR client for VR telepresence. The server provides HTTP signaling and two synchronized video tracks; the browser receives both tracks and renders them per eye.

## What it does
- Captures left/right mono frames from the OAK-D, rectifies them with DepthAI, and serves them as two WebRTC video tracks.
- The browser creates a WebXR session and renders the two tracks to left/right eye viewports.
- Optional: a WebRTC data channel can forward pose data to a separate servo controller over TCP.

## Run the server
From the repo root:

```bash
python vr_server.py
```

The server listens on http://0.0.0.0:8080.

## Python 3.10 setup
Create and activate a Python 3.10 virtual environment, then install dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout
```
index.html              — HTML shell, served at /
static/
  index.js              — WebXR + WebGL render loop, WebRTC client
  index.css             — UI styles
  logging.js            — in-VR log overlay helpers (shared)
  sanity.html           — standalone per-eye colour test (no camera)
stereo_camera.py        — OAK-D DepthAI pipeline (rectified stereo)
vr_server.py            — aiohttp + aiortc signaling & streaming server
utils/local_display.py  — OpenCV preview of the stereo feed
tests/                  — pytest suite (runs without camera hardware)
```

## Cloudflare Tunnel (required for VR access)
WebXR generally requires a secure context (HTTPS). Use Cloudflare Tunnel or a deployed HTTPS server so the headset can access the page.

Start the tunnel:

```bash
cloudflared tunnel run vr-headset
```

Once the tunnel is up, open the tunnel URL in the headset browser to access the VR page.

## Flat stereo pipeline (camera → goggles)

The system streams **flat** (2D) video to each eye — there is no 3D scene graph or depth-based rendering. The physical camera moves with the user's head via servos, so the VR headset simply displays each camera's rectified image in the corresponding eye viewport.

```
OAK-D mono left/right
  │
  ▼
StereoDepth node (rectification + sync)
  │
  ▼
StereoCamera.get_frames()          — grayscale → RGB, resize to 1280×720
  │
  ▼
CameraTrack.recv()                 — RGB → BGR, wrap as av.VideoFrame(bgr24)
  │  (left track drives capture;
  │   right track reuses cached pair to prevent eye desync)
  │
  ▼
WebRTC (two sendonly video tracks)
  │
  ▼
Browser <video> elements            — hidden, used only as texture sources
  │
  ▼
WebGL texImage2D per eye            — uploads each <video> as a GPU texture
  │
  ▼
XR render loop (onXRFrame)          — draws a fullscreen quad per eye viewport
  │
  ▼
Per-eye texture-coordinate offset   — shifts UVs to align each camera's optical
  (uTexOffset uniform)                axis with the viewport center and fuse
                                      the stereo pair at a given convergence
                                      distance
```

### Why a texture-coordinate offset, not a full projection matrix

Because the video is already a flat 2D image (not a 3D scene), the quad must fill the viewport at 1:1 scale. There is no 3D geometry to project — only two corrections are needed:

1. **Principal-point correction** — account for the difference between each camera's optical center (`cx`, `cy`) and the frame center.
2. **Convergence shift** — apply a half-disparity offset so the stereo pair fuses at a chosen distance.

A full OpenCV-style perspective projection (`2*fx/width`, depth terms, etc.) is designed for projecting 3D world-space geometry through a pinhole model. Applying it to an already-captured flat image over-scales the quad and distorts convergence, causing diplopia.

Both corrections are combined into a single `uTexOffset` uniform (in texture-coordinate space, 0–1) computed from calibration intrinsics served by `/calibration`:

```
dx_pp   = (cx − width/2) / width          — principal-point offset (texcoord space)
conv_tc = (fx × baseline) / (2 × conv × width) — half-disparity for convergence
du      = −dx_pp ± conv_tc                 — combined per-eye horizontal offset
dv      = (cy − height/2) / height         — vertical principal-point offset
```

The `±` sign is `+` for the left eye and `−` for the right, pushing the textures apart by the disparity that corresponds to the target convergence distance (default 1.2 m).

### Head tracking

The XR render loop obtains `pose.views` from the headset but does **not** apply the headset's view/projection matrices to the video. The headset pose is forwarded over a WebRTC data channel (`poseData`) to a servo controller that physically moves the camera. The video in the goggles always shows exactly what the camera sees — no synthetic re-projection.

## Quick camera sanity test
Use the local display script to verify the camera feed (requires a connected monitor):

```bash
python utils/local_display.py
```

## Tests (with mocks)
Unit tests use pytest and mock DepthAI/OpenCV so they can run without camera hardware.
The fakes live in [tests/test_stereo_camera.py](tests/test_stereo_camera.py) and stub out:
- DepthAI pipeline/device/queues to return deterministic frames
- OpenCV color conversion/resize to keep assertions simple

Run tests:

```bash
pytest
```