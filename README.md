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

## Cloudflare Tunnel (required for VR access)
WebXR generally requires a secure context (HTTPS). Use Cloudflare Tunnel or a deployed HTTPS server so the headset can access the page.

Start the tunnel:

```bash
cloudflared tunnel run vr-headset
```

Once the tunnel is up, open the tunnel URL in the headset browser to access the VR page.

## Quick camera sanity test
Use the direct DepthAI display test to verify the camera feed:

```bash
python test_display.py
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