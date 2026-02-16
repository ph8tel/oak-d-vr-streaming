"""Unit tests for vr_server.

All heavy dependencies (aiortc, DepthAI, av) are faked so these tests
run without camera hardware or a real WebRTC stack.
"""

import asyncio
import fractions
import json

import numpy as np
import pytest

import vr_server


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeStereoCamera:
    """Minimal stand-in for StereoCamera used by CameraTrack and endpoints."""

    def __init__(self, left_frame=None, right_frame=None):
        self._left = left_frame if left_frame is not None else np.full((720, 1280, 3), 10, dtype=np.uint8)
        self._right = right_frame if right_frame is not None else np.full((720, 1280, 3), 20, dtype=np.uint8)
        self._cached = None
        self.stopped = False

    def get_frames_once(self):
        if self._cached is None:
            self._cached = (self._left.copy(), self._right.copy())
        return self._cached

    def clear_cache(self):
        self._cached = None

    def stop(self):
        self.stopped = True

    def get_calibration(self):
        return type("Calibration", (), {
            "k_left": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
            "k_right": [[801, 0, 635], [0, 801, 361], [0, 0, 1]],
            "baseline_m": 0.075,
        })()


class FakeWriter:
    """Stands in for asyncio.StreamWriter."""

    def __init__(self, *, fail_on_write=False):
        self.written = []
        self.closed = False
        self._fail = fail_on_write

    def write(self, data):
        if self._fail:
            raise ConnectionResetError("pipe broken")
        self.written.append(data)

    async def drain(self):
        if self._fail:
            raise ConnectionResetError("pipe broken")

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_camera_track_state():
    """Reset CameraTrack class-level shared state between tests."""
    vr_server.CameraTrack._current_pts = 0
    vr_server.CameraTrack._cached_left = None
    vr_server.CameraTrack._cached_right = None
    vr_server.CameraTrack._cache_lock = None


# ---------------------------------------------------------------------------
# CameraTrack tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_track_state():
    """Ensure each test starts with fresh CameraTrack shared state."""
    _reset_camera_track_state()
    yield
    _reset_camera_track_state()


@pytest.mark.asyncio
async def test_left_track_captures_and_returns_frame():
    cam = FakeStereoCamera()
    track = vr_server.CameraTrack(cam, side="left")

    frame = await track.recv()

    assert frame.width == 1280
    assert frame.height == 720
    assert frame.time_base == fractions.Fraction(1, 90000)
    assert frame.pts == vr_server.CameraTrack._PTS_STEP


@pytest.mark.asyncio
async def test_right_track_reuses_cached_frame():
    cam = FakeStereoCamera()
    left_track = vr_server.CameraTrack(cam, side="left")
    right_track = vr_server.CameraTrack(cam, side="right")

    # Left captures first
    left_frame = await left_track.recv()
    # Right reuses the cache — no new capture
    right_frame = await right_track.recv()

    assert right_frame.width == 1280
    assert right_frame.height == 720
    # PTS should be the same — right didn't increment
    assert right_frame.pts == left_frame.pts


@pytest.mark.asyncio
async def test_right_track_before_left_returns_black_fallback():
    cam = FakeStereoCamera()
    right_track = vr_server.CameraTrack(cam, side="right")

    frame = await right_track.recv()

    # Should get a black frame instead of crashing
    arr = frame.to_ndarray(format="bgr24")
    assert arr.shape == (720, 1280, 3)
    assert np.all(arr == 0)


@pytest.mark.asyncio
async def test_pts_increments_only_on_left_recv():
    cam = FakeStereoCamera()
    left_track = vr_server.CameraTrack(cam, side="left")
    right_track = vr_server.CameraTrack(cam, side="right")

    await left_track.recv()  # PTS → 3000
    await right_track.recv()  # PTS stays 3000
    await left_track.recv()  # PTS → 6000

    assert vr_server.CameraTrack._current_pts == 2 * vr_server.CameraTrack._PTS_STEP


@pytest.mark.asyncio
async def test_left_and_right_frames_differ():
    left_px = np.full((720, 1280, 3), 50, dtype=np.uint8)
    right_px = np.full((720, 1280, 3), 200, dtype=np.uint8)
    cam = FakeStereoCamera(left_frame=left_px, right_frame=right_px)

    left_track = vr_server.CameraTrack(cam, side="left")
    right_track = vr_server.CameraTrack(cam, side="right")

    left_frame = await left_track.recv()
    right_frame = await right_track.recv()

    left_arr = left_frame.to_ndarray(format="bgr24")
    right_arr = right_frame.to_ndarray(format="bgr24")
    assert not np.array_equal(left_arr, right_arr)


# ---------------------------------------------------------------------------
# send_pose_to_servo tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_pose_writes_json_with_newline():
    writer = FakeWriter()
    vr_server.servo_writer = writer
    pose = '{"type":"pose","x":1}'

    await vr_server.send_pose_to_servo(pose)

    assert len(writer.written) == 1
    assert writer.written[0] == (pose + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_send_pose_noop_when_not_connected():
    vr_server.servo_writer = None

    # Should not raise
    await vr_server.send_pose_to_servo('{"x":0}')


@pytest.mark.asyncio
async def test_send_pose_resets_writer_on_error():
    writer = FakeWriter(fail_on_write=True)
    vr_server.servo_writer = writer

    await vr_server.send_pose_to_servo('{"x":0}')

    assert vr_server.servo_writer is None


# ---------------------------------------------------------------------------
# calibration endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calibration_returns_intrinsics(aiohttp_client):
    from aiohttp import web

    cam = FakeStereoCamera()
    vr_server.stereo_cam = cam

    app = web.Application()
    app.router.add_get("/calibration", vr_server.calibration)
    client = await aiohttp_client(app)

    resp = await client.get("/calibration")
    assert resp.status == 200

    data = await resp.json()
    assert data["k_left"] == [[800, 0, 640], [0, 800, 360], [0, 0, 1]]
    assert data["k_right"] == [[801, 0, 635], [0, 801, 361], [0, 0, 1]]
    assert data["baseline_m"] == 0.075


@pytest.mark.asyncio
async def test_calibration_returns_error_on_failure(aiohttp_client):
    from aiohttp import web

    vr_server.stereo_cam = None  # will cause AttributeError

    app = web.Application()
    app.router.add_get("/calibration", vr_server.calibration)
    client = await aiohttp_client(app)

    resp = await client.get("/calibration")
    assert resp.status == 200  # endpoint returns 200 with error body

    data = await resp.json()
    assert "error" in data


# ---------------------------------------------------------------------------
# CORS middleware tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cors_preflight_returns_allow_headers(aiohttp_client):
    from aiohttp import web

    async def dummy(request):
        return web.Response(text="ok")

    app = web.Application(middlewares=[vr_server.cors_middleware])
    app.router.add_get("/test", dummy)
    # Need an OPTIONS route for preflight
    app.router.add_route("OPTIONS", "/test", dummy)
    client = await aiohttp_client(app)

    resp = await client.options("/test")
    assert resp.headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in resp.headers["Access-Control-Allow-Methods"]


@pytest.mark.asyncio
async def test_cors_adds_headers_to_normal_response(aiohttp_client):
    from aiohttp import web

    async def dummy(request):
        return web.Response(text="ok")

    app = web.Application(middlewares=[vr_server.cors_middleware])
    app.router.add_get("/test", dummy)
    client = await aiohttp_client(app)

    resp = await client.get("/test")
    assert resp.headers["Access-Control-Allow-Origin"] == "*"
    assert resp.headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
# on_startup / on_shutdown tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_startup_initialises_camera(monkeypatch):
    from aiohttp import web

    vr_server.stereo_cam = None

    # Stub out connect_to_servo_controller so it doesn't try the network
    async def _noop(**kw):
        return False
    monkeypatch.setattr(vr_server, "connect_to_servo_controller", _noop)

    # Patch StereoCamera to return our fake
    fake_cam = FakeStereoCamera()
    monkeypatch.setattr(vr_server, "StereoCamera", lambda size: fake_cam)

    app = web.Application()
    await vr_server.on_startup(app)

    assert vr_server.stereo_cam is fake_cam


@pytest.mark.asyncio
async def test_on_startup_survives_camera_failure(monkeypatch):
    from aiohttp import web

    vr_server.stereo_cam = None

    async def _noop(**kw):
        return False
    monkeypatch.setattr(vr_server, "connect_to_servo_controller", _noop)
    monkeypatch.setattr(vr_server, "StereoCamera",
                        lambda size: (_ for _ in ()).throw(RuntimeError("no camera")))

    app = web.Application()
    await vr_server.on_startup(app)

    assert vr_server.stereo_cam is None  # gracefully None, not crashed


@pytest.mark.asyncio
async def test_on_shutdown_stops_camera_and_closes_writer():
    from aiohttp import web

    cam = FakeStereoCamera()
    writer = FakeWriter()

    vr_server.stereo_cam = cam
    vr_server.servo_writer = writer
    vr_server.pcs.clear()

    app = web.Application()
    await vr_server.on_shutdown(app)

    assert cam.stopped
    assert writer.closed


@pytest.mark.asyncio
async def test_on_shutdown_tolerates_no_camera_no_writer():
    from aiohttp import web

    vr_server.stereo_cam = None
    vr_server.servo_writer = None
    vr_server.pcs.clear()

    # Should not raise
    await vr_server.on_shutdown(app=web.Application())
