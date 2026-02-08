import numpy as np
import stereo_camera


class FakeLink:
    def link(self, _):
        pass


class FakeMonoCamera:
    def __init__(self):
        self.initialControl = FakeCameraControl()
        self.out = FakeLink()

    def setBoardSocket(self, _):
        pass

    def setResolution(self, _):
        pass


class FakeStereoDepth:
    class PresetMode:
        HIGH_ACCURACY = object()

    def __init__(self):
        self.left = FakeLink()
        self.right = FakeLink()
        self.rectifiedLeft = FakeLink()
        self.rectifiedRight = FakeLink()

    def setDefaultProfilePreset(self, _):
        pass

    def setRectifyEdgeFillColor(self, _):
        pass

    def setDepthAlign(self, _):
        pass

    def setSubpixel(self, _):
        pass

    def setLeftRightCheck(self, _):
        pass


class FakeXLinkOut:
    def __init__(self):
        self.input = FakeLink()

    def setStreamName(self, _):
        pass


class FakePipeline:
    def createMonoCamera(self):
        return FakeMonoCamera()

    def createStereoDepth(self):
        return FakeStereoDepth()

    def createXLinkOut(self):
        return FakeXLinkOut()


class FakeCameraControl:
    class AntiBandingMode:
        AUTO = object()

    def setAutoExposureEnable(self):
        pass

    def setAntiBandingMode(self, _):
        pass


class FakeMonoCameraProperties:
    class SensorResolution:
        THE_720_P = object()


class FakeCameraBoardSocket:
    LEFT = object()
    RIGHT = object()


class FakeQueue:
    def __init__(self, frames):
        self._frames = list(frames)
        self.get_call_count = 0

    def get(self):
        self.get_call_count += 1
        if not self._frames:
            return FakeMsg(np.zeros((720, 1280), dtype=np.uint8))
        return FakeMsg(self._frames.pop(0))


class FakeMsg:
    def __init__(self, frame):
        self._frame = frame

    def getCvFrame(self):
        return self._frame


class FakeDevice:
    def __init__(self, _pipeline, left_frames, right_frames):
        self._queues = {
            "left": FakeQueue(left_frames),
            "right": FakeQueue(right_frames),
        }

    def getOutputQueue(self, name, maxSize=4, blocking=False):
        return self._queues[name]

    def close(self):
        pass


class FakeDai:
    Pipeline = FakePipeline
    MonoCameraProperties = FakeMonoCameraProperties
    CameraControl = FakeCameraControl
    CameraBoardSocket = FakeCameraBoardSocket

    class node:
        StereoDepth = FakeStereoDepth

    def __init__(self, left_frames, right_frames):
        self._left_frames = left_frames
        self._right_frames = right_frames

    def Device(self, pipeline):
        return FakeDevice(pipeline, self._left_frames, self._right_frames)


class FakeCv2:
    COLOR_GRAY2RGB = 0

    def __init__(self):
        self.resize_called = 0

    def cvtColor(self, frame, _):
        return np.stack([frame, frame, frame], axis=-1)

    def resize(self, frame, size):
        self.resize_called += 1
        return np.zeros((size[1], size[0], frame.shape[2]), dtype=frame.dtype)


def test_get_frames_converts_and_resizes(monkeypatch):
    left = np.full((720, 1280), 10, dtype=np.uint8)
    right = np.full((720, 1280), 20, dtype=np.uint8)

    fake_dai = FakeDai([left], [right])
    fake_cv2 = FakeCv2()

    monkeypatch.setattr(stereo_camera, "dai", fake_dai)
    monkeypatch.setattr(stereo_camera, "cv2", fake_cv2)

    cam = stereo_camera.StereoCamera(size=(640, 360))
    frameL, frameR = cam.get_frames()

    assert frameL.shape == (360, 640, 3)
    assert frameR.shape == (360, 640, 3)
    assert fake_cv2.resize_called == 2


def test_get_frames_once_caches(monkeypatch):
    left = np.full((720, 1280), 1, dtype=np.uint8)
    right = np.full((720, 1280), 2, dtype=np.uint8)

    fake_dai = FakeDai([left], [right])
    fake_cv2 = FakeCv2()

    monkeypatch.setattr(stereo_camera, "dai", fake_dai)
    monkeypatch.setattr(stereo_camera, "cv2", fake_cv2)

    cam = stereo_camera.StereoCamera(size=(1280, 720))
    _ = cam.get_frames_once()
    _ = cam.get_frames_once()

    assert cam.q_left.get_call_count == 1
    assert cam.q_right.get_call_count == 1


def test_clear_cache_allows_new_frames(monkeypatch):
    left1 = np.full((720, 1280), 1, dtype=np.uint8)
    right1 = np.full((720, 1280), 2, dtype=np.uint8)
    left2 = np.full((720, 1280), 3, dtype=np.uint8)
    right2 = np.full((720, 1280), 4, dtype=np.uint8)

    fake_dai = FakeDai([left1, left2], [right1, right2])
    fake_cv2 = FakeCv2()

    monkeypatch.setattr(stereo_camera, "dai", fake_dai)
    monkeypatch.setattr(stereo_camera, "cv2", fake_cv2)

    cam = stereo_camera.StereoCamera(size=(1280, 720))
    frameL1, frameR1 = cam.get_frames_once()
    cam.clear_cache()
    frameL2, frameR2 = cam.get_frames_once()

    assert not np.array_equal(frameL1, frameL2)
    assert not np.array_equal(frameR1, frameR2)
