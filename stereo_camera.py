import depthai as dai
import numpy as np
import cv2

class StereoCamera:
    def __init__(self, size=(1280, 720)):
        self.size = size

        self.pipeline = dai.Pipeline()

        # --- Mono cameras ---
        mono_left = self.pipeline.createMonoCamera()
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        left_ctrl = mono_left.initialControl
        left_ctrl.setAutoExposureEnable()
        left_ctrl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)

        mono_right = self.pipeline.createMonoCamera()
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
 
        right_ctrl = mono_right.initialControl
        right_ctrl.setAutoExposureEnable()
        right_ctrl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)

        # --- StereoDepth node (rectification + sync) ---
        stereo = self.pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setRectifyEdgeFillColor(0)  # black borders
        stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)
        stereo.setSubpixel(False)
        stereo.setLeftRightCheck(True)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # --- Output queues for rectified left/right ---
        xout_left = self.pipeline.createXLinkOut()
        xout_left.setStreamName("left")
        stereo.rectifiedLeft.link(xout_left.input)

        xout_right = self.pipeline.createXLinkOut()
        xout_right.setStreamName("right")
        stereo.rectifiedRight.link(xout_right.input)

        # --- Start device ---
        self.device = dai.Device(self.pipeline)
        self.q_left = self.device.getOutputQueue("left", maxSize=4, blocking=False)
        self.q_right = self.device.getOutputQueue("right", maxSize=4, blocking=False)

        print("StereoCamera initialized with OAK-D Pro (rectified stereo)")

    def get_frames(self):
        # Pull synchronized frames
        left_msg = self.q_left.get()
        right_msg = self.q_right.get()

        frameL = left_msg.getCvFrame()
        frameR = right_msg.getCvFrame()

        # Convert mono → RGB to match Pi API
        frameL_rgb = cv2.cvtColor(frameL, cv2.COLOR_GRAY2RGB)
        frameR_rgb = cv2.cvtColor(frameR, cv2.COLOR_GRAY2RGB)

        # Resize to requested size
        if (frameL_rgb.shape[1], frameL_rgb.shape[0]) != self.size:
            frameL_rgb = cv2.resize(frameL_rgb, self.size)
            frameR_rgb = cv2.resize(frameR_rgb, self.size)

        return frameL_rgb, frameR_rgb

    def get_stereo_frame(self):
        L, R = self.get_frames()
        return np.hstack((L, R))

    def stop(self):
        try:
            self.device.close()
        except:
            pass
    
    def get_frames_once(self):
        if not hasattr(self, "_cached"):
            self._cached = self.get_frames()
        return self._cached

    def clear_cache(self):
        if hasattr(self, "_cached"):
            del self._cached