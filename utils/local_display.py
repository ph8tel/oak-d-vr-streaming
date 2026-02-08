import cv2
import depthai as dai
import numpy as np
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()


monoLeft.setCamera("left")
monoRight.setCamera("right")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

monoLeft.setFps(30)
monoRight.setFps(30)

# Manual settings if needed
# test_ctrl = monoLeft.initialControl
# test_ctrl.setManualExposure(20000, 800)

# test_ctrl = monoRight.initialControl
# test_ctrl.setManualExposure(20000, 800)


left_ctrl = monoLeft.initialControl
right_ctrl = monoRight.initialControl

left_ctrl.setAutoExposureEnable()
left_ctrl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)

right_ctrl.setAutoExposureEnable()
right_ctrl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties


# Linking
stereo.rectifiedLeft.link(xoutLeft.input)
stereo.rectifiedRight.link(xoutRight.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        inLeft = qLeft.get()
        inRight = qRight.get()

        if inLeft is not None:
            left_frame = inLeft.getCvFrame()
            cv2.imshow("left", left_frame)

        if inRight is not None:
            right_frame = inRight.getCvFrame()
            cv2.imshow("right", right_frame)

        if cv2.waitKey(1) == ord('q'):
            break