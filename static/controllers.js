/**
 * controllers.js — Read VR hand controller inputs and send them
 * over the WebRTC data channel to the servo server.
 *
 * Depends on the global `dataChannel` from index.js.
 */

function sendControllerData(inputSources) {
  if (!dataChannel || dataChannel.readyState !== "open") {
    return;
  }

  const controllerMessage = {
    type: "controller",
    timestamp: Date.now(),
    leftJoystick: { x: 0, y: 0 },
    rightJoystick: { x: 0, y: 0 },
    buttons: {}
  };

  for (const inputSource of inputSources) {
    if (!inputSource.gamepad) continue;

    const gamepad = inputSource.gamepad;
    const hand = inputSource.handedness; // "left" or "right"

    // Joystick axes: axes[2] = X, axes[3] = Y (standard XR mapping)
    if (gamepad.axes.length >= 4) {
      const stick = { x: gamepad.axes[2], y: -gamepad.axes[3] }; // invert Y so forward = positive
      if (hand === "left")  controllerMessage.leftJoystick  = stick;
      if (hand === "right") controllerMessage.rightJoystick = stick;
    }

    // Buttons — Quest 3S standard mapping:
    //   0: Trigger   1: Grip   4: A/X   5: B/Y
    const btns = gamepad.buttons;

    if (btns[0] && btns[0].pressed) controllerMessage.buttons[`${hand}_trigger`] = true;
    if (btns[1] && btns[1].pressed) controllerMessage.buttons[`${hand}_grip`]    = true;
    if (btns[4] && btns[4].pressed) controllerMessage.buttons["plow_up"]         = true;
    if (btns[5] && btns[5].pressed) controllerMessage.buttons["plow_down"]       = true;

    // Convenience aliases
    if (hand === "right" && btns[1] && btns[1].pressed) controllerMessage.buttons["lights"] = true;
    if (hand === "left"  && btns[1] && btns[1].pressed) controllerMessage.buttons["horn"]   = true;
  }

  try {
    dataChannel.send(JSON.stringify(controllerMessage));
  } catch (e) {
    logToOverlay("Controller send error: " + e.message);
  }
}
