// Main client JS extracted from index.html
let pc;
const leftVideo = document.getElementById("leftVideo");
const rightVideo = document.getElementById("rightVideo");

let xrSession = null;
let gl = null;
let xrRefSpace = null;
let xrLayer = null;

let leftTexture = null;
let rightTexture = null;
let program = null;
let positionBuffer = null;

let hasLoggedLeftVideo = false;
let hasLoggedRightVideo = false;

let uTexOffsetLoc = null;
let samplerLoc = null;
let positionAttrLoc = -1;

let texOffsetLeft = null;
let texOffsetRight = null;

let calib = null;
let dataChannel = null;

async function loadCalibration() {
  try {
    const res = await fetch("/calibration");
    if (!res.ok) throw new Error(`Calibration request failed: ${res.status}`);
    calib = await res.json();
    if (!calib || !calib.k_left || !calib.k_right || typeof calib.baseline_m !== "number") {
      throw new Error("Calibration payload missing required fields");
    }
    setStatus("Calibration loaded");
    logToOverlay("Calibration: " + JSON.stringify(calib), "info");
  } catch (err) {
    logError("Failed to load calibration", err);
    throw err;
  }
}

/**
 * Compute a per-eye horizontal texture-coordinate offset.
 *
 * Shifts which part of each camera's image is displayed, effectively
 * reducing or increasing the stereo baseline as seen by the viewer.
 * This moves the zero-disparity (fusion) plane from infinity to a
 * finite convergence distance.
 *
 * Left eye: shift texcoords right (+u) → crops from the right side of
 *           the left camera image → moves image nasally.
 * Right eye: shift texcoords left (−u) → crops from the left side of
 *            the right camera image → moves image nasally.
 *
 * The shift in texcoord units (0–1):
 *   offset = fx * baseline_m / (2 * convergence_m * width)
 *
 * @param {number[][]} K              3×3 intrinsic matrix
 * @param {number}     width          frame width in pixels
 * @param {number}     height         frame height in pixels
 * @param {number}     baseline_m     camera baseline in metres
 * @param {number}     convergence_m  target fusion distance in metres
 * @param {string}     eye            "left" or "right"
 * @returns {Float32Array} [du, dv] texture-coordinate offset
 */
function perEyeTexOffset(K, width, height, baseline_m, convergence_m, eye) {
  const fx = K[0][0];
  const cx = K[0][2];
  const cy = K[1][2];

  // Principal-point correction in texcoord space (0–1)
  const dx_pp = (cx - width  / 2) / width;
  const dy_pp = (cy - height / 2) / height;

  // Convergence shift in texcoord space (0–1), half-disparity per eye
  const conv_tc = (fx * baseline_m) / (2.0 * convergence_m * width);
  const eye_sign = (eye === "left") ? 1 : -1;

  const du = -dx_pp + eye_sign * conv_tc;
  const dv = dy_pp;

  return new Float32Array([du, dv]);
}

function createDefaultTextureCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#333333';
  ctx.fillRect(0, 0, 256, 256);
  ctx.fillStyle = '#666666';
  for (let i = 0; i < 256; i += 32) {
    for (let j = 0; j < 256; j += 32) {
      if ((i + j) % 64 === 0) ctx.fillRect(i, j, 32, 32);
    }
  }
  return canvas;
}

async function startWebRTC() {
  pc = new RTCPeerConnection({
    iceServers: [
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:stun1.l.google.com:19302" }
    ]
  });
  setStatus("Starting WebRTC");

  pc.onconnectionstatechange = () => setStatus(`WebRTC connection: ${pc.connectionState}`);
  pc.oniceconnectionstatechange = () => setStatus(`ICE state: ${pc.iceConnectionState}`);

  // Client is the offerer — create the data channel here so SCTP
  // is included in the offer SDP.  The server picks it up via ondatachannel.
  dataChannel = pc.createDataChannel("poseData", { ordered: true, maxRetransmits: 0 });
  dataChannel.onopen = () => setStatus("Data channel open");
  dataChannel.onclose = () => { dataChannel = null; setStatus("Data channel closed"); };
  dataChannel.onmessage = (e) => { logToOverlay("Server: " + e.data); };

  let videoCount = 0;
  pc.ontrack = (event) => {
    if (event.track.kind !== "video") return;
    const stream = new MediaStream([event.track]);
    let targetVideo = null;
    if (event.transceiver && event.transceiver.mid !== null) {
      targetVideo = event.transceiver.mid === "0" ? leftVideo : rightVideo;
    } else {
      targetVideo = videoCount === 0 ? leftVideo : rightVideo;
    }
    targetVideo.srcObject = stream;
    targetVideo.onloadedmetadata = () => { targetVideo.play().catch(() => {}); };
    setStatus(`Video track ${videoCount + 1} attached`);
    videoCount++;
  };

  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("video", { direction: "recvonly" });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const response = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
  });

  const answer = await response.json();
  if (answer.error) throw new Error(answer.error);
  await pc.setRemoteDescription(answer);
}

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(info || "Shader compile failed");
  }
  return shader;
}

function initGLResources() {
  const canvas = document.getElementById("glcanvas");
  gl = canvas.getContext("webgl", { xrCompatible: true });
  const vsSource = `
    attribute vec2 a_position;
    uniform vec2 uTexOffset;
    varying vec2 v_texcoord;
    void main() {
      // Derive texcoords from NDC position: (-1,-1)..(1,1) → (0,1)..(1,0)
      // Flip Y because texcoord (0,0) is top-left but NDC (-1,-1) is bottom-left
      v_texcoord = vec2(a_position.x * 0.5 + 0.5, 1.0 - (a_position.y * 0.5 + 0.5)) + uTexOffset;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;
  const fsSource = `
    precision mediump float;
    varying vec2 v_texcoord;
    uniform sampler2D u_texture;
    void main() { gl_FragColor = texture2D(u_texture, v_texcoord); }
  `;
  const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
  program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    throw new Error(info || "Program link failed");
  }
  uTexOffsetLoc = gl.getUniformLocation(program, "uTexOffset");
  samplerLoc = gl.getUniformLocation(program, "u_texture");
  positionAttrLoc = gl.getAttribLocation(program, "a_position");
  positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  // Use TRIANGLE_STRIP quad (4 vertices) in NDC to fill each eye's viewport
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
     1,  1
  ]), gl.STATIC_DRAW);

  // Bind position attribute if present
  if (positionAttrLoc >= 0) {
    gl.enableVertexAttribArray(positionAttrLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(positionAttrLoc, 2, gl.FLOAT, false, 0, 0);
  } else {
    console.warn('position attribute not found in index.js shader');
  }

  // Initialize textures with default content
  leftTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, leftTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const defaultTexture = createDefaultTextureCanvas();
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, defaultTexture);

  rightTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, rightTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, defaultTexture);
}

async function startXR() {
  if (!navigator.xr) { logError("WebXR not supported"); return; }
  try {
    await startWebRTC();
    await loadCalibration();
  } catch (err) { logError("Startup failed", err); return; }

  if (!gl) {
    try { initGLResources(); } catch (err) { logError("WebGL init failed", err); return; }
  }
  const width = 1280, height = 720;

  // Convergence distance — objects at this distance will fuse (zero
  // disparity).  Closer objects pop forward, farther ones recede.
  // 1.2 m ≈ 4 ft is good for a bookcase-distance test.
  const CONVERGENCE_M = 1.2;

  texOffsetLeft  = perEyeTexOffset(calib.k_left,  width, height,
                                    calib.baseline_m, CONVERGENCE_M, "left");
  texOffsetRight = perEyeTexOffset(calib.k_right, width, height,
                                    calib.baseline_m, CONVERGENCE_M, "right");

  const shiftPx = (calib.k_left[0][0] * calib.baseline_m / (2 * CONVERGENCE_M)).toFixed(1);
  logToOverlay(
    `Convergence: ${CONVERGENCE_M}m, baseline: ${(calib.baseline_m * 1000).toFixed(1)}mm, ` +
    `fx: ${calib.k_left[0][0].toFixed(1)}, shift: ${shiftPx}px/eye`, "info");


  xrSession = await navigator.xr.requestSession("immersive-vr", {
    requiredFeatures: ["local-floor"], optionalFeatures: ["dom-overlay"], domOverlay: { root: document.body }
  });
  await gl.makeXRCompatible();
  xrLayer = new XRWebGLLayer(xrSession, gl);
  xrSession.updateRenderState({ baseLayer: xrLayer });
  xrRefSpace = await xrSession.requestReferenceSpace("local-floor");
  xrSession.requestAnimationFrame(onXRFrame);
}

function updateTextureFromVideo(texture, video) {
  if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
    if (video === leftVideo && !hasLoggedLeftVideo) { setStatus("Left video not ready yet"); hasLoggedLeftVideo = true; }
    if (video === rightVideo && !hasLoggedRightVideo) { setStatus("Right video not ready yet"); hasLoggedRightVideo = true; }
    return;
  }
  try {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
  } catch (err) {
    console.error("Error updating texture from video:", err);
  }
}

function onXRFrame(time, frame) {
  try {
    const session = frame.session;
    const pose = frame.getViewerPose(xrRefSpace);
    if (!pose) { session.requestAnimationFrame(onXRFrame); return; }

    // Send head pose to servo controller via data channel
    if (dataChannel && dataChannel.readyState === "open") {
      const p = pose.transform.position;
      const q = pose.transform.orientation;
      dataChannel.send(JSON.stringify({
        type: "pose",
        position: { x: p.x, y: p.y, z: p.z },
        orientation: { x: q.x, y: q.y, z: q.z, w: q.w }
      }));
    }

    // Send controller joystick + button state
    sendControllerData(session.inputSources);
    // Only require the texture sampler to be available — render fixed video
    // per-eye even if camera calibration or projection data is not present.
    if (!samplerLoc) {
      setStatus("Waiting for GL sampler");
      session.requestAnimationFrame(onXRFrame);
      return;
    }
    const baseLayer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
    gl.clearColor(0,0,0,1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);

    // Re-bind position attribute every frame — the XR compositor
    // clobbers WebGL1 global attribute state between frames.
    if (positionAttrLoc >= 0) {
      gl.enableVertexAttribArray(positionAttrLoc);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.vertexAttribPointer(positionAttrLoc, 2, gl.FLOAT, false, 0, 0);
    }

    // Render for each eye — use XRView.eye to pick the correct texture
    // and texcoord offset.
    const views = pose.views;
    for (let i = 0; i < views.length; i++) {
      const view = views[i];
      const viewport = baseLayer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);

      const isLeft = (view.eye === "left");

      // Update and bind the correct texture for this eye
      if (isLeft) {
        updateTextureFromVideo(leftTexture, leftVideo);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, leftTexture);
      } else {
        updateTextureFromVideo(rightTexture, rightVideo);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, rightTexture);
      }

      if (samplerLoc) gl.uniform1i(samplerLoc, 0);

      // Set per-eye texture offset for convergence
      const offset = isLeft ? texOffsetLeft : texOffsetRight;
      if (uTexOffsetLoc) {
        gl.uniform2fv(uTexOffsetLoc, offset || new Float32Array([0, 0]));
      }

      // Draw the fullscreen quad as a TRIANGLE_STRIP (4 verts)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    session.requestAnimationFrame(onXRFrame);
  } catch (err) { logError("XR frame error", err); }
}

document.getElementById("enter-vr").addEventListener("click", () => { startXR(); });
window.addEventListener("error", (event) => { logError("Unhandled error", event.error || event.message); });
window.addEventListener("unhandledrejection", (event) => { logError("Unhandled promise rejection", event.reason); });