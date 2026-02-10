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
let texcoordBuffer = null;



let hasLoggedLeftVideo = false;
let hasLoggedRightVideo = false;

let uProjectionLoc = null;
let uViewLoc = null;
let samplerLoc = null;

let projLeft = null;
let projRight = null;

let calib = null;

async function loadCalibration() {
  try {
    const res = await fetch("/calibration");
    if (!res.ok) throw new Error(`Calibration request failed: ${res.status}`);
    calib = await res.json();
    if (!calib || !calib.k_left || !calib.k_right || typeof calib.baseline_m !== "number") {
      throw new Error("Calibration payload missing required fields");
    }
    setStatus("Calibration loaded");
    logError("Calibration details", JSON.stringify(calib));
  } catch (err) {
    logError("Failed to load calibration", err);
    throw err;
  }
}

/**
 * Build a per-eye shift matrix for flat video passthrough.
 *
 * For flat stereo video the quad must fill the viewport exactly (identity
 * scale).  The only correction needed is a horizontal translation that
 * aligns the physical camera's optical axis (cx) with the headset
 * viewport centre.  A positive shift moves the image left (pushes
 * convergence closer), a negative shift moves it right.
 *
 * dx = (cx - width/2) / (width/2)   → NDC units
 * dy = (cy - height/2) / (height/2) → NDC units (usually ~0)
 */
function principalPointShift(K, width, height) {
  const cx = K[0][2];
  const cy = K[1][2];
  const dx = (cx - width  / 2) / (width  / 2);
  const dy = (cy - height / 2) / (height / 2);
  // Column-major identity + translation
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    -dx, dy, 0, 1   // negate dx: camera cx right-of-centre → shift image left
  ]);
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
    attribute vec2 a_texcoord;
    uniform mat4 uProjection;
    varying vec2 v_texcoord;
    void main() {
      v_texcoord = a_texcoord;
      gl_Position = uProjection * vec4(a_position, 0.0, 1.0);
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
  uProjectionLoc = gl.getUniformLocation(program, "uProjection");
  samplerLoc = gl.getUniformLocation(program, "u_texture");
  const positionLoc = gl.getAttribLocation(program, "a_position");
  const texcoordLoc = gl.getAttribLocation(program, "a_texcoord");
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
  if (positionLoc >= 0) {
    gl.enableVertexAttribArray(positionLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  } else {
    console.warn('position attribute not found in index.js shader');
  }

  texcoordBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    0, 1,
    1, 1,
    0, 0,
    1, 0
  ]), gl.STATIC_DRAW);

  if (texcoordLoc >= 0) {
    gl.enableVertexAttribArray(texcoordLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
    gl.vertexAttribPointer(texcoordLoc, 2, gl.FLOAT, false, 0, 0);
  } else {
    console.warn('texcoord attribute not found in index.js shader');
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
  projLeft = principalPointShift(calib.k_left, width, height);
  projRight = principalPointShift(calib.k_right, width, height);

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

    // Render for each eye with fixed view (no head tracking)
    const views = pose.views;
    for (let i = 0; i < views.length; i++) {
      const view = views[i];
      const viewport = baseLayer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);

      // Update and bind the correct texture for this eye
      if (i === 0) {
        updateTextureFromVideo(leftTexture, leftVideo);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, leftTexture);
      } else {
        updateTextureFromVideo(rightTexture, rightVideo);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, rightTexture);
      }

      // Ensure the sampler is set to texture unit 0
      if (samplerLoc) gl.uniform1i(samplerLoc, 0);

      // Set per-eye projection from camera calibration
      const proj = (i === 0) ? projLeft : projRight;
      if (uProjectionLoc && proj) {
        gl.uniformMatrix4fv(uProjectionLoc, false, proj);
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

 