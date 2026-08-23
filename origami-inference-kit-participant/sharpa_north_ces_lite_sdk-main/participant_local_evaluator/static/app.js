import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import URDFLoader from './vendor/URDFLoader.js';

const $ = id => document.getElementById(id);
const ui = {
  remoteBadge: $('remoteBadge'), policyBadge: $('policyBadge'), urdfBadge: $('urdfBadge'),
  meshBadge: $('meshBadge'),
  operation: $('operation'), statusText: $('statusText'), logs: $('logs'), metrics: $('metrics'),
  warnings: $('warnings'), stateRows: $('stateRows'), robotView: $('robotView'),
  timeline: $('timeline'), stepText: $('stepText'), playPause: $('playPause'),
  stepBack: $('stepBack'), stepForward: $('stepForward'),
};
const model = {
  config: null, robot: null, current: null, prediction: [], validation: null,
  step: 0, mode: 'current', playing: false, timer: null, imageVersion: 0,
};
let scene;
let camera;
let renderer;
let controls;

async function api(path, method = 'GET', body) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  let value;
  try { value = await response.json(); }
  catch { throw new Error(`HTTP ${response.status}: invalid JSON response`); }
  if (!response.ok || value?.ok === false) throw new Error(String(value?.error || `HTTP ${response.status}`));
  return value;
}

function badge(node, text, tone = '') {
  node.textContent = String(text);
  node.className = `badge${tone ? ` ${tone}` : ''}`;
}

function setOperation(text, tone = '') {
  ui.operation.textContent = String(text);
  ui.operation.className = `operation${tone ? ` ${tone}` : ''}`;
}

async function invoke(button, message, action) {
  button.disabled = true;
  setOperation(message, 'warn');
  try {
    const result = await action();
    setOperation('Operation completed', 'ok');
    await refreshStatus();
    return result;
  } catch (error) {
    setOperation(`Operation failed: ${error.message}`, 'bad');
    return null;
  } finally {
    button.disabled = false;
  }
}

function value(id) { return $(id).value.trim(); }

const sleep = milliseconds => new Promise(resolve => window.setTimeout(resolve, milliseconds));

function rememberImage(image) {
  if (typeof image !== 'string' || !image.trim()) return;
  $('image').value = image.trim();
  try { window.localStorage.setItem('origamiParticipantImage', image.trim()); }
  catch { /* localStorage may be disabled */ }
}

async function startPolicyWithRecovery(image) {
  try {
    return await api('/api/submission/start', 'POST', {image});
  } catch (originalError) {
    // Loading a large GPU policy may outlive a browser/proxy request even though
    // the local backend completes successfully. Recover from actual backend state.
    for (let attempt = 0; attempt < 90; attempt += 1) {
      await sleep(1000);
      try {
        const status = await api('/api/status');
        if (status.policy?.connected && status.container?.running) {
          return {recovered_after_response_disconnect: true, status};
        }
        if (status.last_error) throw new Error(status.last_error);
      } catch (statusError) {
        if (statusError.message !== originalError.message && attempt > 5) {
          // Keep polling through short connection saturation while STL assets load.
        }
      }
    }
    throw originalError;
  }
}

async function shadowWithRecovery(requestBody) {
  try {
    return await api('/api/policy/shadow', 'POST', requestBody);
  } catch (originalError) {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      await sleep(500);
      try {
        const status = await api('/api/status');
        if (status.trajectory_available && String(status.last_operation).startsWith('shadow complete')) {
          return await api('/api/trajectory');
        }
        if (status.last_error) throw new Error(status.last_error);
      } catch {
        // Keep polling through short connection saturation while meshes load.
      }
    }
    throw originalError;
  }
}

async function waitForArchiveLoad(jobId, onProgress) {
  const started = Date.now();
  while (Date.now() - started < 30 * 60 * 1000) {
    const job = await api(`/api/submission/load/status?job_id=${encodeURIComponent(jobId)}`);
    if (job.status === 'completed') return job.result;
    if (job.status === 'failed') throw new Error(job.error || 'Archive loading failed');
    const elapsed = Math.round((Date.now() - started) / 1000);
    const message = `Validating and running docker load in the background… waited ${elapsed}s`;
    if (typeof onProgress === 'function') onProgress(message);
    else setOperation(message, 'warn');
    await sleep(1000);
  }
  throw new Error('Archive loading exceeded the 30-minute timeout');
}

async function recoverArchiveLoadAfterDisconnect() {
  const status = await api('/api/status');
  const job = status.archive_job;
  if (!job?.job_id || job.status !== 'running') return null;
  return waitForArchiveLoad(job.job_id);
}

function uploadArchiveFile(file, expectedSha256) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', '/api/submission/upload');
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.setRequestHeader('X-Origami-Filename', encodeURIComponent(file.name));
    if (expectedSha256) request.setRequestHeader('X-Origami-Sha256', expectedSha256);
    request.upload.onprogress = event => {
      if (!event.lengthComputable) return;
      const percent = Math.min(100, (event.loaded / event.total) * 100);
      $('uploadProgress').textContent = `${percent.toFixed(1)}% · ${(event.loaded / 1024 ** 3).toFixed(2)} / ${(event.total / 1024 ** 3).toFixed(2)} GiB`;
      setOperation(`Streaming local submission archive: ${percent.toFixed(1)}%`, 'warn');
    };
    request.onerror = async () => {
      try {
        const recovered = await recoverArchiveLoadAfterDisconnect();
        if (recovered) {
          resolve({...recovered, recovered_after_response_disconnect: true});
          return;
        }
      } catch {
        // Fall through to the original upload error.
      }
      reject(new Error('Local file upload connection failed. If progress reached 100%, refresh the page to check the background loading status.'));
    };
    request.onload = () => {
      let response;
      try { response = JSON.parse(request.responseText); }
      catch { reject(new Error(`HTTP ${request.status}: invalid JSON response`)); return; }
      if (request.status < 200 || request.status >= 300 || response?.ok === false) {
        reject(new Error(String(response?.error || `HTTP ${request.status}`)));
        return;
      }
      resolve(response);
    };
    request.send(file);
  });
}

async function loadArchiveWithPolling(requestBody) {
  try {
    const job = await api('/api/submission/load', 'POST', requestBody);
    if (job.images?.length) return job;
    return waitForArchiveLoad(job.job_id);
  } catch (originalError) {
    const recovered = await recoverArchiveLoadAfterDisconnect();
    if (recovered) return recovered;
    throw originalError;
  }
}

$('connectRemote').addEventListener('click', () => invoke(
  $('connectRemote'), 'Connecting to the public read-only observation through the local backend…', async () => {
    const token = $('token').value;
    if (!token) throw new Error('Token cannot be empty');
    try {
      return await api('/api/remote/connect', 'POST', {
        endpoint: value('endpoint'), session_id: value('remoteSession'), token,
        tls_ca: value('tlsCa'), tls_certificate: value('tlsCert'), tls_private_key: value('tlsKey'),
      });
    } finally {
      $('token').value = '';
      $('tlsKey').value = '';
    }
  },
));

$('uploadArchive').addEventListener('click', () => invoke(
  $('uploadArchive'), 'Uploading, validating, and loading the local .tar.zst…', async () => {
    const file = $('archiveFile').files?.[0];
    if (!file) throw new Error('Select the participant submission .tar.zst file first');
    if (!file.name.endsWith('.tar.zst')) throw new Error('The filename must end with .tar.zst');
    const upload = await uploadArchiveFile(file, value('uploadSha'));
    const result = upload.images?.length
      ? upload
      : await waitForArchiveLoad(
          upload.job_id,
          message => setOperation(message, 'warn'),
        );
    if (result.images?.length) rememberImage(result.images[0]);
    $('uploadProgress').textContent = `Complete · ${(upload.upload_size_bytes / 1024 ** 3).toFixed(2)} GiB · SHA-256 ${upload.sha256}`;
    return result;
  },
));

$('loadArchive').addEventListener('click', () => invoke(
  $('loadArchive'), 'Validating SHA-256 and zstd integrity, then loading the image…', async () => {
    const result = await loadArchiveWithPolling({
      archive_path: value('archivePath'), sha256: value('archiveSha'),
    });
    if (result.images?.length) rememberImage(result.images[0]);
    return result;
  },
));

$('startPolicy').addEventListener('click', () => invoke(
  $('startPolicy'), 'Creating the isolated network, Zenoh router, and sandboxed container…',
  async () => {
    let image = value('image');
    const status = await api('/api/status');
    if (status.policy?.connected && status.policy?.image) {
      rememberImage(status.policy.image);
      return {already_running: true, status};
    }
    if (!image && status.container?.running && status.container?.image) {
      image = status.container.image;
      rememberImage(image);
    }
    if (!image) {
      throw new Error('Load a submission archive first, or enter the name of an image already loaded into local Docker');
    }
    return startPolicyWithRecovery(image);
  },
));
$('stopPolicy').addEventListener('click', () => invoke(
  $('stopPolicy'), 'Stopping and cleaning up the local container…',
  () => api('/api/submission/stop', 'POST', {}),
));
$('resetPolicy').addEventListener('click', () => invoke(
  $('resetPolicy'), 'Resetting the policy episode…', async () => {
    const result = await api('/api/policy/reset', 'POST', {});
    setTrajectory(null);
    return result;
  },
));
$('shadow').addEventListener('click', () => invoke(
  $('shadow'), 'Fetching one real observation and running Shadow locally only…', async () => {
    const result = await shadowWithRecovery({
      preview_steps: Number(value('previewSteps') || 100),
      control_hz: Number(value('controlHz') || 30),
    });
    setTrajectory(result);
    return result;
  },
));

async function refreshStatus() {
  try {
    const status = await api('/api/status');
    if (!value('image') && status.policy?.image) rememberImage(status.policy.image);
    badge(
      ui.remoteBadge,
      status.remote?.connected ? `Remote ${status.remote.session_id}` : 'Remote disconnected',
      status.remote?.connected ? 'ok' : 'bad',
    );
    badge(
      ui.policyBadge,
      status.policy?.connected ? `Policy ${status.policy.metadata?.action_horizon ?? '?'} steps/chunk` : 'Policy not started',
      status.policy?.connected ? 'ok' : 'bad',
    );
    ui.statusText.textContent = JSON.stringify({
      last_operation: status.last_operation,
      last_error: status.last_error,
      remote: status.remote,
      policy: status.policy,
      container: status.container,
    }, null, 2);
    if (status.observation_available) refreshObservation();
    refreshLogs();
  } catch (error) {
    badge(ui.remoteBadge, error.message, 'bad');
  }
}

async function refreshObservation() {
  try {
    const observation = await api('/api/observation');
    model.current = observation.state;
    renderState(observation);
    if (model.mode === 'current') renderRobot(model.current);
    model.imageVersion += 1;
    $('headLeft').src = `/api/image/head-left?v=${model.imageVersion}`;
    $('headRight').src = `/api/image/head-right?v=${model.imageVersion}`;
    $('leftWrist').src = `/api/image/left-wrist?v=${model.imageVersion}`;
    $('rightWrist').src = `/api/image/right-wrist?v=${model.imageVersion}`;
  } catch { /* status polling will expose persistent failures */ }
}

async function refreshLogs() {
  try {
    const logs = await api('/api/logs');
    ui.logs.textContent = `${(logs.events || []).join('\n')}\n\n${logs.container || ''}`.slice(-100000);
  } catch { /* best effort */ }
}

function renderState(observation) {
  $('prompt').textContent = observation.prompt || '—';
  const fragment = document.createDocumentFragment();
  observation.joint_names.forEach((name, index) => {
    const row = document.createElement('tr');
    for (const text of [index, name, Number(observation.state[index]).toFixed(6)]) {
      const cell = document.createElement('td');
      cell.textContent = String(text);
      row.append(cell);
    }
    fragment.append(row);
  });
  ui.stateRows.replaceChildren(fragment);
}

function metric(label, value) {
  const row = document.createElement('div');
  row.className = 'metric';
  const name = document.createElement('span');
  const content = document.createElement('span');
  name.textContent = label;
  content.textContent = value === null || value === undefined ? '—' : String(value);
  row.append(name, content);
  return row;
}

function renderMetrics(result) {
  const rows = result ? [
    ['Compatible', result.compatible ? 'YES' : 'NO'],
    ['Validation', result.validation?.validation_level],
    ['Shape', JSON.stringify(result.action_shape)],
    ['Chunks', result.chunk_count],
    ['Latency', `${result.latency_ms} ms`],
    ['Range', `${result.action_min?.toFixed(4)} … ${result.action_max?.toFixed(4)}`],
    ['URDF limits', model.config?.urdf_limits_loaded ? 'position + velocity' : 'unavailable'],
  ] : [['Trajectory', 'Not run yet']];
  ui.metrics.replaceChildren(...rows.map(([name, item]) => metric(name, item)));
}

function setTrajectory(result) {
  stopPlaying();
  model.prediction = Array.isArray(result?.prediction) ? result.prediction.slice(0, 100) : [];
  model.validation = result?.validation || null;
  if (Array.isArray(result?.current_state)) model.current = result.current_state;
  model.step = 0;
  model.mode = model.prediction.length ? 'predicted' : 'current';
  ui.timeline.max = String(Math.max(0, model.prediction.length - 1));
  ui.timeline.value = '0';
  const enabled = model.prediction.length > 0;
  for (const node of [ui.timeline, ui.playPause, ui.stepBack, ui.stepForward]) node.disabled = !enabled;
  renderMetrics(result);
  updatePlayer();
  if (enabled) startPlaying();
}

function selectedState() {
  return model.mode === 'predicted' && model.prediction.length
    ? model.prediction[model.step]
    : model.current;
}

function setStep(step) {
  if (!model.prediction.length) return;
  model.step = Math.max(0, Math.min(Number(step) || 0, model.prediction.length - 1));
  model.mode = 'predicted';
  ui.timeline.value = String(model.step);
  updatePlayer();
}

function updatePlayer() {
  ui.stepText.textContent = model.mode === 'predicted'
    ? `Predicted ${model.step + 1} / ${model.prediction.length}`
    : `Current · ${model.prediction.length} steps`;
  ui.stepBack.disabled = !model.prediction.length || model.step <= 0;
  ui.stepForward.disabled = !model.prediction.length || model.step >= model.prediction.length - 1;
  renderRobot(selectedState());
  renderWarnings();
}

function startPlaying() {
  if (!model.prediction.length) return;
  stopPlaying();
  model.playing = true;
  ui.playPause.textContent = '❚❚';
  if (model.step >= model.prediction.length - 1) model.step = 0;
  model.timer = window.setInterval(() => {
    if (model.step >= model.prediction.length - 1) { stopPlaying(); return; }
    setStep(model.step + 1);
  }, 200);
}

function stopPlaying() {
  model.playing = false;
  ui.playPause.textContent = '▶';
  if (model.timer !== null) window.clearInterval(model.timer);
  model.timer = null;
}

$('showCurrent').addEventListener('click', () => { stopPlaying(); model.mode = 'current'; updatePlayer(); });
ui.playPause.addEventListener('click', () => model.playing ? stopPlaying() : startPlaying());
ui.stepBack.addEventListener('click', () => { stopPlaying(); setStep(model.step - 1); });
ui.stepForward.addEventListener('click', () => { stopPlaying(); setStep(model.step + 1); });
ui.timeline.addEventListener('input', event => { stopPlaying(); setStep(event.target.value); });

function renderWarnings() {
  const reports = model.validation?.step_reports;
  const report = Array.isArray(reports) && model.mode === 'predicted' ? reports[model.step] : null;
  const issues = report?.violations || [];
  const fragment = document.createDocumentFragment();
  if (!issues.length) {
    const item = document.createElement('li');
    item.textContent = model.validation ? 'None' : 'No validation result yet';
    fragment.append(item);
  } else {
    issues.slice(0, 100).forEach(issue => {
      const item = document.createElement('li');
      item.textContent = `${issue.joint_name || issue.field || 'trajectory'}: ${issue.type || issue.message}`;
      fragment.append(item);
    });
  }
  ui.warnings.replaceChildren(fragment);
}

async function loadUrdf() {
  try {
    const config = await api('/api/robot/config');
    model.config = config;
    if (!config.urdf_available) throw new Error(config.urdf_error || 'Official URDF is unavailable');
    await loadUrdfModel(config);
  } catch (error) {
    badge(ui.urdfBadge, error.message, 'bad');
    badge(ui.meshBadge, 'Mesh loading failed', 'bad');
  }
}

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07101b);
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
  camera.position.set(2.6, 1.8, 3.2);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  ui.robotView.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0.8, 0);
  scene.add(new THREE.HemisphereLight(0xd7e8ff, 0x26313d, 2.4));
  const key = new THREE.DirectionalLight(0xffffff, 3);
  key.position.set(4, 7, 5);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x72aaff, 1.4);
  rim.position.set(-5, 3, -4);
  scene.add(rim);
  scene.add(new THREE.GridHelper(12, 24, 0x36506f, 0x1a2a3d));
  const resize = () => {
    const width = Math.max(1, ui.robotView.clientWidth);
    const height = Math.max(1, ui.robotView.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(ui.robotView);
  resize();
  requestAnimationFrame(renderFrame);
}

function renderFrame() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(renderFrame);
}

function loadUrdfModel(config) {
  return new Promise((resolve, reject) => {
    const url = new URL(config.urdf_url, window.location.href).href;
    const manager = new THREE.LoadingManager();
    manager.onStart = () => badge(ui.meshBadge, 'Loading mesh…', 'warn');
    manager.onProgress = (_url, loaded, total) => badge(ui.meshBadge, `Mesh ${loaded}/${total}`, 'warn');
    manager.onLoad = () => badge(ui.meshBadge, 'Mesh ready', 'ok');
    manager.onError = () => badge(ui.meshBadge, 'Some meshes failed to load', 'bad');
    const loader = new URDFLoader(manager);
    const urdfBase = new URL('.', url);
    loader.packages = urdfBase.pathname.endsWith('/urdf/')
      ? new URL('../', urdfBase).href
      : urdfBase.href;
    loader.load(url, robot => {
      if (model.robot) scene.remove(model.robot);
      model.robot = robot;
      robot.rotation.x = -Math.PI / 2;
      robot.traverse(object => {
        if (object.isMesh) {
          object.castShadow = true;
          object.receiveShadow = true;
        }
      });
      scene.add(robot);
      fitRobot(robot);
      badge(
        ui.urdfBadge,
        config.urdf_limits_loaded
          ? `URDF ready · ${Object.keys(robot.joints || {}).length} joints · full limit checks`
          : `URDF ready · ${Object.keys(robot.joints || {}).length} joints · shape/finite only`,
        config.urdf_limits_loaded ? 'ok' : 'warn',
      );
      renderRobot(selectedState());
      resolve();
    }, undefined, error => reject(error instanceof Error ? error : new Error('URDF loading failed')));
  });
}

function fitRobot(robot) {
  const box = new THREE.Box3().setFromObject(robot);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 0.5);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(radius * 1.8, radius * 1.15, radius * 2.1));
  camera.near = Math.max(radius / 1000, 0.005);
  camera.far = Math.max(radius * 100, 100);
  camera.updateProjectionMatrix();
  controls.update();
}

function renderRobot(state) {
  if (!model.robot || !Array.isArray(state)) return;
  for (const entry of model.config?.joint_map || []) {
    const value = Number(state[entry.index]);
    const joint = model.robot.joints?.[entry.name];
    if (!joint || !Number.isFinite(value) || typeof joint.setJointValue !== 'function') continue;
    try { joint.setJointValue(value); } catch { /* validation reports incompatible values */ }
  }
}

renderMetrics(null);
renderWarnings();
try { rememberImage(window.localStorage.getItem('origamiParticipantImage') || ''); }
catch { /* localStorage may be disabled */ }
initScene();
loadUrdf();
refreshStatus();
window.setInterval(refreshStatus, 1500);
