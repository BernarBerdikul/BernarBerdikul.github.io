// Hero star field: stars are strewn along the contours of an SVG shape, drift along them, twinkle,
// and turn in 3D while the visitor drags the stage. Any SVG works, presets are drawn from formulas.
import * as THREE from 'three';
import { SVGLoader } from 'three/addons/loaders/SVGLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const TAU = Math.PI * 2;
const { clamp } = THREE.MathUtils;
const VIEW_HEIGHT = 10; // world units across the hero's height
const REFERENCE = 6; // shapes are built with their shorter side this long, then scaled to the stage every frame
const MAX_CONTOURS = 64; // one draw call per contour
const PATH_SAMPLES = 256;
const FONT_URL = 'https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/DejaVuSans-Bold.ttf'; // covers Cyrillic and Kazakh
const PALETTE = ['#FFC46B', '#7CC8FF', '#A8B4FF', '#FF8F6B', '#F4F7FF'].map((hex) => new THREE.Color(hex));
const PALETTE_WEIGHTS = [0.22, 0.3, 0.14, 0.1, 0.24];

// ---------- Shapes ----------
const svgOf = (paths) => `<svg xmlns="http://www.w3.org/2000/svg">${paths.map((d) => `<path d="${d}"/>`).join('')}</svg>`;
function polyline(n, t0, t1, f, closed = true) {
  let d = '';
  for (let k = 0; k <= n; k++) { const [x, y] = f(t0 + ((t1 - t0) * k) / n, k); d += `${k ? 'L' : 'M'}${x.toFixed(3)} ${y.toFixed(3)}`; }
  return closed ? `${d}Z` : d;
}
// Logarithmic spiral arm, from the core outwards.
const spiralArm = (k, arms) => polyline(160, 0, 1, (u) => {
  const r = 7 + 93 * u, a = (k * TAU) / arms + Math.log(r / 7) / Math.tan(0.34);
  return [r * Math.cos(a), r * Math.sin(a)];
}, false);
// SVG y points down, hence the minus signs.
const heart = (s) => polyline(240, 0, TAU, (t) => [16 * s * Math.sin(t) ** 3, -s * (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t) + 2.5)]);
const lemniscateLoop = (a, t0) => polyline(160, t0, t0 + Math.PI, (t) => { const q = 1 + Math.sin(t) ** 2; return [(a * Math.cos(t)) / q, (a * Math.sin(t) * Math.cos(t)) / q]; });
const ellipse = (rx, ry, turn) => polyline(200, 0, TAU, (t) => {
  const x = rx * Math.cos(t), y = ry * Math.sin(t);
  return [x * Math.cos(turn) - y * Math.sin(turn), x * Math.sin(turn) + y * Math.cos(turn)];
});
// Airliner seen from above, nose up: the right half, mirrored into the left.
const PLANE_HALF = [[0, -50], [2.6, -47], [4.2, -41], [5, -30], [5, -13], [44, 7], [44, 13], [5, 5], [4.4, 28], [17, 38], [17, 43], [3, 40.5], [0, 47]];
const plane = () => {
  const left = PLANE_HALF.slice(1, -1).reverse().map(([x, y]) => [-x, y]);
  return `M${[...PLANE_HALF, ...left].map(([x, y]) => `${x} ${y}`).join('L')}Z`;
};

// scatter and depth are world units at build size, so they stay proportional once the shape is scaled.
const SHAPES = {
  galaxy: { svg: svgOf([0, 1, 2, 3].map((k) => spiralArm(k, 4))), core: 360, scatter: 0.42, depth: 0.45, tilt: -0.55, spin: 0.06 },
  plane: { svg: svgOf([plane()]), scatter: 0.14, depth: 0.25 },
  heart: { svg: svgOf([heart(1), heart(0.62)]), scatter: 0.26, depth: 0.5 },
  infinity: { svg: svgOf([lemniscateLoop(100, -Math.PI / 2), lemniscateLoop(100, Math.PI / 2)]), scatter: 0.3, depth: 0.5 },
  atom: { svg: svgOf([0, 1, 2].map((k) => ellipse(100, 34, (k * Math.PI) / 3))), core: 160, scatter: 0.2, depth: 0.4 },
  text: { scatter: 0.16, depth: 0.25 },
  svg: { scatter: 0.18, depth: 0.35 },
};

function readContours(svg) {
  // Chrome's DOMParser keeps the tree up to an XML error, so a broken SVG would otherwise be drawn in part without a word.
  const error = new DOMParser().parseFromString(svg, 'image/svg+xml').querySelector('parsererror');
  if (error) throw new Error(`the SVG does not parse: ${(error.querySelector('div') ?? error).textContent.trim()}`);
  const contours = [];
  for (const path of new SVGLoader().parse(svg).paths) {
    for (const source of path.subPaths) {
      if (!source.curves.length) continue;
      const gap = source.curves[0].getPoint(0).distanceTo(source.curves.at(-1).getPoint(1));
      if (source.autoClose && gap > 1e-6) source.closePath();
      const length = source.getLength();
      if (length > 0 && Number.isFinite(length)) contours.push({ source, length, closed: source.autoClose || gap < 1e-3 * length });
    }
  }
  return contours;
}
function frameOf(contours) {
  const box = new THREE.Box2(), size = new THREE.Vector2();
  for (const c of contours) for (let k = 0; k <= 64; k++) box.expandByPoint(c.source.getPointAt(k / 64));
  box.getSize(size);
  const scale = REFERENCE / Math.max(Math.min(size.x, size.y), 0.05 * Math.max(size.x, size.y));
  return { cx: (box.min.x + box.max.x) / 2, cy: (box.min.y + box.max.y) / 2, scale, size: size.multiplyScalar(scale) };
}

// FNV-1a, so every contour of every shape gets its own stable seed.
function seedOf(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
  return h >>> 0;
}
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function pickColor(r) {
  let acc = 0;
  for (let k = 0; k < PALETTE.length; k++) { acc += PALETTE_WEIGHTS[k]; if (r < acc) return PALETTE[k]; }
  return PALETTE.at(-1);
}

// ---------- Shaders ----------
const VERTEX = /* glsl */ `
uniform sampler2D uPath; uniform float uHasPath; uniform float uClosed; uniform float uFlow;
uniform float uTime; uniform float uIntro; uniform float uPixelRatio; uniform float uPointScale;
uniform vec2 uPointer; uniform float uPush; uniform vec2 uScatterBox;
attribute float aT; attribute vec2 aOffset; attribute float aSize; attribute float aGlow; attribute vec3 aColor; attribute float aSeed;
varying vec3 vColor; varying float vAlpha; varying float vSpike;

float hash(float n) { return fract(sin(n * 91.3458) * 47453.5453); }
// Row 0 holds points along the contour, row 1 the unit normals worked out on the CPU.
vec3 pathRow(float t, float row) {
  float x = clamp(t, 0.0, 1.0) * ${PATH_SAMPLES - 1}.0;
  float i = floor(x);
  float v = (row + 0.5) / 2.0;
  vec3 a = texture2D(uPath, vec2((i + 0.5) / ${PATH_SAMPLES}.0, v)).xyz;
  vec3 b = texture2D(uPath, vec2((min(i + 1.0, ${PATH_SAMPLES - 1}.0) + 0.5) / ${PATH_SAMPLES}.0, v)).xyz;
  return mix(a, b, x - i);
}
void main() {
  vec3 p = position;
  float fade = 1.0;
  if (uHasPath > 0.5) {
    float t = fract(aT + uFlow);
    p = pathRow(t, 0.0) + pathRow(t, 1.0) * aOffset.x + vec3(0.0, 0.0, aOffset.y);
    // An open path has ends: stars thin out towards them and come back at the other end.
    if (uClosed < 0.5) fade = pow(sin(3.14159265 * t), 0.35);
  }
  // Intro: each star enters from a ring beyond the edges and arrives along its own curve.
  float angle = hash(aSeed) * 6.2831853;
  vec3 from = vec3(cos(angle), sin(angle), 0.0) * 0.62 * max(uScatterBox.x, uScatterBox.y) + vec3(0.0, 0.0, (hash(aSeed + 1.7) - 0.5) * 3.0);
  vec3 via = mix(from, p, 0.5) + vec3(-sin(angle), cos(angle), 0.0) * length(from.xy) * (hash(aSeed + 3.1) - 0.5) * 0.9;
  float delay = hash(aSeed + 5.3) * 0.45;
  float k = clamp((uIntro - delay) / (1.0 - delay), 0.0, 1.0);
  float e = k * k * (3.0 - 2.0 * k);
  p = mix(mix(from, via, e), mix(via, p, e), e);

  vec4 view = modelViewMatrix * vec4(p, 1.0);
  // A moving cursor ruffles nearby stars, measured on screen so every shape reacts the same.
  vec2 away = view.xy - uPointer;
  view.xy += normalize(away + vec2(1e-5)) * uPush * exp(-dot(away, away) * 2.6);
  gl_Position = projectionMatrix * view;

  float twinkle = 0.78 + 0.22 * sin(uTime * (0.7 + 1.5 * hash(aSeed + 9.1)) + aSeed * 6.2831);
  gl_PointSize = max(1.5, uPixelRatio * uPointScale * aSize * 2.6 * (0.88 + 0.12 * twinkle) * mix(0.35, 1.0, e));
  vColor = aColor;
  vAlpha = aGlow * twinkle * fade * smoothstep(0.0, 0.3, k);
  vSpike = step(2.5, aSize);
}`;

const FRAGMENT = /* glsl */ `
varying vec3 vColor; varying float vAlpha; varying float vSpike;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 16.0);
  float halo = exp(-r2 * 4.5) * 0.42;
  float spikes = vSpike * (exp(-abs(q.x) * 24.0) + exp(-abs(q.y) * 24.0)) * (1.0 - sqrt(r2)) * 0.55;
  float alpha = (core + halo + spikes) * vAlpha;
  gl_FragColor = vec4(mix(vColor, vec3(1.0), core * 0.65), alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

// ---------- Mount ----------
export function mountStarfield(hero) {
  const canvas = hero.querySelector('canvas.sky'), stage = hero.querySelector('.stage'), picker = hero.querySelector('.shapes');
  const coarse = matchMedia('(pointer: coarse)').matches, motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
  const STAR_BUDGET = coarse ? 2200 : 3600, SKY_STARS = coarse ? 260 : 520;
  let reduced = motionQuery.matches;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
  renderer.setClearColor(0x050a18, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const scene = new THREE.Scene(), camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  camera.position.set(0, 0, 40);
  const field = new THREE.Group(), rotor = new THREE.Group(), skyGroup = new THREE.Group();
  field.add(rotor); scene.add(skyGroup, field);

  let composer = null, bloom = null;
  if (!coarse) {
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.85, 0.55, 0.18);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
  }

  const common = {
    uTime: { value: 0 }, uIntro: { value: 0 }, uPixelRatio: { value: 1 }, uPointScale: { value: 1 },
    uPointer: { value: new THREE.Vector2(99, 99) }, uPush: { value: 0 }, uScatterBox: { value: new THREE.Vector2(20, 12) },
  };
  const material = (extra) => new THREE.ShaderMaterial({
    vertexShader: VERTEX, fragmentShader: FRAGMENT, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { ...common, uPath: { value: null }, uHasPath: { value: 0 }, uClosed: { value: 1 }, uFlow: { value: 0 }, ...extra },
  });
  function geometryOf(n, fill) {
    const g = new THREE.BufferGeometry(), a = {
      position: new Float32Array(3 * n), aT: new Float32Array(n), aOffset: new Float32Array(2 * n), aSize: new Float32Array(n),
      aGlow: new Float32Array(n), aColor: new Float32Array(3 * n), aSeed: new Float32Array(n) };
    for (let i = 0; i < n; i++) fill(i, a);
    const sizes = { position: 3, aOffset: 2, aColor: 3 };
    for (const [name, array] of Object.entries(a)) g.setAttribute(name, new THREE.BufferAttribute(array, sizes[name] ?? 1));
    return g;
  }
  function starLook(i, a, rnd, brightChance) {
    const bright = rnd() < brightChance;
    a.aSize[i] = bright ? 2.6 + 1.4 * rnd() : 0.7 + 1.4 * rnd() ** 2;
    a.aGlow[i] = bright ? 1 : 0.4 + 0.5 * rnd();
    pickColor(rnd()).toArray(a.aColor, 3 * i);
    a.aSeed[i] = rnd() * 1000;
  }

  // Sky: faint stars across the whole hero that never turn with the shape.
  const skyRnd = mulberry32(seedOf('sky'));
  const sky = new THREE.Points(geometryOf(SKY_STARS, (i, a) => {
    a.position.set([skyRnd() - 0.5, skyRnd() - 0.5, skyRnd() - 0.5], 3 * i);
    starLook(i, a, skyRnd, 0.03);
    a.aSize[i] *= 0.7; a.aGlow[i] *= 0.6;
  }), material({ uIntro: { value: 1 }, uPush: { value: 0 } }));
  sky.frustumCulled = false; skyGroup.add(sky);

  // ---------- Field: rebuilt from scratch whenever the shape changes ----------
  const chains = [];
  let shape = null;
  function disposeField() {
    rotor.traverse((o) => { if (o.isPoints) { o.geometry.dispose(); o.material.uniforms.uPath.value?.dispose(); o.material.dispose(); } });
    rotor.clear(); chains.length = 0;
  }
  // Every contour sits in its own tilt and spin groups inside the rotor that the visitor turns.
  function addChain(points, flowSpeed, spec) {
    const tilt = new THREE.Group(), spin = new THREE.Group();
    tilt.rotation.x = spec.tilt ?? 0; points.frustumCulled = false;
    tilt.add(spin); spin.add(points); rotor.add(tilt);
    chains.push({ spin, flowSpeed, material: points.material });
  }
  function buildContour(i, { source, closed }, frame, spec, count) {
    const rnd = mulberry32(seedOf(`contour ${i}`));
    const data = new Float32Array(PATH_SAMPLES * 8), phase = rnd() * TAU, depth = spec.depth * (0.6 + 0.4 * rnd());
    for (let k = 0; k < PATH_SAMPLES; k++) {
      const t = k / (PATH_SAMPLES - 1), p = source.getPointAt(t), d = source.getTangentAt(t);
      // Whole periods only, so a closed contour meets itself again at the same depth.
      const z = depth * (0.65 * Math.sin(TAU * t + phase) + 0.35 * Math.sin(2 * TAU * t + 1.7 * phase));
      const length = Math.hypot(d.x, d.y) || 1;
      data.set([(p.x - frame.cx) * frame.scale, (frame.cy - p.y) * frame.scale, z, 1], 4 * k);
      // SVG y points down, so the tangent's y flips before it turns into the left-hand normal.
      data.set([d.y / length, d.x / length, 0, 1], 4 * (PATH_SAMPLES + k));
    }
    const texture = new THREE.DataTexture(data, PATH_SAMPLES, 2, THREE.RGBAFormat, THREE.FloatType);
    texture.needsUpdate = true;
    const geometry = geometryOf(count, (n, a) => {
      a.aT[n] = rnd();
      a.aOffset[2 * n] = (rnd() + rnd() - 1) * spec.scatter * (0.3 + 0.7 * rnd());
      a.aOffset[2 * n + 1] = (rnd() + rnd() - 1) * spec.scatter * 0.5;
      starLook(n, a, rnd, 0.07);
    });
    const speed = (0.01 + 0.006 * (i % 3)) * (closed && i % 2 ? -1 : 1);
    addChain(new THREE.Points(geometry, material({ uPath: { value: texture }, uHasPath: { value: 1 }, uClosed: { value: closed ? 1 : 0 } })), speed, spec);
  }
  function buildCore(count, spec) {
    const rnd = mulberry32(seedOf('core'));
    const geometry = geometryOf(count, (i, a) => {
      const r = REFERENCE * 0.07 * Math.sqrt(-2 * Math.log(1 - rnd() * 0.999)), angle = rnd() * TAU;
      a.position.set([Math.cos(angle) * r, Math.sin(angle) * r, (rnd() - 0.5) * r * 0.5], 3 * i);
      starLook(i, a, rnd, 0.05);
      if (r < REFERENCE * 0.05) PALETTE[rnd() < 0.5 ? 0 : 4].toArray(a.aColor, 3 * i);
      a.aGlow[i] = Math.min(1, a.aGlow[i] * 1.3);
    });
    addChain(new THREE.Points(geometry, material({})), 0, spec);
  }
  // Returns how many contours were left out by the MAX_CONTOURS cap.
  function buildField(spec) {
    let contours = readContours(spec.svg);
    if (!contours.length) throw new Error('no contours found in this SVG');
    const found = contours.length;
    if (found > MAX_CONTOURS) contours = contours.sort((a, b) => b.length - a.length).slice(0, MAX_CONTOURS);
    const frame = frameOf(contours), total = contours.reduce((s, c) => s + c.length, 0);
    const budget = STAR_BUDGET - (spec.core ? Math.round(spec.core * (STAR_BUDGET / 3600)) : 0);
    disposeField();
    shape = { ...spec, size: frame.size };
    contours.forEach((c, i) => buildContour(i, c, frame, spec, Math.max(24, Math.round((budget * c.length) / total))));
    if (spec.core) buildCore(Math.round(spec.core * (STAR_BUDGET / 3600)), spec);
    return found - contours.length;
  }

  // ---------- Layout: the shape sits in the stage element, the sky fills the hero ----------
  const view = { w: 1, h: 1, W: 1, stage: { x: 0, y: 0, w: 1, h: 1 } };
  function resize() {
    const rect = hero.getBoundingClientRect(), w = Math.max(1, Math.round(rect.width)), h = Math.max(1, Math.round(rect.height));
    const dpr = Math.min(devicePixelRatio || 1, 2, Math.sqrt(2.6e6 / (w * h)));
    renderer.setPixelRatio(dpr); renderer.setSize(w, h, false);
    if (composer) { composer.setPixelRatio(dpr); composer.setSize(w, h); }
    const W = (VIEW_HEIGHT * w) / h, s = stage.getBoundingClientRect();
    Object.assign(camera, { left: -W / 2, right: W / 2, top: VIEW_HEIGHT / 2, bottom: -VIEW_HEIGHT / 2 });
    camera.updateProjectionMatrix();
    view.w = w; view.h = h; view.W = W;
    view.stage = { x: ((s.left + s.width / 2 - rect.left) / w - 0.5) * W, y: (0.5 - (s.top + s.height / 2 - rect.top) / h) * VIEW_HEIGHT,
      w: (s.width / w) * W, h: (s.height / h) * VIEW_HEIGHT };
    skyGroup.scale.set(W * 1.05, VIEW_HEIGHT * 1.05, 2);
    common.uPixelRatio.value = dpr;
    invalidate();
  }

  // ---------- Input: dragging throws the shape around, a spring brings it back to rest ----------
  const rot = new THREE.Vector2(), vel = new THREE.Vector2(), pointer = { x: 0, y: 0 };
  let drag = null, stir = 0;
  const toWorld = (e) => {
    const r = hero.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width - 0.5) * view.W, y: (0.5 - (e.clientY - r.top) / r.height) * VIEW_HEIGHT };
  };
  const endDrag = () => {
    if (!drag) return;
    try { stage.releasePointerCapture(drag.id); } catch { /* capture already gone */ }
    drag = null; stage.dataset.dragging = 'false';
    if (vel.length() > 10) vel.setLength(10);
    invalidate();
  };
  stage.addEventListener('pointerdown', (e) => {
    if (!e.isPrimary || e.button !== 0 || drag) return;
    try { stage.setPointerCapture(e.pointerId); } catch { return; }
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, at: e.timeStamp }; vel.set(0, 0); stage.dataset.dragging = 'true';
  });
  stage.addEventListener('pointermove', (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const pitch = (e.clientY - drag.y) * 0.007, yaw = (e.clientX - drag.x) * 0.007, seconds = Math.max(0.008, (e.timeStamp - drag.at) / 1000);
    rot.x += pitch; rot.y += yaw;
    if (!reduced) vel.lerp({ x: pitch / seconds, y: yaw / seconds }, 0.35);
    Object.assign(drag, { x: e.clientX, y: e.clientY, at: e.timeStamp });
    invalidate();
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) stage.addEventListener(type, (e) => { if (drag?.id === e.pointerId) endDrag(); });
  stage.addEventListener('keydown', (e) => {
    const step = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
    if (step) {
      e.preventDefault();
      if (reduced) { rot.x += step[0] * 0.15; rot.y += step[1] * 0.15; } else { vel.x += step[0] * 2.2; vel.y += step[1] * 2.2; }
      invalidate();
    }
    if (e.key === 'Escape' || e.key === 'Home') { rot.set(0, 0); vel.set(0, 0); invalidate(); }
  });
  hero.addEventListener('pointerenter', (e) => Object.assign(pointer, toWorld(e)));
  hero.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse') return;
    const p = toWorld(e);
    // Only movement stirs the stars, so a resting cursor never leaves a hole in the shape.
    if (!drag && !reduced) stir = Math.min(1, stir + Math.hypot(p.x - pointer.x, p.y - pointer.y) * 0.8);
    Object.assign(pointer, p); invalidate();
  });
  addEventListener('blur', endDrag);

  // ---------- Frame loop: runs only while the hero is on screen and the tab is visible ----------
  const state = { time: 0, intro: 0, introSeconds: 3.2 };
  let raf = 0, last = 0, onScreen = true, dead = false;
  function invalidate() { if (!raf && !dead) raf = requestAnimationFrame(frame); }
  function frame(now) {
    raf = 0;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    const moving = update(dt);
    (composer ?? renderer).render(scene, camera);
    if (!hero.dataset.starfield) hero.dataset.starfield = 'ready';
    if (onScreen && !document.hidden && (!reduced || moving)) invalidate();
    else last = 0;
  }
  function update(dt) {
    if (reduced) { state.intro = 1; } else { state.time += dt; state.intro = Math.min(1, state.intro + dt / state.introSeconds); }
    const { stage: st, h } = view;
    const fit = Math.min((0.9 * st.h) / Math.max(shape.size.y, 1e-6), (0.9 * st.w) / Math.max(shape.size.x, 1e-6));
    field.position.set(st.x, st.y, 0);
    field.scale.setScalar(fit);
    // Sprites have a fixed pixel size, so a shape drawn small gets smaller stars instead of a glowing smear.
    const shortSidePx = (REFERENCE * fit * h) / VIEW_HEIGHT;
    common.uPointScale.value = clamp(shortSidePx / 260, 0.45, 1);
    common.uScatterBox.value.set(view.W / fit, VIEW_HEIGHT / fit);
    common.uTime.value = state.time; common.uIntro.value = state.intro;
    stir *= Math.exp(-3.5 * dt);
    common.uPush.value = 0.22 * stir; common.uPointer.value.set(pointer.x, pointer.y);
    // Once let go the shape keeps its momentum, then a slightly underdamped spring settles it facing forward.
    // With reduced motion it simply stays where the visitor left it.
    if (!drag && !reduced) {
      vel.x += (-7 * rot.x - 3.2 * vel.x) * dt; vel.y += (-7 * rot.y - 3.2 * vel.y) * dt;
      rot.addScaledVector(vel, dt);
    }
    const sway = reduced ? 0 : 0.14 * Math.sin(state.time * 0.35);
    rotor.rotation.set(rot.x, rot.y + sway, 0);
    for (const c of chains) {
      if (!reduced) { c.spin.rotation.z += dt * (shape.spin ?? 0); c.material.uniforms.uFlow.value = (c.material.uniforms.uFlow.value + dt * c.flowSpeed) % 1; }
    }
    const turning = !reduced && !drag && rot.lengthSq() + vel.lengthSq() > 1e-6;
    return turning || state.intro < 1 || stir > 1e-3;
  }

  const observer = new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; if (onScreen) invalidate(); });
  observer.observe(hero);
  // The stage shrinks when the text or SVG field opens under it, without the hero changing size.
  const sizes = new ResizeObserver(resize); sizes.observe(hero); sizes.observe(stage);
  document.addEventListener('visibilitychange', () => { last = 0; invalidate(); });
  motionQuery.addEventListener('change', () => { reduced = motionQuery.matches; last = 0; invalidate(); });
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); dead = true; hero.classList.add('static'); });
  document.fonts?.ready.then(resize);

  // ---------- Shape picker ----------
  const note = picker.querySelector('.note');
  let fontLoad = null, picked = 'galaxy', ticket = 0;
  const loadFont = () => (fontLoad ??= (async () => {
    const [{ default: opentype }, response] = await Promise.all([import('opentype.js'), fetch(FONT_URL)]);
    if (!response.ok) throw new Error(`the font did not load (HTTP ${response.status})`);
    return opentype.parse(await response.arrayBuffer());
  })().catch((err) => { fontLoad = null; throw err; }));
  const wrapWords = (text, perLine) => text.split(/\s+/).filter(Boolean).reduce((lines, word) => {
    const lastLine = lines.at(-1);
    if (lastLine !== undefined && `${lastLine} ${word}`.length <= perLine) lines[lines.length - 1] = `${lastLine} ${word}`; else lines.push(word);
    return lines;
  }, []);
  // opentype.js turns glyph outlines into SVG path data, one centred path per line.
  async function textSvg(text) {
    const font = await loadFont();
    const missing = [...new Set(Array.from(text).filter((ch) => ch.trim() && font.charToGlyphIndex(ch) === 0))];
    if (missing.length) throw new Error(`the font has no ${missing.join(' ')}`);
    const lines = wrapWords(text, view.stage.w / view.stage.h < 0.9 ? 10 : 16);
    return svgOf(lines.map((line, i) => font.getPath(line, -font.getAdvanceWidth(line, 100) / 2, i * 120, 100).toPathData(2)));
  }
  // Inline SVG copied from a web page often lacks namespaces, and SVGLoader resolves <use> only through xlink:href.
  const customSvg = (value) => (value.startsWith('<') ? value : svgOf([value.replace(/^d\s*=\s*/, '').replace(/["']/g, '')]))
    .replace(/<svg\b(?![^>]*\sxmlns=)/, '<svg xmlns="http://www.w3.org/2000/svg"')
    .replace(/<svg\b(?![^>]*\sxmlns:xlink=)/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"')
    .replace(/(<use\b[^>]*?\s)href=/g, '$1xlink:href=');
  async function showShape(id) {
    const mine = ++ticket;
    picked = id;
    for (const box of picker.querySelectorAll('[data-for]')) box.hidden = box.dataset.for !== id;
    const text = picker.elements.text.value.trim(), svg = picker.elements.svg.value.trim();
    note.textContent = '';
    if (id === 'svg' && !svg) { picker.elements.svg.focus(); return; }
    try {
      const markup = id === 'text' ? await textSvg(text || 'Bernar') : id === 'svg' ? customSvg(svg) : SHAPES[id].svg;
      if (mine !== ticket) return; // a newer choice won while the font was loading
      const dropped = buildField({ ...SHAPES[id], svg: markup });
      for (const b of picker.querySelectorAll('[data-shape]')) b.setAttribute('aria-pressed', String(b.dataset.shape === id));
      if (dropped) note.textContent = `Showing the ${MAX_CONTOURS} longest contours, ${dropped} more left out.`;
      state.intro = 0; state.introSeconds = 2.2; rot.set(0, 0); vel.set(0, 0);
      invalidate();
    } catch (err) {
      if (mine === ticket) note.textContent = `Could not draw that: ${err.message}`;
    }
  }
  picker.addEventListener('click', (e) => { const b = e.target.closest('[data-shape]'); if (b) showShape(b.dataset.shape); });
  picker.addEventListener('submit', (e) => { e.preventDefault(); showShape(picked); });

  buildField({ ...SHAPES.galaxy });
  resize();
  hero.classList.add('live');
}
