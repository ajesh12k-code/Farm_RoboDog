// Three.js view of the MuJoCo farm world. MuJoCo is z-up, so the whole scene is kept z-up
// (camera.up = +z) and MuJoCo coordinates are used verbatim; only quaternions are reordered
// (MuJoCo w,x,y,z -> three x,y,z,w). Capsules/cylinders are along local z in MuJoCo but
// along y in three, hence the rotateX(pi/2) on their geometry.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SUN_OFFSET = new THREE.Vector3(3.5, -2.5, 6.0);
const BASKET_GLOW = new THREE.Color(0x2bd46b);
const NO_GLOW = new THREE.Color(0x000000);

function canvasTexture(size, paint) {
  const c = document.createElement('canvas');
  c.width = size[0];
  c.height = size[1];
  paint(c.getContext('2d'), c.width, c.height);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function skyTexture() {
  return canvasTexture([2, 256], (ctx, w, h) => {
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0.0, '#5d93c9');
    grad.addColorStop(0.55, '#a9cbe6');
    grad.addColorStop(1.0, '#e9e2cf');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  });
}

function soilTexture() {
  const tex = canvasTexture([256, 256], (ctx, w, h) => {
    ctx.fillStyle = '#5a4330';
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 2600; i++) {  // clods and grit
      const shade = 40 + Math.floor(Math.random() * 60);
      ctx.fillStyle = `rgba(${shade + 40}, ${shade + 18}, ${shade}, ${0.18 + Math.random() * 0.3})`;
      const r = 0.6 + Math.random() * 2.4;
      ctx.beginPath();
      ctx.arc(Math.random() * w, Math.random() * h, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = 'rgba(30, 20, 12, 0.22)';  // faint furrows along x
    ctx.lineWidth = 6;
    for (let y = 16; y < h; y += 64) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  });
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

function rgbaMaterial(rgba) {
  const [r, g, b, a] = rgba;
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace),
    roughness: 0.75,
    metalness: 0.08,
    transparent: a < 0.999,
    opacity: a,
  });
}

/** One MuJoCo geom -> a Mesh in its body's frame (null for unsupported types). */
function geomMesh(g, soil, maxAniso) {
  const [s0, s1, s2] = g.size;
  let geo;
  let mat = null;
  switch (g.type) {
    case 'box': geo = new THREE.BoxGeometry(2 * s0, 2 * s1, 2 * s2); break;
    case 'sphere': geo = new THREE.SphereGeometry(s0, 24, 16); break;
    case 'ellipsoid': geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s0, s1, s2); break;
    case 'capsule': geo = new THREE.CapsuleGeometry(s0, 2 * s1, 6, 16); geo.rotateX(Math.PI / 2); break;
    case 'cylinder': geo = new THREE.CylinderGeometry(s0, s0, 2 * s1, 24); geo.rotateX(Math.PI / 2); break;
    case 'plane': {
      const hx = s0 > 0 ? s0 : 60;  // MuJoCo size 0 = infinite plane
      const hy = s1 > 0 ? s1 : 60;
      geo = new THREE.PlaneGeometry(2 * hx, 2 * hy);  // lies in local xy, normal +z: same as MuJoCo
      const map = soil.clone();
      map.repeat.set(hx, hy);  // one 256px tile per 2 m
      map.anisotropy = maxAniso;
      map.needsUpdate = true;
      mat = new THREE.MeshStandardMaterial({ map, roughness: 1.0, metalness: 0.0 });
      break;
    }
    default: return null;
  }
  const mesh = new THREE.Mesh(geo, mat ?? rgbaMaterial(g.rgba));
  mesh.position.fromArray(g.pos);
  mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  mesh.castShadow = g.type !== 'plane';
  mesh.receiveShadow = true;
  mesh.name = g.name ?? '';
  return mesh;
}

function segmentLines(segments, color, opacity) {
  const pts = new Float32Array(segments.length * 6);
  segments.forEach((s, i) => pts.set([...s[0], ...s[1]], i * 6));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
}

function disposeTree(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) {
      o.material.map?.dispose();
      o.material.dispose();
    }
  });
}

export class SceneView {
  constructor(container) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);
    this.maxAniso = this.renderer.capabilities.getMaxAnisotropy();

    this.scene = new THREE.Scene();
    this.scene.background = skyTexture();
    this.scene.fog = new THREE.Fog(0xc9dbe6, 14, 48);
    this.soil = soilTexture();

    this.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.03, 200);
    this.camera.up.set(0, 0, 1);  // must precede OrbitControls, which caches the up axis
    this.camera.position.set(-1.7, -1.6, 1.15);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.25, 0, 0.3);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 25;
    this.controls.maxPolarAngle = Math.PI * 0.495;  // stay above the ground

    const hemi = new THREE.HemisphereLight(0xdfefff, 0x5a4630, 0.9);
    hemi.position.set(0, 0, 1);  // hemisphere "up" for a z-up world
    this.scene.add(hemi);
    this.sun = new THREE.DirectionalLight(0xfff3df, 2.2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 20 });
    this.sun.shadow.camera.updateProjectionMatrix();
    this.sun.shadow.bias = -0.0004;
    this.sunTarget = new THREE.Object3D();
    this.sun.target = this.sunTarget;
    this.scene.add(this.sun, this.sunTarget);

    this.root = null;
    this.bodyGroups = [];       // in the server's pose-stream order
    this.trunk = null;
    this.crops = new Map();     // crop name -> Mesh[]
    this.envelope = null;
    this.follow = true;
    this._lastTrunk = null;

    window.addEventListener('resize', () => this._resize());
    this.renderer.setAnimationLoop(() => this._frame());
  }

  /** Rebuild everything from a {"type":"scene"} message. */
  build(desc) {
    if (this.root) {
      this.scene.remove(this.root);
      disposeTree(this.root);
    }
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.envelope = null;

    const byId = new Map([[0, new THREE.Group()]]);  // world body: static, identity pose
    this.root.add(byId.get(0));
    this.bodyGroups = desc.bodies.map((b) => {
      const grp = new THREE.Group();
      grp.name = b.name ?? `body_${b.id}`;
      byId.set(b.id, grp);
      this.root.add(grp);  // poses are world-frame, so bodies are not nested
      return grp;
    });
    this.crops = new Map(desc.crops.map((n) => [n, []]));
    for (const g of desc.geoms) {
      const parent = byId.get(g.body);
      const mesh = parent && geomMesh(g, this.soil, this.maxAniso);
      if (!mesh) continue;
      parent.add(mesh);
      this.crops.get(parent.name)?.push(mesh);
    }
    this.trunk = byId.get(desc.trunk) ?? null;
    this._lastTrunk = null;
  }

  /** Flat [x y z qw qx qy qz] per streamed body, same order as scene.bodies. */
  applyPoses(p) {
    if (!Array.isArray(p) || p.length !== this.bodyGroups.length * 7) return;
    this.bodyGroups.forEach((grp, i) => {
      const k = i * 7;
      grp.position.set(p[k], p[k + 1], p[k + 2]);
      grp.quaternion.set(p[k + 4], p[k + 5], p[k + 6], p[k + 3]);
    });
    if (!this.trunk) return;
    const t = this.trunk.position;
    if (this.follow && this._lastTrunk) {
      const d = t.clone().sub(this._lastTrunk);
      this.camera.position.add(d);
      this.controls.target.add(d);
    }
    this._lastTrunk = t.clone();
    this.sunTarget.position.copy(t);
    this.sun.position.copy(t).add(SUN_OFFSET);
  }

  setFollow(on) {
    this.follow = on;
    this._lastTrunk = this.trunk ? this.trunk.position.clone() : null;
    if (on && this.trunk) {  // re-centre on the robot, keep the current viewing offset
      const d = this.trunk.position.clone().add(new THREE.Vector3(0.25, 0, 0)).sub(this.controls.target);
      this.camera.position.add(d);
      this.controls.target.add(d);
    }
  }

  /** Envelope segments are in the trunk frame, so the lines ride on the trunk group. */
  setEnvelope(data) {
    if (this.envelope) {
      this.envelope.parent?.remove(this.envelope);
      disposeTree(this.envelope);
      this.envelope = null;
    }
    if (!data || !this.trunk) return;
    this.envelope = new THREE.Group();
    this.envelope.add(segmentLines(data.canopy_segments ?? [], 0x26d9f2, 0.45));
    this.envelope.add(segmentLines(data.floor_segments ?? [], 0xff3838, 0.7));
    this.trunk.add(this.envelope);
  }

  markBasket(names) {
    const inBasket = new Set(Array.isArray(names) ? names : []);
    for (const [name, meshes] of this.crops) {
      const glow = inBasket.has(name) ? BASKET_GLOW : NO_GLOW;
      for (const m of meshes) m.material.emissive.copy(glow).multiplyScalar(0.45);
    }
  }

  _resize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  _frame() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
