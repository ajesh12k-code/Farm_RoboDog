import os, sys, time, json
sys.argv = ["x"]; exec(open("treesim_proto.py").read().split('if __name__ == "__main__":')[0])
lay = orchard("mango", 4, 8, 2.5, 2.5, 2.4)
m, b, _ = build(lay, {**DEF, "foliage_mesh": True})
d = mujoco.MjData(m); reset(m, d); mujoco.mj_forward(m, d)
out = {}
r = mujoco.Renderer(m, 120, 160, max_geom=20000)
opt = mujoco.MjvOption(); opt.geomgroup[VIS_LEAF] = 1; opt.geomgroup[VIS_PEST] = 1
def run(tag, cams):
    ts = []
    for k in range(12):
        for c in cams:
            t1 = time.perf_counter(); r.update_scene(d, c, opt)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0; r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            r.render(); ts.append(time.perf_counter() - t1)
    out[tag] = dict(ms_per_pass=round(1e3 * float(np.median(ts[4:])), 1), scene_ngeom=r.scene.ngeom)
cam = mujoco.MjvCamera(); cam.lookat[:] = [1.2, 1.25, 1.0]; cam.distance = 2.0; cam.elevation = -10
run("free_all", [cam])


# cull: alpha 0 for geoms of tree bodies (and static fruit) farther than 4 m from robot
rgba0 = m.geom_rgba.copy()
rob = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")]
far = 0
for g in range(m.ngeom):
    bid = m.geom_bodyid[g]
    if bid and m.body_weldid[bid] == 0 and np.linalg.norm(d.xpos[bid][:2] - rob[:2]) > 4.0:
        m.geom_rgba[g, 3] = 0; far += 1
t0 = time.perf_counter(); 
run("free_cull4m", [cam])
out["culled_geoms"] = far
# 360 video estimate: 320x240 orbit, culled
r2 = mujoco.Renderer(m, 240, 320, max_geom=20000); ts = []
for k in range(24):
    cam.azimuth = 15 * k; t1 = time.perf_counter(); r2.update_scene(d, cam, opt)
    r2.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0; r2.render(); px = r2.render(); ts.append(time.perf_counter() - t1)
out["orbit_320x240_cull_2render"] = round(1e3 * float(np.median(ts[2:])), 1)
# segmentation pass cost (ground truth for pests/fruit)
r.enable_segmentation_rendering(); ts = []
for k in range(10):
    t1 = time.perf_counter(); r.update_scene(d, cam, opt); seg = r.render(); ts.append(time.perf_counter() - t1)
out["seg_160x120_cull"] = round(1e3 * float(np.median(ts[2:])), 1)
print(json.dumps(out, indent=1))
