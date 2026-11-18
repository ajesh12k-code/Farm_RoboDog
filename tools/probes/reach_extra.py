import json, sys
sys.argv=[sys.argv[0]]
from tools.probes import reach_proto as R
for kw in [dict(variant="none", slope=10), dict(variant="none", slope=5), dict(variant="mast", ext=1.6, slope=5), dict(variant="mast", ext=0.0, slope=5),
           dict(variant="mast", ext=1.6, slope=5, walk=0.3, secs=6.0), dict(variant="none", slope=10, walk=0.6, secs=6.0), dict(variant="none", slope=0, secs=20.0), dict(variant="none", slope=10, secs=20.0)]:
    r = R.run(**kw); print(json.dumps({k: r.get(k) for k in ("variant","ext","slope","walk","com_h","hexa_com_margin","tri_A","tri_B","max_tilt","fell","trunk_disp")}), flush=True)
