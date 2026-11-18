for c in "side 1.0 0.15 1.0" "side 1.0 0.2 1.0" "side 1.0 0.3 0.5" "side 1.0 0.15 0.5" "side 0.5 0.15 1.0" "face 1.0 0.3 1.0" "face 1.0 0.5 1.0" "face 1.0 0.2 1.0" "face 1.0 0.4 0.5" "side 1.0 0.2 1.0 rough" "face 1.0 0.3 1.0 rough" "side 1.0 0.2 1.0 ridges"; do
  set -- $c; python3 scratch_missions/orbit_probe.py $1 $2 $3 $4 60 ${5:-flat}; done
