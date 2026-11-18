for c in "face -1.0 0.1 1.0" "face -1.0 0.15 1.0" "face -1.0 0.2 0.5" "face -1.0 0.1 0.5" "side 1.0 0.1 0.5" "side 1.0 0.17 0.5" "side 1.0 0.2 0.5 rough" "face -1.0 0.15 1.0 rough"; do
  set -- $c; python3 scratch_missions/orbit_probe.py $1 $2 $3 $4 90 ${5:-flat}; done
