
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py m3cad_carla_ue5 --root-path ./data/m3cad_carla_ue5 \
       --out-dir ./data/infos \
       --extra-tag m3cad_carla_ue5 \
       --version v1.0-mini \
       --canbus ./data/m3cad_carla_ue5 \
