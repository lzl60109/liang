python -m vgks.download_d4rl_dataset --env-name halfcheetah-medium-v2 --output-dir data

python generate_vgks.py --config configs/vgks/config.yaml

python train_td3bc.py --config configs/offline_rl/td3bc.yaml
python train_iql.py --config configs/offline_rl/iql.yaml
python train_cql.py --config configs/offline_rl/cql.yaml
