from pathlib import Path
import yaml

checkpoints = sorted(
    list(Path('./completed_runs').glob('*/*.ckpt')),
    key=lambda p: int(str(p.parent).split('/')[-1].split('_')[-1])
)
hparams = [Path(p.parent, 'hparams.yaml') for p in checkpoints]

heatmap_layers = {
    'Simple1DCNN': 'pool1 pool2 pool3',

}

commands = []
for checkpoint, hparam in zip(checkpoints, hparams):
    assert checkpoint.exists() and hparam.exists()
    with open(hparam, 'r') as stream:
        hparam_dict = yaml.safe_load(stream)
    model_name = hparam_dict['model_name']
    heatmap_layers = 'pool1 pool2 pool3' if model_name == 'Simple1DCNN' else 'layer2 layer3 layer4'
    updated_checkpoint = Path(checkpoint.parent, 'checkpoints', str(checkpoint).split('/')[-1])
    updated_checkpoint = str(updated_checkpoint).replace('completed_runs', 'lightning_logs')
    commands.append(
        f'python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint {updated_checkpoint} --model_name {model_name} --heatmap_layers {heatmap_layers}'
    )
    commands.append(
        "nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9"
    )

with open('./generate_heatmaps.sh', 'w') as f:
    f.write(" &&\n".join(commands))
