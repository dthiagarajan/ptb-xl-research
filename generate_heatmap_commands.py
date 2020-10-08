from pathlib import Path
import yaml

checkpoints = list(Path('./completed_runs').glob('*/*.ckpt'))
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
    commands.append(
        f'python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint {checkpoint} --model_name {model_name} --heatmap_layers {heatmap_layers}'
    )

with open('./generate_heatmaps.sh', 'w') as f:
    f.write('\n'.join(commands))
