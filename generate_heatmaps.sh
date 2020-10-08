python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_8/epoch=19.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_36/epoch=15.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_31/epoch=23.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_38/epoch=19.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_6/epoch=15.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_1/epoch=13.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_54/epoch=1.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_53/epoch=5.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_65/epoch=19.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_62/epoch=18.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_0/epoch=23.ckpt --model_name Simple1DCNN --heatmap_layers pool1 pool2 pool3 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_39/epoch=13.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_7/epoch=17.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_30/epoch=23.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_9/epoch=2.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_37/epoch=9.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_63/epoch=19.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_64/epoch=4.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_52/epoch=18.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_55/epoch=7.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_48/epoch=10.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_70/epoch=15.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_41/epoch=18.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_15/epoch=17.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_12/epoch=17.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_24/epoch=16.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_23/epoch=14.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_40/epoch=5.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_47/epoch=9.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_71/epoch=6.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_49/epoch=24.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_22/epoch=10.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_25/epoch=7.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_13/epoch=21.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_14/epoch=11.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_50/epoch=21.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_57/epoch=20.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_68/epoch=15.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_61/epoch=16.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_66/epoch=18.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_59/epoch=13.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_32/epoch=21.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_35/epoch=20.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_2/epoch=13.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_5/epoch=9.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_67/epoch=3.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_58/epoch=17.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_60/epoch=18.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_56/epoch=9.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_69/epoch=19.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_51/epoch=22.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_4/epoch=17.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_3/epoch=19.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_34/epoch=23.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_33/epoch=21.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_11/epoch=13.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_16/epoch=17.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_29/epoch=13.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_20/epoch=21.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_27/epoch=2.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_18/epoch=15.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_73/epoch=3.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_42/epoch=15.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_45/epoch=6.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_26/epoch=16.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_19/epoch=8.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_21/epoch=16.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_17/epoch=20.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_28/epoch=21.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_10/epoch=7.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_44/epoch=13.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_43/epoch=7.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint completed_runs/version_72/epoch=10.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9
