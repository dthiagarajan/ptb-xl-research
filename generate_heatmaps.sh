python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_0/checkpoints/epoch=23.ckpt --model_name Simple1DCNN --heatmap_layers pool1 pool2 pool3 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_1/checkpoints/epoch=13.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_2/checkpoints/epoch=13.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_3/checkpoints/epoch=19.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_4/checkpoints/epoch=17.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_5/checkpoints/epoch=9.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_6/checkpoints/epoch=15.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_7/checkpoints/epoch=17.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_8/checkpoints/epoch=19.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_9/checkpoints/epoch=2.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_10/checkpoints/epoch=7.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_11/checkpoints/epoch=13.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_12/checkpoints/epoch=17.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_13/checkpoints/epoch=21.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_14/checkpoints/epoch=11.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_15/checkpoints/epoch=17.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_16/checkpoints/epoch=17.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_17/checkpoints/epoch=20.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_18/checkpoints/epoch=15.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_19/checkpoints/epoch=8.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_20/checkpoints/epoch=21.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_21/checkpoints/epoch=16.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_22/checkpoints/epoch=10.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_23/checkpoints/epoch=14.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_24/checkpoints/epoch=16.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_25/checkpoints/epoch=7.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_26/checkpoints/epoch=16.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_27/checkpoints/epoch=2.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_28/checkpoints/epoch=21.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_29/checkpoints/epoch=13.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_30/checkpoints/epoch=23.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_31/checkpoints/epoch=23.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_32/checkpoints/epoch=21.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_33/checkpoints/epoch=21.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_34/checkpoints/epoch=23.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_35/checkpoints/epoch=20.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_36/checkpoints/epoch=15.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_37/checkpoints/epoch=9.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_38/checkpoints/epoch=19.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_39/checkpoints/epoch=13.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_40/checkpoints/epoch=5.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_41/checkpoints/epoch=18.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_42/checkpoints/epoch=15.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_43/checkpoints/epoch=7.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_44/checkpoints/epoch=13.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_45/checkpoints/epoch=6.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_47/checkpoints/epoch=9.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_48/checkpoints/epoch=10.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_49/checkpoints/epoch=24.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_50/checkpoints/epoch=21.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_51/checkpoints/epoch=22.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_52/checkpoints/epoch=18.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_53/checkpoints/epoch=5.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_54/checkpoints/epoch=1.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_55/checkpoints/epoch=7.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_56/checkpoints/epoch=9.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_57/checkpoints/epoch=20.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_58/checkpoints/epoch=17.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_59/checkpoints/epoch=13.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_60/checkpoints/epoch=18.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_61/checkpoints/epoch=16.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_62/checkpoints/epoch=18.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_63/checkpoints/epoch=19.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_64/checkpoints/epoch=4.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_65/checkpoints/epoch=19.ckpt --model_name resnet18 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_66/checkpoints/epoch=18.ckpt --model_name resnet34 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_67/checkpoints/epoch=3.ckpt --model_name resnet50 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_68/checkpoints/epoch=15.ckpt --model_name resnet101 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_69/checkpoints/epoch=19.ckpt --model_name resnet152 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_70/checkpoints/epoch=15.ckpt --model_name resnext50_32x4d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_71/checkpoints/epoch=6.ckpt --model_name resnext101_32x8d --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_72/checkpoints/epoch=10.ckpt --model_name wide_resnet50_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9 &&
python show_heatmaps.py --distributed_backend ddp --gpus 4 --model_checkpoint lightning_logs/version_73/checkpoints/epoch=3.ckpt --model_name wide_resnet101_2 --heatmap_layers layer2 layer3 layer4 &&
nvidia-smi | grep 'python3' | awk '{ print $5 }' | xargs -n1 kill -9