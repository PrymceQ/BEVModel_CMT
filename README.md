# BEVModel_CMT
The reproduction project of the BEV model # Cross Modal Transformer (CMT), which includes some code annotation work.

Thanks for the CMT authors！[Paper](https://github.com/junjie18/CMT) | [Code](https://github.com/junjie18/CMT)

## Necessary File Format
- data/nuscenes/
  - maps/
  - samples/
  - sweeps/
  - v1.0-test/
  - v1.0-trainval/
- projects/
- tools/
- ckpts/

## Data create

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

## Train Code
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash tools/dist_train.sh projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py 4
```

## Test Code
```
python tools/test.py projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py ckpts/voxel0100_r50_800x320_epoch20.pth --eval bbox
```

## Training Result Record

ID | Name | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | Per-class results | Epochs | Data | Learning rate | Batch_size | GPUs | Train_time | Eval_time | Log_file
:----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :----------- | :-----------
0 | voxel0100_r50_800x320_epoch20 | 0.6275 | 0.6784 | 0.3294 | 0.2541 | 0.3035 | 0.2810 | 0.1853 |    | 20 | All | optimizer.lr=0.00007, lr_config.target_ratio=(3, 0.0001), | 8, sample per gpu=2 | 4 x Nvidia Geforce 3090 | 4days8hours | 83.6s | 


## Resolved issues
1. Out of memey

2. "grad_norm：nan" from 7-8 epoches.

3. After CTRL+C terminated the CMT training program, cuda memory still occupied.

