# PACER: Pedestrian Animation Controller

Official implementation of PACER, Pedestrian Animation ControllER, of CVPR 2023 paper: "Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion".

**Note: this repo only contains the PACER component of this paper** (i.e., the pedestrian animation controler). For TRACE (the trajectory diffusion model), please see [this other repository](https://github.com/nv-tlabs/trace).

[[paper]](https://arxiv.org/abs/2304.01893) [[website]](https://research.nvidia.com/labs/toronto-ai/trace-pace/) [[Video]](https://www.youtube.com/watch?v=225c52QDkzg)

<div float="center">
    <img src="assets/gif/trace_pace.gif" />
</div>

## News 🚩

[Nov 6, 2023] Code released.

## Introduction
This repo implements PACER, Pedestrian Animation ControllER. PACER is a framework for controlling simulated humanoids of different body shapes to navigate diverse type of terrains while avoiding obstacles and other humanoids. PACER is built upon the [AMP](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) code base and trained with selected locomotion sequences from the AMASS dataset. 


## Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:
```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl

```

## Data processing for evaluating/training PACER

### Evaluation
Use the following script to download trained models and sample data.

```
bash download_data.sh
```

### Training

We train on a subset of the [AMASS](https://amass.is.tue.mpg.de/) dataset.

For prcessing the AMASS, first, download the AMASS dataset from [AMASS](https://amass.is.tue.mpg.de/). Then, run the following script on the unzipped data:


```
python uhc/data_process/process_amass_raw.py
```

which dumps the data into the `amass_db_smplh.pt` file. Then, run 

```
python uhc/data_process/process_amass_db.py
```

We further process these data into Motionlib format by running the following script:

```
python uhc/data_process/convert_amass_isaac.py
```

### Using your own mesh

Refer to `scripts/render_mesh.py` and `scripts/render_mesh_ply.py` in creating compatible mesh files and height maps. 

Refer to `create_mesh_ground` function in `pacer/env/tasks/humanoid_pedestrain_terrain.py` for how to load your own mesh. 

Then, add `--real_mesh` flag to the command line to use your own mesh.

## Evaluation 

```
## Model with terrain + trajectory following 
python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer_no_shape.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/release/pacer_no_shape --test --num_envs 1 --epoch -1 --small_terrain --no_virtual_display

## Model with terrain + trajectory following + shape 
python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/release/pacer --test --num_envs 1 --epoch -1 --small_terrain --no_virtual_display

## Model with terrain + trajectory following + shape + group
python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer_group_cnn.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_cnn_task.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/release/pacer_group_cnn  --test --num_envs 1 --epoch -1 --small_terrain --no_virtual_display

## Model with terrain + trajectory following + shape + getup
python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer_getup.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/release/pacer_getup  --test --num_envs 1 --epoch -1 --small_terrain  --no_virtual_display

```


## Training

```
python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer_no_shape.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml --motion_file data/amass/pkls/amass_isaac_run_upright_slim.pkl --network_path output/exp/pacer_no_shape --headless 

python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml --motion_file data/amass/pkls/amass_isaac_run_upright_slim.pkl --network_path output/exp/pacer --headless 

python pacer/run.py --task HumanoidPedestrianTerrain --cfg_env pacer/data/cfg/pacer_group_cnn.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_cnn_task.yaml --motion_file data/amass/pkls/amass_isaac_run_upright_slim.pkl --network_path output/exp/pacer_group_cnn --headless 

python pacer/run.py --task HumanoidPedestrianTerrainGetup --cfg_env pacer/data/cfg/pacer_getup.yaml --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml --motion_file data/amass/pkls/amass_isaac_run_crawl_upright_slim.pkl --network_path output/exp/pacer_getup --headless 
```

## Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| F | focus on humanoid |
| Right click + WASD | change view port |
| Shift + Right click + WASD | change view port fast |
| B | create stright path |
| G | go to fixed point on flat ground |
| L | record screenshot, press again to stop recording|
| ; | cancel screen shot|
| J | apply large force to humanoid |
| M | cancel reset |

... more short cut can be found in `pacer/env/tasks/base_task.py`

Notes on rendering: I am using pyvirtualdisplay to record the video such that you can see all humanoids at the same time (default function will only capture the first environment). You can disable it in `pacer/env/tasks/base_task.py`. 


## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}            
```

## References
This repository is built on top of the following amazing repositories: 
* Main code framework is from: [AMP](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* Part of the SMPL_robot code is from: [UHC](https://github.com/ZhengyiLuo/UniversalHumanoidControl)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)
* Some scripts are from [PHC](https://github.com/ZhengyiLuo/PerpetualHumanoidControl). PHC also has a more detailed README on SMPL_Robot. 

Please follow the license of the above repositories for usage of that part of the codebase (the licenses are included in [this repo](./assets/licenses/)). 
