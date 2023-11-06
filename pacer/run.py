# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from pacer.utils.flags import flags

import numpy as np
import copy
import torch
import wandb

from learning import amp_continuous
from learning import amp_continuous_value
from learning import amp_players
from learning import amp_value_players
from learning import amp_models
from learning import amp_sept_models
from learning import amp_sept_value_models
from learning import amp_network_builder
from learning import amp_network_sept_builder
from learning import amp_network_sept_value_builder
from learning import amp_network_sept_cnn_builder

args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)

    

    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))

    runner.algo_factory.register_builder('amp_continuous_value', lambda **kwargs : amp_continuous_value.AMPValueAgent(**kwargs))
    runner.player_factory.register_builder(
        'amp_continuous_value',
        lambda **kwargs: amp_value_players.AMPPlayerContinuousValue(**kwargs))

    runner.model_builder.model_factory.register_builder('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    runner.model_builder.model_factory.register_builder('continuous_amp_sept',lambda network, **kwargs: amp_sept_models.ModelAMPContinuousSept(network))
    runner.model_builder.model_factory.register_builder('continuous_amp_sept_value',lambda network, **kwargs: amp_sept_value_models.ModelAMPContinuousSeptValue(network))

    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
    runner.model_builder.network_factory.register_builder('amp_sept', lambda **kwargs: amp_network_sept_builder.AMPSeptBuilder())
    runner.model_builder.network_factory.register_builder('amp_sept_value', lambda **kwargs: amp_network_sept_value_builder.AMPSeptValueBuilder())
    runner.model_builder.network_factory.register_builder('amp_sept_cnn', lambda **kwargs: amp_network_sept_cnn_builder.AMPSeptCNNBuilder())

    
    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg_env_name = args.cfg_env.split("/")[-1].split(".")[0]

    args.logdir = args.network_path
    cfg, cfg_train, logdir = load_cfg(args)
    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path, flags.small_terrain, flags.show_traj, flags.server_mode, flags.slow, flags.height_debug, flags.random_heading, flags.no_virtual_display= \
        args.debug, args.follow, False, False, False, False, False, args.small_terrain, True, args.server_mode, False, False, args.random_heading, args.no_virtual_display

    flags.add_proj = args.add_proj

    if args.server_mode:

        flags.follow = args.follow = True
        flags.fixed = args.fixed = True
        flags.no_collision_check = True
        flags.show_traj = True


    if args.real_mesh:
        cfg['env']['episodeLength'] = 900

    project_name = cfg.get("project_name", "crossroads_amp")

    if (not args.no_log) and (not args.test) and (not args.debug):
        wandb.init(
            project=project_name,
            resume=not args.resume_str is None,
            id=args.resume_str,
            notes=cfg.get("'notes", "no notes"),
        )
        wandb.config.update(cfg, allow_val_change=True)
        wandb.run.name = cfg_env_name
        wandb.run.save()

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size

    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file
    flags.test = args.test


    # Create default directories for weights and statistics
    cfg_train['params']['config']['network_path'] = args.network_path
    args.log_path = osp.join(args.log_path, cfg['name'], cfg_env_name)
    cfg_train['params']['config']['log_path'] = args.log_path
    cfg_train['params']['config']['train_dir'] = args.log_path # jp hack fix this asap

    os.makedirs(args.network_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    vargs = vars(args)

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    return

if __name__ == '__main__':
    main()
