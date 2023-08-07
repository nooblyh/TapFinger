import os
import sys

import gym
import tianshou
import torch
from tianshou.data import Collector
from tianshou.env.venvs import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import custom_gym.myenv
from nn.model import Actors, ToyModel
from nn.model import Critics, HAN
from policy.ppo import PPOPolicy
from trace_util import convert_trace
from utils import config

if __name__ == '__main__':
    train_env_num = 32
    test_env_num = 64
    trace_fit = convert_trace.csv_to_dict()
    envs = DummyVectorEnv([lambda: gym.make('ENVTEST-v3', is_random=False, is_test=False, gnn_state=False, trace_fit=trace_fit) for _ in range(train_env_num)])
    tmp = [lambda: gym.make('ENVTEST-v3', is_random=False, is_test=True, needs_print=True, gnn_state=False, trace_fit=trace_fit)]
    test_env = DummyVectorEnv(
        tmp + [lambda: gym.make('ENVTEST-v3', is_random=False, is_test=True, gnn_state=False, trace_fit=trace_fit) for _ in range(1, test_env_num)])

    model = ToyModel(config.stacking_dim, config.hidden_dim)

    actor = Actors(model, pointer_type="attn", no_GNN=True, is_toy=True)
    critic = Critics(model, preprocess_net_output_dim=config.hidden_dim, device=config.device, no_GNN=True, is_toy=True)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)
    # scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 0.99999 ** epoch if epoch < 92101 else 1e-2)
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist_fn=None,
        max_batchsize=1024,
        # lr_scheduler=scheduler,
    ).to(config.device)

    train_collector = Collector(policy, envs, tianshou.data.VectorReplayBuffer(total_size=64000, buffer_num=train_env_num))
    test_collector = Collector(policy, test_env, None)

    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    trial_name = current_time

    # trial_name = sys.argv[1]

    log_path = os.path.join('runs', trial_name)

    if os.path.exists(log_path):
        if config.device == torch.device("cpu"):
            policy.load_state_dict(
                torch.load(os.path.join(log_path, "checkpoint_policy.pth"), map_location=torch.device("cpu")))
            optim.load_state_dict(
                torch.load(os.path.join(log_path, "checkpoint_optim.pth"), map_location=torch.device("cpu")))
        else:
            policy.load_state_dict(torch.load(os.path.join(log_path, "checkpoint_policy.pth")))
            optim.load_state_dict(torch.load(os.path.join(log_path, "checkpoint_optim.pth")))
        resume_from_log = True
    else:
        resume_from_log = False

    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, update_interval=1)


    def save_best_fn(m):
        # torch.save(m.state_dict(), os.path.join(log_path, 'best_policy.pth'))
        torch.save(actor.state_dict(), os.path.join(log_path, 'best_actor.pth'))
        return


    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # torch.save(policy.state_dict(), os.path.join(log_path, 'checkpoint_policy.pth'))
        # torch.save(actor.state_dict(), os.path.join(log_path, 'checkpoint_actor.pth'))
        # torch.save(optim.state_dict(), os.path.join(log_path, 'checkpoint_optim.pth'))
        return


    result = tianshou.trainer.onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=400, step_per_epoch=20000, repeat_per_collect=2, episode_per_collect=32, episode_per_test=64,
        batch_size=256, logger=logger,
        save_best_fn=save_best_fn, save_checkpoint_fn=save_checkpoint_fn, resume_from_log=resume_from_log)
    print(f'Finished training! Use {result["duration"]}')
