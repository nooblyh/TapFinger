import os

import gym
import tianshou
import torch
from tianshou.data import Collector
from tianshou.env.venvs import DummyVectorEnv
from policy.ppo import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic
from torch.utils.tensorboard import SummaryWriter

import custom_gym.myenv
from nn.model import Actor
from nn.model import Critics, HAN
from utils import config, tools

if __name__ == '__main__':
    env_num = 80
    envs = DummyVectorEnv([lambda: gym.make('ENVTEST-v1', is_random=False, is_test=False) for _ in range(env_num)])
    tmp = [lambda: gym.make('ENVTEST-v1', is_random=False, is_test=True, needs_print=True)]
    test_env = DummyVectorEnv(tmp + [lambda: gym.make('ENVTEST-v1', is_random=False, is_test=True) for _ in range(1, env_num)])
    assert len(envs) == env_num

    model = HAN()

    actor = Actor(model, config.one_discrete_action_space, preprocess_net_output_dim=config.hidden_dim,
                  device=config.device)
    critic = Critics(model, preprocess_net_output_dim=config.hidden_dim, device=config.device)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist_fn=None,
    ).to(config.device)

    train_collector = Collector(policy, envs, tianshou.data.VectorReplayBuffer(total_size=64000, buffer_num=env_num))
    test_collector = Collector(policy, test_env, None)

    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    trial_name = current_time

    log_path = os.path.join('runs', trial_name)

    if os.path.exists(log_path):
        if config.device == torch.device("cpu"):
            policy.load_state_dict(
                torch.load(os.path.join(log_path, "checkpoint_policy.pth"), map_location=torch.device("cpu")))
        else:
            policy.load_state_dict(torch.load(os.path.join(log_path, "checkpoint_policy.pth")))
        resume_from_log = True
    else:
        resume_from_log = False

    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)


    def save_best_fn(m):
        torch.save(m.state_dict(), os.path.join(log_path, 'best_policy.pth'))


    def save_checkpoint_fn(epoch, env_step, gradient_step):
        torch.save(policy.state_dict(), os.path.join(log_path, 'checkpoint_policy.pth'))
        return


    result = tianshou.trainer.onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=400, step_per_epoch=100000, repeat_per_collect=2, episode_per_collect=64, episode_per_test=128,
        batch_size=256, logger=logger,
        save_best_fn=save_best_fn, save_checkpoint_fn=save_checkpoint_fn, resume_from_log=resume_from_log)
    print(f'Finished training! Use {result["duration"]}')
