import sys
from offpolicy.config import get_config
from offpolicy.envs.mpe.MPE_Env import MPEEnv
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Retrieval(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Retrieval, self).__init__()

        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, obs_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, observations, actions):
        oa = torch.cat([observations, actions], -1)

        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        return q1


class TrainRetrieval(object):
    def __init__(
            self,
            obs_dim,
            action_dim,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        self.retrieval = Retrieval(obs_dim, action_dim).to(device)
        self.retrieval_target = copy.deepcopy(self.retrieval)
        self.retrieval_optimizer = torch.optim.Adam(self.retrieval.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        obs, action, next_obs, reward, not_done = replay_buffer.sample(batch_size)

        pre_obs = self.retrieval(next_obs, action)

        # Compute retrieval loss
        retrieval_loss = F.mse_loss(pre_obs, obs)
        print(retrieval_loss)

        # Optimize the retrieval
        self.retrieval_optimizer.zero_grad()
        retrieval_loss.backward()
        self.retrieval_optimizer.step()

    def save(self, filename):
        torch.save(self.retrieval.state_dict(), filename + "_retrieval")
        torch.save(self.retrieval_optimizer.state_dict(), filename + "_retrieval_optimizer")

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of agents")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main():
    max_episode_steps = 100
    n_agents = 3
    obs_dim = 18
    action_dim = 5
    min_action = 0
    max_action = 1
    hidden_dim = 256
    policy = TrainRetrieval(obs_dim, action_dim)
    replay_buffer_size = 100000
    replay_buffer = utils.MAReplayBuffer(n_agents, obs_dim, action_dim)
    max_frames = 100000
    start_timesteps = 1000
    frame_idx = 0
    rewards = []
    test_rewards = []
    batch_size = 256
    test_epoch = 0
    expl_noise = 0.1
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    parser = get_config()
    all_args = parse_args([], parser)

    env = MPEEnv(all_args)
    test_env = MPEEnv(all_args)

    env.reset()
    test_env.reset()

    observations, done = env.reset(), False

    while frame_idx < max_frames:

        episode_timesteps += 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        next_observations, rewards, dones, truncations, infos = env.step(actions)
        reward = sum(rewards.values()) / len(rewards.values())
        done_bool = (True in dones.values() or True in truncations.values()) if episode_timesteps < max_episode_steps else 1
        replay_buffer.add(np.array(list(observations.values())), np.array(list(actions.values())),
                          np.array(list(next_observations.values())), reward, done_bool)

        observations = next_observations
        episode_reward += reward

        if frame_idx >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if frame_idx >= start_timesteps and frame_idx % 10000 == 0:
            torch.save(policy.retrieval.state_dict(), 'retrieval_spread.pkl')

        if done_bool:
            (observations, _), done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        frame_idx += 1


if __name__ == "__main__":
    main()

