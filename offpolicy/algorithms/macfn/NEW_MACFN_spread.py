from offpolicy.config import get_config
from offpolicy.envs.mpe.MPE_Env import MPEEnv

import random
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import pickle
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class MAReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.buffer)


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


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


class Network(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Network, self).__init__()

        self.l1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        oa = torch.cat([obs, action], -1)

        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = F.softplus(self.l3(q1))
        return q1


class CFN(object):
    def __init__(
            self,
            n_agents,
            obs_dim,
            action_dim,
            hidden_dim,
            min_action,
            max_action,
            uniform_action_size,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.network = Network(obs_dim, action_dim, hidden_dim).to(device)
        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.retrieval = Retrieval(obs_dim, action_dim).to(device)
        self.retrieval_optimizer = torch.optim.Adam(self.retrieval.parameters(), lr=3e-5)

        self.n_agents = n_agents
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action

        self.uniform_action_size = uniform_action_size
        self.uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                                size=(self.uniform_action_size, self.action_dim))
        self.uniform_action = torch.Tensor(self.uniform_action).to(device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def select_action(self, obs, is_max):
        sample_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                          size=(self.uniform_action_size, self.action_dim))
        with torch.no_grad():
            sample_action = torch.Tensor(sample_action).to(device)
            state = torch.FloatTensor(obs.reshape(1, -1)).repeat(self.uniform_action_size, 1).to(device)
            edge_flow = self.network(state, sample_action).reshape(-1)
            if is_max == 0:
                idx = Categorical(edge_flow.float()).sample(torch.Size([1]))
                action = sample_action[idx[0]]
            elif is_max == 1:
                action = sample_action[edge_flow.argmax()]
        return action.cpu().data.numpy().flatten()

    def set_uniform_action(self):
        self.uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                                size=(self.uniform_action_size, self.action_dim))
        self.uniform_action = torch.Tensor(self.uniform_action).to(device)
        return self.uniform_action

    def train(self, replay_buffer, frame_idx, batch_size=256, max_episode_steps=50, sample_flow_num=100):
        # Sample replay buffer
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(np.float32(not_done)).to(device)

        with torch.no_grad():
            uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                               size=(batch_size, max_episode_steps, self.n_agents, sample_flow_num,
                                                     self.action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            current_obs = next_obs.repeat(1, 1, 1, sample_flow_num).reshape(batch_size, max_episode_steps,
                                                                            self.n_agents,
                                                                            sample_flow_num, -1)
            inflow_state = self.retrieval(current_obs,
                                          uniform_action)  # (batch_size, max_episode_steps, self.n_agents, sample_flow_num, self.obs_dim)
            inflow_state = torch.cat(
                [inflow_state, obs.reshape(batch_size, max_episode_steps, self.n_agents, -1, self.obs_dim)], -2)
            uniform_action = torch.cat(
                [uniform_action, action.reshape(batch_size, max_episode_steps, self.n_agents, -1, self.action_dim)],
                -2)

        epi = torch.ones((batch_size, max_episode_steps)).to(device)

        edge_inflow = self.network(inflow_state, uniform_action).reshape(batch_size, max_episode_steps, self.n_agents,
                                                                         -1)
        inflow = torch.log(
            torch.sum(torch.exp(torch.log(edge_inflow)), -1).prod(dim=-1) + epi)  # (batch_size, max_episode_steps)

        with torch.no_grad():
            uniform_action = np.random.uniform(low=self.min_action, high=self.max_action,
                                               size=(batch_size, max_episode_steps, self.n_agents, sample_flow_num,
                                                     self.action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            outflow_obs = next_obs.repeat(1, 1, 1, (sample_flow_num + 1)).reshape(batch_size, max_episode_steps,
                                                                                  self.n_agents,
                                                                                  (sample_flow_num + 1), -1)
            last_action = torch.zeros((batch_size, 1, self.n_agents, self.action_dim)).to(device)
            last_action = torch.cat([action[:, 1:, :, :], last_action],
                                    1)  # (batch_size, max_episode_steps, self.n_agents, self.action_dim)
            uniform_action = torch.cat(
                [uniform_action, last_action.reshape(batch_size, max_episode_steps, self.n_agents, 1, self.action_dim)],
                -2)  # (batch_size, max_episode_steps, self.n_agents, sample_flow_num + 1, self.action_dim)

        edge_outflow = self.network(outflow_obs, uniform_action).reshape(batch_size, max_episode_steps, self.n_agents,
                                                                         -1)

        outflow = torch.log(torch.sum(torch.exp(torch.log(edge_outflow)), -1).prod(dim=-1) + reward + epi)

        network_loss = F.mse_loss(inflow, outflow, reduction='none')
        network_loss = torch.mean(torch.sum(network_loss, dim=1))
        print(network_loss)
        self.network_optimizer.zero_grad()
        network_loss.backward()
        self.network_optimizer.step()

        if frame_idx % 5 == 0:
            pre_state = self.retrieval(next_obs, action)
            retrieval_loss = F.mse_loss(pre_state, obs)
            print(retrieval_loss)

            # Optimize the network
            self.retrieval_optimizer.zero_grad()
            retrieval_loss.backward()
            self.retrieval_optimizer.step()


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
    env = "MPE"
    scenario = "simple_spread"
    num_landmarks = 3
    n_agents = 3
    exp = "debug"
    seed = 1
    max_episode_steps = 25

    args = [
        '--env_name', env,
        '--experiment_name', exp,
        '--scenario_name', scenario,
        '--num_agents', str(n_agents),
        '--num_landmarks', str(num_landmarks),
        '--seed', str(seed),
        '--episode_length', str(max_episode_steps),
        '--use_soft_update',
        '--lr', '7e-4',
        '--hard_update_interval_episode', '200',
        '--num_env_steps', '10000000',
        '--use_wandb',
    ]

    parser = get_config()
    all_args = parse_args(args, parser)

    env = MPEEnv(all_args)
    test_env = MPEEnv(all_args)
    env.reset()
    test_env.reset()

    obs_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].shape[0]
    min_action = env.action_space[0].low[0]
    max_action = env.action_space[0].high[0]
    hidden_dim = 256
    uniform_action_size = 1000
    replay_buffer_size = 8000
    replay_buffer = MAReplayBuffer(replay_buffer_size)
    max_frames = 1666
    start_timesteps = 260
    frame_idx = 0
    episode_rewards = []
    test_rewards = []
    x_idx = []
    batch_size = 200
    test_epoch = 0
    expl_noise = 0.4
    sample_flow_num = 99
    repeat_episode_num = 5
    sample_episode_num = 1000

    writer = SummaryWriter(log_dir="runs/MACFN_Spread_" + now_time)

    policy = CFN(n_agents, obs_dim, action_dim, hidden_dim, min_action, max_action, uniform_action_size)
    # policy.retrieval.load_state_dict(torch.load('retrieval_spread.pkl'))

    while frame_idx < max_frames:
        observations = env.reset()
        episode_reward = 0

        obs_buf = []
        action_buf = []
        reward_buf = []
        next_obs_buf = []
        not_done_buf = []

        for step in range(max_episode_steps):
            with torch.no_grad():
                actions = np.array([policy.select_action(observations[agent], 0) for agent in np.arange(n_agents)])

            next_observations, rewards, dones, infos = env.step(actions)
            next_observations = np.array(next_observations)
            observations = np.array(observations)
            rewards = np.array(rewards)
            dones = np.array(dones)
            done = True in dones

            reward = np.sum(rewards) / rewards.shape[0]

            obs_buf.append(observations)
            action_buf.append(actions)
            reward_buf.append(reward)
            next_obs_buf.append(next_observations)
            not_done_buf.append(1 - done)

            observations = next_observations
            episode_reward += reward

            if done:
                frame_idx += 1
                print(frame_idx)
                replay_buffer.push(obs_buf, action_buf, reward_buf, next_obs_buf, not_done_buf)
                break

            if frame_idx >= start_timesteps :
                policy.train(replay_buffer, frame_idx, batch_size, max_episode_steps, sample_flow_num)

        episode_rewards.append(episode_reward)
        print(episode_reward)

        # if frame_idx > start_timesteps and frame_idx % 25 == 0:
        #     print(frame_idx)
        #     test_epoch += 1
        #     avg_test_episode_reward = 0
        #     for i in range(repeat_episode_num):
        #         test_observations = test_env.reset()
        #         test_episode_reward = 0
        #         for s in range(max_episode_steps):
        #             with torch.no_grad():
        #                 test_actions = np.array(
        #                     [policy.select_action(test_observations[agent], 0) for agent in np.arange(n_agents)])
        #
        #             test_next_observations, test_rewards, test_dones, test_infos = test_env.step(test_actions)
        #             test_next_observations = np.array(test_next_observations)
        #             test_rewards = np.array(test_rewards)
        #             test_dones = np.array(test_dones)
        #             test_done = True in test_dones
        #
        #             test_reward = np.sum(test_rewards) / test_rewards.shape[0]
        #             test_observations = test_next_observations
        #             test_episode_reward += test_reward
        #
        #             if test_done:
        #                 break
        #         avg_test_episode_reward += test_episode_reward

           #  torch.save(policy.network.state_dict(), "spread.pkl")
            # writer.add_scalar("MACFN_Spread_reward", avg_test_episode_reward / repeat_episode_num,
            #                  global_step=frame_idx * max_episode_steps)

if __name__ == "__main__":
    main()
