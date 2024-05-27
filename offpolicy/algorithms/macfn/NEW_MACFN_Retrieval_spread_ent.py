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
        self.fc_mu = nn.Linear(256, obs_dim)
        self.fc_logvar = nn.Linear(256, obs_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, observations, actions):
        oa = torch.cat([observations, actions], -1)

        q1 = F.relu(self.l1(oa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        mu = self.fc_mu(q1)
        logvar = self.fc_logvar(q1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        next_obs = mu + eps * std

        return next_obs, mu, logvar

def calculate_entropy(logvar):
    constant = torch.tensor(2.0 * np.pi).to(logvar.device)
    entropy = 0.5 * torch.sum(1 + logvar + torch.log(constant))
    return entropy

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

        pre_obs, mu, logvar = self.retrieval(next_obs, action)

        # Compute retrieval loss
        mse_loss = F.mse_loss(pre_obs, obs)
        entropy = calculate_entropy(logvar)
        retrieval_loss = mse_loss - 0.00001 * entropy

        print(f"Retrieval Loss: {retrieval_loss.item()}")

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

    # assume action, observation space is homogeneous
    action_dim = env.action_space[0].shape[0]
    min_action = env.action_space[0].low[0]
    max_action = env.action_space[0].high[0]
    obs_dim = env.observation_space[0].shape[0]
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


    observations, done = env.reset(), False

    while frame_idx < max_frames:

        episode_timesteps += 1

        actions = np.array([np.random.uniform(low=-1., high=1., size=(2)) for _ in np.arange(n_agents)])

        next_observations, rewards, dones, infos = env.step(actions)
        next_observations = np.array(next_observations)
        observations = np.array(observations)
        rewards = np.array(rewards)
        dones = np.array(dones)

        reward = np.sum(rewards) / rewards.shape[0]
        done_bool = (
                    True in dones) if episode_timesteps < max_episode_steps else 1
        replay_buffer.add(observations, actions,
                          next_observations, reward, done_bool)

        observations = next_observations
        episode_reward += reward

        if frame_idx >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if frame_idx >= start_timesteps and frame_idx % 10000 == 0:
            torch.save(policy.retrieval.state_dict(), 'retrieval_spread_ent.pkl')

        if done_bool:
            observations, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        frame_idx += 1


if __name__ == "__main__":
    main()
