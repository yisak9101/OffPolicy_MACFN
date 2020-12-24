import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from offpolicy.utils.util import init, to_torch

class M_QMixer(nn.Module):
    """
    computes Q_tot from individual Q_a values and the state
    """

    def __init__(self, args, num_agents, cent_obs_dim, device, multidiscrete_list=None):
        """
        init mixer class
        """
        super(M_QMixer, self).__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.cent_obs_dim = cent_obs_dim
        self._use_orthogonal = args.use_orthogonal

        # dimension of the hidden layer of the mixing net
        self.hidden_layer_dim = args.mixer_hidden_dim
        # dimension of the hidden layer of each hypernet
        self.hypernet_hidden_dim = args.hypernet_hidden_dim

        if multidiscrete_list:
            self.num_mixer_q_inps = sum(multidiscrete_list)
        else:
            self.num_mixer_q_inps = self.num_agents

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # hypernets output the weight and bias for the 2 layer MLP which takes in the state and agent Qs and outputs Q_tot
        if args.hypernet_layers == 1:
            # each hypernet only has 1 layer to output the weights
            # hyper_w1 outputs weight matrix which is of dimension (hidden_layer_dim x N)
            self.hyper_w1 = init_(nn.Linear(self.cent_obs_dim, self.num_mixer_q_inps * self.hidden_layer_dim))
            # hyper_w2 outputs weight matrix which is of dimension (1 x hidden_layer_dim)
            self.hyper_w2 = init_(nn.Linear(self.cent_obs_dim, self.hidden_layer_dim))
        elif args.hypernet_layers == 2:
            # 2 layer hypernets: output dimensions are same as above case
            self.hyper_w1 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.num_mixer_q_inps * self.hidden_layer_dim))
            )
            self.hyper_w2 = nn.Sequential(
                init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
                nn.ReLU(),
                init_(nn.Linear(self.hypernet_hidden_dim, self.hidden_layer_dim))
            )

        # hyper_b1 outputs bias vector of dimension (1 x hidden_layer_dim)
        self.hyper_b1 = init_(nn.Linear(self.cent_obs_dim, self.hidden_layer_dim))
        # hyper_b2 outptus bias vector of dimension (1 x 1)
        self.hyper_b2 = nn.Sequential(
            init_(nn.Linear(self.cent_obs_dim, self.hypernet_hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.hypernet_hidden_dim, 1))
        )
        self.to(device)

    def forward(self, agent_q_inps, states):
        """outputs Q_tot, using the individual agent Q values and the centralized env state as inputs"""
        agent_q_inps = to_torch(agent_q_inps).to(**self.tpdv)
        states = to_torch(states).to(**self.tpdv)

        batch_size = agent_q_inps.size(0)
        states = states.view(-1, self.cent_obs_dim).float()
        # reshape agent_q_inps into shape (batch_size x 1 x N) to work with torch.bmm
        agent_q_inps = agent_q_inps.view(-1, 1, self.num_mixer_q_inps).float()

        # get the first layer weight matrix batch, apply abs val to ensure nonnegative derivative
        w1 = torch.abs(self.hyper_w1(states))
        # get first bias vector
        b1 = self.hyper_b1(states)
        # reshape to batch_size x N x Hidden Layer Dim (there's a different weight mat for each batch element)
        w1 = w1.view(-1, self.num_mixer_q_inps, self.hidden_layer_dim)
        # reshape to batch_size x 1 x Hidden Layer Dim
        b1 = b1.view(-1, 1, self.hidden_layer_dim)
        # pass the agent qs through first layer defined by the weight matrices, and apply Elu activation
        hidden_layer = F.elu(torch.bmm(agent_q_inps, w1) + b1)
        # get second layer weight matrix batch
        w2 = torch.abs(self.hyper_w2(states))
        # get second layer bias batch
        b2 = self.hyper_b2(states)

        # reshape to shape (batch_size x hidden_layer dim x 1)
        w2 = w2.view(-1, self.hidden_layer_dim, 1)
        # reshape to shape (batch_size x 1 x 1)
        b2 = b2.view(-1, 1, 1)
        # pass the hidden layer results through output layer, with no activataion
        out = torch.bmm(hidden_layer, w2) + b2
        # reshape to (batch_size, 1, 1)
        q_tot = out.view(batch_size, -1, 1)

        return q_tot
