import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import utilities
import itertools
from pathlib import Path
from actor.actor import GNNPolicy
from datetime import datetime

class Brain:
    """
    Brain class. Holds the policy, and receives requests from the agents to sample actions using
    it, given a state. It also performs policy updates after receiving training samples from the
    agents. 
    """
    def __init__(self, config, device, problem, mode):
        self.config = config
        self.device = device
        self.problem = problem
        self.mode = mode
        self.actor = GNNPolicy().to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config['lr'])
        self.random = np.random.RandomState(seed=self.config['seed'])

        # Create timestamp to save weights
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H.%M.%S')
        self.timestamp = f"{current_date}--{current_time}"

    def sample_action_idx(self, states, greedy):

        if isinstance(greedy, bool):
            greedy = torch.tensor(np.repeat(greedy, len(states), dtype=torch.long))
        elif not isinstance(greedy, torch.Tensor):
            greedy = torch.tensor(greedy, dtype=torch.long)

        states_loader = torch_geometric.data.DataLoader(states, batch_size=self.config['batch_size'])
        greedy_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(greedy), batch_size=self.config['batch_size'])

        action_idxs = []
        for batch, (greedy,) in zip(states_loader, greedy_loader):
            with torch.no_grad():
                batch = batch.to(self.device)
                logits = self.actor(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                logits = logits[batch.action_set]

                logits_end = batch.action_set_size.cumsum(-1)
                logits_start = logits_end - batch.action_set_size
                for start, end, greedy in zip(logits_start, logits_end, greedy):
                    if greedy:
                        action_idx = logits[start:end].argmax()
                    else:
                        action_idx = torch.distributions.categorical.Categorical(logits=logits[start:end]).sample()
                    action_idxs.append(action_idx.item())

        return action_idxs

    def update(self, transitions):
        n_samples = len(transitions)
        if n_samples < 1:
           stats = {'loss': 0.0, 'reinforce_loss': 0.0, 'entropy': 0.0}
           return stats

        transitions = torch_geometric.data.DataLoader(transitions, batch_size=16, shuffle=True)

        stats = {}

        self.optimizer.zero_grad()
        for batch in transitions:
            batch = batch.to(self.device)
            loss = torch.tensor([0.0], device=self.device)
            logits = self.actor(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            logits = utilities.pad_tensor(logits[batch.action_set], batch.action_set_size)
            dist = torch.distributions.categorical.Categorical(logits=logits)

            # REINFORCE
            returns = batch.returns.float()
            reinforce_loss = - (returns * dist.log_prob(batch.action)).sum()
            reinforce_loss /= n_samples
            loss += reinforce_loss

            # ENTROPY
            entropy = dist.entropy().sum()
            entropy /= n_samples
            loss += - self.config['entropy_bonus']*entropy

            loss.backward()

            # Update stats
            stats['loss'] = stats.get('loss', 0.0) + loss.item()
            stats['reinforce_loss'] = stats.get('reinforce_loss', 0.0) + reinforce_loss.item()
            stats['entropy'] = stats.get('entropy', 0.0) + entropy.item()

        self.optimizer.step()

        return stats

    def save(self):
        # Save in the same directory as the pretrained params
        torch.save(self.actor.state_dict(),
                   f"actor/{self.problem}/0/{self.timestamp}--best_params--{self.mode}.pkl")
