import os
import json
import gzip
import ecole
import pickle
import logging
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from datetime import datetime
from pathlib import Path


def log(str, logfile=None):
    str = f'[{datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

class State(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_index, edge_attr, variable_features,
                 action_set, action_set_size, node_id):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.variable_features = variable_features
        self.action_set = action_set
        self.action_set_size = action_set_size
        self.node_id = node_id

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)

    def to(self, device):
        """
        Current version is inplace, which is incoherent with how pytorch's to() function works.
        This overloads it.
        """
        cuda_values = {key: self[key].to(device) if isinstance(self[key], torch.Tensor) else self[key]
                        for key in self.keys}
        return State(**cuda_values)

class Transition(torch_geometric.data.Data):
    def __init__(self, state, action=None, cum_nnodes=None):
        super().__init__()
        self.constraint_features = state.constraint_features
        self.edge_index = state.edge_index
        self.edge_attr = state.edge_attr
        self.variable_features = state.variable_features
        self.action_set = state.action_set
        self.action_set_size = state.action_set_size
        self.node_id = state.node_id
        self.num_nodes = state.num_nodes

        self.action = action
        self.cum_nnodes = cum_nnodes
        self.returns = None

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)

    def to(self, device):
        """
        Current version is inplace, which is incoherent with how pytorch's to() function works.
        This overloads it.
        """
        cuda_values = {key: self[key].to(device) if isinstance(self[key], torch.Tensor) else self[key]
                        for key in self.keys}
        return Transition(**cuda_values)


def extract_state(observation, action_set, node_id):
    constraint_features = torch.FloatTensor(observation.row_features)
    edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64))
    edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1))
    variable_features = torch.FloatTensor(observation.column_features)
    action_set = torch.LongTensor(np.array(action_set, dtype=np.int64))
    action_set_size = action_set.shape[0]
    node_id = node_id

    state = State(constraint_features, edge_index, edge_attr, variable_features, action_set, action_set_size, node_id)
    state.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
    return state


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class FormatterWithHeader(logging.Formatter):
    """
    From
    https://stackoverflow.com/questions/33468174/write-header-to-a-python-log-file-but-only-if-a-record-gets-written
    """
    def __init__(self, header, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.header = header
        self.format = self.first_line_format

    def first_line_format(self, record):
        self.format = super().format
        return self.header + "\n" + self.format(record)


def configure_logging(header=""):
    os.makedirs("logs/", exist_ok=True)
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    output_file = f"logs/{current_date}--{current_time.replace(':','.')}.log"
    logging_header = (
    f"rl2branch log\n"
    f"-------------\n"
    f"Training started on {current_date} at {current_time}\n"
    )

    logger = logging.getLogger("rl2branch")
    logger.setLevel(logging.DEBUG)

    formatter = FormatterWithHeader(header=header,
                                    fmt='[%(asctime)s %(levelname)-8s]  %(threadName)-12s  %(message)s',
                                    datefmt='%H:%M:%S')

    handler_file = logging.FileHandler(output_file, 'w', 'utf-8')
    handler_file.setLevel(logging.DEBUG)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)

    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.DEBUG)
    handler_console.setFormatter(formatter)
    logger.addHandler(handler_console)
    return logger


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
