import ecole
import threading
import queue
import utilities
import numpy as np
from collections import namedtuple


class AgentPool():
    """
    Class holding the reference to the agents and the policy sampler.
    Puts jobs in the queue through job sponsors.
    """
    def __init__(self, brain, n_agents, time_limit, mode):
        self.jobs_queue = queue.Queue()
        self.policy_queries_queue = queue.Queue()
        self.policy_sampler = PolicySampler("Policy Sampler", brain, self.policy_queries_queue)
        self.agents = [Agent(f"Agent {i}", time_limit, self.jobs_queue, self.policy_queries_queue, mode) for i in range(n_agents)]

    def start(self):
        self.policy_sampler.start()
        for agent in self.agents:
            agent.start()

    def close(self):
        # order the episode sampling agents to stop
        for _ in self.agents:
            self.jobs_queue.put(None)
        self.jobs_queue.join()
        # order the policy sampler to stop
        self.policy_queries_queue.put(None)
        self.policy_queries_queue.join()

    def start_job(self, instances, sample_rate, greedy=False, block_policy=False):
        """
        Starts a job.
        A job is a set of tasks. A task consists of an instance that needs to be solved and instructions
        to do so (sample rate, greediness).
        The job queue is loaded with references to the job sponsor, which is in itself a queue specific
        to a job. It is the job sponsor who holds the lists of tasks. The role of the job sponsor is to
        keep track of which tasks have been completed.
        """
        job_sponsor = queue.Queue()
        samples = []
        stats = []

        policy_access = threading.Event()
        if not block_policy:
            policy_access.set()

        for instance in instances:
            task = {'instance': instance, 'sample_rate': sample_rate, 'greedy': greedy,
                    'samples': samples, 'stats': stats, 'policy_access': policy_access}
            job_sponsor.put(task)
            self.jobs_queue.put(job_sponsor)

        ret = (samples, stats, job_sponsor)
        if block_policy:
            ret = (*ret, policy_access)

        return ret

    def wait_completion(self):
        # wait for all running episodes to finish
        self.jobs_queue.join()


class PolicySampler(threading.Thread):
    """
    Gathers policy sampling requests from the agents, and process them in a batch.
    """
    def __init__(self, name, brain, requests_queue):
        super().__init__(name=name)
        self.brain = brain
        self.requests_queue = requests_queue

    def run(self):
        stop_order_received = False
        while True:
            requests = []
            request = self.requests_queue.get()
            while True:
                # check for a stopping order
                if request is None:
                    self.requests_queue.task_done()
                    stop_order_received = True
                    break
                # add request to the batch
                requests.append(request)
                # keep collecting more requests if available, without waiting
                try:
                    request = self.requests_queue.get(block=False)
                except queue.Empty:
                    break

            states = [r['state'] for r in requests]
            greedys = [r['greedy'] for r in requests]
            receivers = [r['receiver'] for r in requests]

            # process all requests in a batch
            action_idxs = self.brain.sample_action_idx(states, greedys)
            for action_idx, receiver in zip(action_idxs, receivers):
                receiver.put(action_idx)
                self.requests_queue.task_done()

            if stop_order_received:
                break


class Agent(threading.Thread):
    """
    Agent class. Receives tasks from the job sponsor, runs them and samples transitions if
    requested.
    """
    def __init__(self, name, time_limit, jobs_queue, policy_queries_queue, mode):
        super().__init__(name=name)
        self.jobs_queue = jobs_queue
        self.policy_queries_queue = policy_queries_queue
        self.policy_answers_queue = queue.Queue()
        self.mode = mode

        # Setup Ecole environment
        scip_params={'separating/maxrounds': 0,
                     'presolving/maxrestarts': 0,
                     'limits/time': time_limit,
                     'timing/clocktype': 2}
        observation_function=(
            ecole.observation.FocusNode(),
            ecole.observation.NodeBipartite()
            )
        reward_function=ecole.reward.NNodes().cumsum()
        information_function={
            'nnodes': ecole.reward.NNodes().cumsum(),
            'lpiters': ecole.reward.LpIterations().cumsum(),
            'time': ecole.reward.SolvingTime().cumsum()
        }

        if mode == 'tmdp+ObjLim':
            self.env = ObjLimBranchingEnv(scip_params=scip_params,
                                          pseudo_candidates=False,
                                          observation_function=observation_function,
                                          reward_function=reward_function,
                                          information_function=information_function)
        elif mode == 'tmdp+DFS':
            self.env = DFSBranchingEnv(scip_params=scip_params,
                                       pseudo_candidates=False,
                                       observation_function=observation_function,
                                       reward_function=reward_function,
                                       information_function=information_function)
        elif mode == 'mdp':
            self.env = MDPBranchingEnv(scip_params=scip_params,
                                       pseudo_candidates=False,
                                       observation_function=observation_function,
                                       reward_function=reward_function,
                                       information_function=information_function)
        else:
            raise NotImplementedError

    def run(self):
        while True:
            job_sponsor = self.jobs_queue.get()

            # check for a stopping order
            if job_sponsor is None:
                self.jobs_queue.task_done()
                break

            # Get task from job sponsor
            task = job_sponsor.get()
            instance = task['instance']
            sample_rate = task['sample_rate']
            greedy = task['greedy'] # should actions be chosen greedily w.r.t. the policy?
            training = not greedy
            samples = task['samples']
            stats = task['stats']
            policy_access = task['policy_access']
            seed = instance['seed']

            transitions = []
            self.env.seed(seed)
            rng = np.random.RandomState(seed)
            if sample_rate > 0:
                tree_recorder = TreeRecorder()

            # Run episode
            observation, action_set, cum_nnodes, done, info = self.env.reset(instance = instance['path'],
                                                                             primal_bound=instance.get('sol', None),
                                                                             training=training)
            policy_access.wait()
            iter_count = 0
            while not done:
                focus_node_obs, node_bipartite_obs = observation
                state = utilities.extract_state(node_bipartite_obs, action_set, focus_node_obs.number)

                # send out policy queries
                self.policy_queries_queue.put({'state': state, 'greedy': greedy, 'receiver': self.policy_answers_queue})
                action_idx = self.policy_answers_queue.get()

                action = action_set[action_idx]

                # collect transition samples if requested
                if sample_rate > 0:
                    tree_recorder.record_branching_decision(focus_node_obs)
                    keep_sample = rng.rand() < sample_rate
                    if keep_sample:
                        transition = utilities.Transition(state, action_idx, cum_nnodes)
                        transitions.append(transition)

                observation, action_set, cum_nnodes, done, info = self.env.step(action)
                iter_count += 1
                if (iter_count>50000) and training: done=True # avoid too large trees during training for stability

            if (iter_count>50000) and training: # avoid too large trees during training for stability
                job_sponsor.task_done()
                self.jobs_queue.task_done()
                continue

            # post-process the collected samples (credit assignment)
            if sample_rate > 0:
                if self.mode in ['tmdp+ObjLim', 'tmdp+DFS']:
                    subtree_sizes = tree_recorder.calculate_subtree_sizes()
                    for transition in transitions:
                        transition.returns = -subtree_sizes[transition.node_id] - 1
                else:
                    assert self.mode == 'mdp'
                    for transition in transitions:
                        transition.returns = transition.cum_nnodes - cum_nnodes

            # record episode samples and stats
            samples.extend(transitions)
            stats.append({'order': task, 'info': info})

            # tell both the agent pool and the original task sponsor that the task is done
            job_sponsor.task_done()
            self.jobs_queue.task_done()


class TreeRecorder:
    """
    Records the branch-and-bound tree from a custom brancher.

    Every node in SCIP has a unique node ID. We identify nodes and their corresponding
    attributes through the same ID system.
    Depth groups keep track of groups of nodes at the same depth. This data structure
    is used to speed up the computation of the subtree size.
    """
    def __init__(self):
        self.tree = {}
        self.depth_groups = []

    def record_branching_decision(self, focus_node, lp_cand=True):
        id = focus_node.number
        # Tree
        self.tree[id] = {'parent': focus_node.parent_number,
                         'lowerbound': focus_node.lowerbound,
                         'num_children': 2 if lp_cand else 3  }
        # Add to corresponding depth group
        if len(self.depth_groups) > focus_node.depth:
            self.depth_groups[focus_node.depth].append(id)
        else:
            self.depth_groups.append([id])

    def calculate_subtree_sizes(self):
        subtree_sizes = {id: 0 for id in self.tree.keys()}
        for group in self.depth_groups[::-1]:
            for id in group:
                parent_id = self.tree[id]['parent']
                subtree_sizes[id] += self.tree[id]['num_children']
                if parent_id >= 0: subtree_sizes[parent_id] += subtree_sizes[id]
        return subtree_sizes


class DFSBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Custom branching environment that changes the node strategy to DFS when training.
    """
    def reset_dynamics(self, model, primal_bound, training, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if training:
            # Set the dfs node selector as the least important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 666666)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 666666)
        else:
            # Set the dfs node selector as the most important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 0)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 0)

        return super().reset_dynamics(model, *args, **kwargs)

class DFSBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = DFSBranchingDynamics

class ObjLimBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Custom branching environment that allows the user to set an initial primal bound.
    """
    def reset_dynamics(self, model, primal_bound, training, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if primal_bound is not None:
            pyscipopt_model.setObjlimit(primal_bound)

        return super().reset_dynamics(model, *args, **kwargs)

class ObjLimBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = ObjLimBranchingDynamics

class MDPBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Regular branching environment that allows extra input parameters, but does
    not use them.
    """
    def reset_dynamics(self, model, primal_bound, training, *args, **kwargs):
        return super().reset_dynamics(model, *args, **kwargs)

class MDPBranchingEnv(ecole.environment.Environment):
    __Dynamics__ = MDPBranchingDynamics
