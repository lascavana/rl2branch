# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates file with solutions to the training instances. Needs to be run once #
# before training.                                                              #                                                                     #
# Usage:                                                                        #
# python 02_get_instance_solutions.py <type> -j <njobs> -n <ninstances>         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import glob
import json
import sys
import argparse
import threading
import queue

import ecole

class OptimalSol:
    def __init__(self):
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done=False):
        pyscipopt_model = model.as_pyscipopt()
        if done: reward = pyscipopt_model.getObjVal(original=True)
        else: reward = None
        return reward


def solve_instance(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which instances are received.
    out_queue : queue.Queue
        Output queue in which to solution.
    """
    reward_fun = OptimalSol()
    while not in_queue.empty():
        instance = in_queue.get()
        env = ecole.environment.Configuring( scip_params={},
                                             observation_function=None,
                                             reward_function=reward_fun )
        env.reset(str(instance))
        print(f'Solving {instance}')
        _, _, solution, _, _ = env.step({})
        out_queue.put({instance: solution})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-n', '--ninst',
        help='Number of instances to solve.',
        type=int,
        default=10000,
    )
    args = parser.parse_args()

    if args.problem == 'setcover':
        instance_dir = 'data/instances/setcover/train_400r_750c_0.05d'
        instances = glob.glob(instance_dir + '/*.lp')
    elif args.problem == 'cauctions':
        instance_dir = 'data/instances/cauctions/train_100_500'
        instances = glob.glob(instance_dir + '/*.lp')
    elif args.problem == 'indset':
        instance_dir = 'data/instances/indset/train_500_4'
        instances = glob.glob(instance_dir + '/*.lp')
    elif args.problem == 'ufacilities':
        instance_dir = 'data/instances/ufacilities/train_35_35_5'
        instances = glob.glob(instance_dir + '/*.lp')
    elif args.problem == 'mknapsack':
        instance_dir = 'data/instances/mknapsack/train_100_6'
        instances = glob.glob(instance_dir + '/*.lp')
    else:
        raise NotImplementedError

    num_inst = min(args.ninst,len(instances))
    orders_queue = queue.Queue()
    answers_queue = queue.Queue()
    for instance in instances[:num_inst]:
        orders_queue.put(instance)
    print(f'{num_inst} instances on queue.')

    workers = []
    for i in range(args.njobs):
        p = threading.Thread(
                target=solve_instance,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    i = 0
    solutions = {}
    while i < num_inst:
        answer = answers_queue.get()
        solutions.update(answer)
        i += 1

    with open(instance_dir + "/instance_solutions.json", "w") as f:
        json.dump(solutions, f)

    for p in workers:
        assert not p.is_alive()
