# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Train agent using the reinforcement learning method. User must provide a      #
# mode in {mdp, tmdp+DFS, tmdp+ObjLim}. The training parameters are read from   #
# a file config.default.json which is overriden by command line inputs, if      #
# provided.                                                                     #
# Usage:                                                                        #
# python 04_train_il.py <type> -s <seed> -g <cudaId>                            #
# Optional: use flag --wandb to log metrics using wandb (requires account)      #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #


import os
import json
import time
import glob
import numpy as np
import argparse

import ecole
from pathlib import Path
from datetime import datetime
from scipy.stats.mstats import gmean



if __name__ == '__main__':
    # read default config file
    with open("config.default.json", 'r') as f:
        config = json.load(f)

    # read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],
    )
    parser.add_argument(
        'mode',
        help='Training mode.',
        choices=['mdp', 'tmdp+DFS', 'tmdp+ObjLim'],
    )
    parser.add_argument(
        '--wandb',
        help="Use wandb?",
        default=False,
        action="store_true",
    )
    # add all config parameters as optional command-line arguments
    for param, value in config.items():
        if param == 'gpu':
            parser.add_argument(
                '-g', '--gpu',
                type=type(value),
                help='CUDA GPU id (-1 for CPU).',
                default=argparse.SUPPRESS,
            )
        elif param == 'seed':
            parser.add_argument(
                '-s', '--seed',
                type=type(value),
                help = 'Random generator seed.',
                default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                f"--{param}",
                type=type(value),
                default=argparse.SUPPRESS,
            )
    args = parser.parse_args()

    # override config with the user config file if provided
    if os.path.isfile("config.json"):
        with open("config.json", 'r') as f:
            user_config = json.load(f)
        unknown_options = user_config.keys() - config.keys()
        if unknown_options:
            raise ValueError(f"Unknown options in config file: {unknown_options}")
        config.update(user_config)

    # override config with command-line arguments if provided
    args_config = {key: getattr(args, key) for key in config.keys() & vars(args).keys()}
    config.update(args_config)

    # configure gpu
    if config['gpu'] == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{config['gpu']}"
        device = f"cuda:0"

    # import torch after gpu configuration
    import torch
    import torch.nn.functional as F
    import utilities
    from brain import Brain
    from agent import AgentPool

    if config['gpu'] > -1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Active CUDA Device: {torch.cuda.current_device()}")

    rng = np.random.RandomState(config['seed'])
    torch.manual_seed(config['seed'])

    logger = utilities.configure_logging()
    if args.wandb:
        import wandb
        wandb.init(project="rl2branch", config=config)


    # data
    if args.problem == "setcover":
        maximization = False
        valid_path = "data/instances/setcover/valid_400r_750c_0.05d"
        train_path = "data/instances/setcover/train_400r_750c_0.05d"

    elif args.problem == "cauctions":
        maximization = True
        valid_path = "data/instances/cauctions/valid_100_500"
        train_path = "data/instances/cauctions/train_100_500"

    elif args.problem == "indset":
        maximization = True
        valid_path = "data/instances/indset/valid_500_4"
        train_path = "data/instances/indset/train_500_4"

    elif args.problem == "ufacilities":
        maximization = False
        valid_path = "data/instances/ufacilities/valid_35_35_5"
        train_path = "data/instances/ufacilities/train_35_35_5"

    elif args.problem == "mknapsack":
        maximization = True
        valid_path = "data/instances/mknapsack/valid_100_6"
        train_path = "data/instances/mknapsack/train_100_6"

    # recover training / validation instances
    valid_instances = [f'{valid_path}/instance_{j+1}.lp' for j in range(config["num_valid_instances"])]
    train_instances = [f'{train_path}/instance_{j+1}.lp' for j in range(len(glob.glob(f'{train_path}/instance_*.lp')))]

    # collect the pre-computed optimal solutions for the training instances
    with open(f"{train_path}/instance_solutions.json", "r") as f:
        train_sols = json.load(f)

    valid_batch = [{'path': instance, 'seed': seed}
        for instance in valid_instances
        for seed in range(config['num_valid_seeds'])]

    def train_batch_generator():
        eps = -0.1 if maximization else 0.1
        while True:
            yield [{'path': instance, 'sol': train_sols[instance] + eps, 'seed': rng.randint(0, 2**32)}
                    for instance in rng.choice(train_instances, size=config['num_episodes_per_epoch'], replace=True)]

    train_batches = train_batch_generator()

    logger.info(f"Training on {len(train_instances)} training instances and {len(valid_instances)} validation instances")


    brain = Brain(config, device, args.problem, args.mode)
    agent_pool = AgentPool(brain, config['num_agents'], config['time_limit'], args.mode)
    agent_pool.start()
    is_validation_epoch = lambda epoch: (epoch % config['validate_every'] == 0) or (epoch == config['num_epochs'])
    is_training_epoch = lambda epoch: (epoch < config['num_epochs'])

    # Already start jobs
    if is_validation_epoch(0):
        _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)
    if is_training_epoch(0):
        train_batch = next(train_batches)
        t_samples_next, t_stats_next, t_queue_next, t_access_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True)

    # training loop
    start_time = datetime.now()
    best_tree_size = np.inf
    for epoch in range(config['num_epochs'] + 1):
        logger.info(f'** Epoch {epoch}')
        wandb_data = {}

        # Allow preempted jobs to access policy
        if is_validation_epoch(epoch):
            v_stats, v_queue, v_access = v_stats_next, v_queue_next, v_access_next
            v_access.set()
            logger.info(f"  {len(valid_batch)} validation jobs running (preempted)")
            # do not do anything with the stats yet, we have to wait for the jobs to finish !
        else:
            logger.info(f"  validation skipped")
        if is_training_epoch(epoch):
            t_samples, t_stats, t_queue, t_access = t_samples_next, t_stats_next, t_queue_next, t_access_next
            t_access.set()
            logger.info(f"  {len(train_batch)} training jobs running (preempted)")
            # do not do anything with the samples or stats yet, we have to wait for the jobs to finish !
        else:
            logger.info(f"  training skipped")

        # Start next epoch's jobs
        if epoch + 1 <= config["num_epochs"]:
            if is_validation_epoch(epoch + 1):
                _, v_stats_next, v_queue_next, v_access_next = agent_pool.start_job(valid_batch, sample_rate=0.0, greedy=True, block_policy=True)
            if is_training_epoch(epoch + 1):
                train_batch = next(train_batches)
                t_samples_next, t_stats_next, t_queue_next, t_access_next = agent_pool.start_job(train_batch, sample_rate=config['sample_rate'], greedy=False, block_policy=True)

        # Validation
        if is_validation_epoch(epoch):
            v_queue.join()  # wait for all validation episodes to be processed
            logger.info('  validation jobs finished')

            v_nnodess = [s['info']['nnodes'] for s in v_stats]
            v_lpiterss = [s['info']['lpiters'] for s in v_stats]
            v_times = [s['info']['time'] for s in v_stats]

            wandb_data.update({
                'valid_nnodes_g': gmean(np.asarray(v_nnodess) + 1) - 1,
                'valid_nnodes': np.mean(v_nnodess),
                'valid_nnodes_max': np.amax(v_nnodess),
                'valid_nnodes_min': np.amin(v_nnodess),
                'valid_time': np.mean(v_times),
                'valid_lpiters': np.mean(v_lpiterss),
            })
            if epoch == 0:
                v_nnodes_0 = wandb_data['valid_nnodes'] if wandb_data['valid_nnodes'] != 0 else 1
                v_nnodes_g_0 = wandb_data['valid_nnodes_g'] if wandb_data['valid_nnodes_g']!= 0 else 1
            wandb_data.update({
                'valid_nnodes_norm': wandb_data['valid_nnodes'] / v_nnodes_0,
                'valid_nnodes_g_norm': wandb_data['valid_nnodes_g'] / v_nnodes_g_0,
            })

            if wandb_data['valid_nnodes_g'] < best_tree_size:
                best_tree_size = wandb_data['valid_nnodes_g']
                logger.info('Best parameters so far (1-shifted geometric mean), saving model.')
                brain.save()

        # Training
        if is_training_epoch(epoch):
            t_queue.join()  # wait for all training episodes to be processed
            logger.info('  training jobs finished')
            logger.info(f"  {len(t_samples)} training samples collected")
            t_losses = brain.update(t_samples)
            logger.info('  model parameters were updated')

            t_nnodess = [s['info']['nnodes'] for s in t_stats]
            t_lpiterss = [s['info']['lpiters'] for s in t_stats]
            t_times = [s['info']['time'] for s in t_stats]

            wandb_data.update({
                'train_nnodes_g': gmean(t_nnodess),
                'train_nnodes': np.mean(t_nnodess),
                'train_time': np.mean(t_times),
                'train_lpiters': np.mean(t_lpiterss),
                'train_nsamples': len(t_samples),
                'train_loss': t_losses.get('loss', None),
                'train_reinforce_loss': t_losses.get('reinforce_loss', None),
                'train_entropy': t_losses.get('entropy', None),
            })

        # Send the stats to wandb
        if args.wandb:
            wandb.log(wandb_data, step = epoch)

        # If time limit is hit, stop process
        elapsed_time = datetime.now() - start_time
        if elapsed_time.days >= 6: break

    logger.info(f"Done. Elapset time: {elapsed_time}")
    if args.wandb:
        wandb.join()
        wandb.finish()

    v_access_next.set()
    t_access_next.set()
    agent_pool.close()
