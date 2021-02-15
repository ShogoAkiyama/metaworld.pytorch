import os
import yaml
import argparse
from datetime import datetime
import torch
import random
import metaworld
from garage.experiment.deterministic import set_seed

from src.env import make_env
from src.algorithm import SAC
from src.agent import Agent


def run(args):
    set_seed(args.seed)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.num_steps is not None:
        config['Agent']['num_steps'] = args.num_steps

    # Create environments.
    task = None
    if args.env_id in metaworld.MT1.ENV_NAMES:
        task = random.choice(metaworld.MT1(args.env_id).train_tasks)
    env = make_env(args.env_id, task)
    test_env = make_env(args.env_id, task)

    # Device to use.
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{args.algo}-seed{args.seed}-{time}')

    if args.algo == 'sac':
        # SAC algorithm.
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed, **config['SAC'])
    else:
        raise Exception('You need to set "--algo sac" or "--algo src".')

    agent = Agent(
        env=env, test_env=test_env, algo=algo, log_dir=log_dir,
        device=device, seed=args.seed, **config['Agent'])
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'metaworld.yaml'))
    parser.add_argument('--num_steps', type=int, required=False)
    parser.add_argument('--env_id', type=str, default='bin-picking-v1')
    parser.add_argument('--algo', choices=['sac'], default='sac')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
