from src.vfa.lstd import LspiAgent, Lstdq, PlotAgent
# from src.agent.agent import BehaviorAgent as Agent
# from src.tools import timer_tools
# from src.agent.episodic_replay_buffer import EpisodicReplayBuffer

import os
import yaml
import csv
from argparse import ArgumentParser
import random
import subprocess
import numpy as np
import datetime
import gymnasium as gym

import jax
import jax.numpy as jnp
import optax

from src.tools import timer_tools

from src.trainer.al import (
    AugmentedLagrangianTrainer
)
from src.agent.episodic_replay_buffer import EpisodicReplayBuffer

from src.nets.mlp import MLP
from src.nets.utils import generate_hk_module_fn
import wandb

os.environ['WANDB_API_KEY'] = '83c25550226f8a86fdd4874026d2c0804cd3dc05'
os.environ['WANDB_ENTITY'] = 'tarod13'


def main(hyperparams):
    os.chdir('../..')
    if hyperparams.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'


    # Load YAML hyperparameters
    with open(f'./src/hyperparam/{hyperparams.config_file}', 'r') as f:
        hparam_yaml = yaml.safe_load(f)  # TODO: Check necessity of hyperparams

    # Replace hparams with command line arguments
    for k, v in vars(hyperparams).items():
        if v is not None:
            hparam_yaml[k] = v

    # Set random seed
    np.random.seed(hparam_yaml['seed'])  # TODO: Check if this is the best way to set the seed
    random.seed(hparam_yaml['seed'])

    # Initialize timer
    timer = timer_tools.Timer()



    # Create trainer
    d = hparam_yaml['d']

    rng_key = jax.random.PRNGKey(hparam_yaml['seed'])

    hidden_dims = hparam_yaml['hidden_dims']

    encoder_fn = generate_hk_module_fn(MLP, d, hidden_dims, hparam_yaml['activation'])

    optimizer = optax.adam(hparam_yaml['lr'])  # TODO: Add hyperparameter to config file

    replay_buffer = EpisodicReplayBuffer(
        max_size=hparam_yaml['n_samples'])  # TODO: Separate hyperparameter for replay buffer size (?)

    if hparam_yaml['use_wandb']:
        # Set wandb save directory
        if hparam_yaml['save_dir'] is None:
            save_dir = os.getcwd()
            os.makedirs(save_dir, exist_ok=True)
            hparam_yaml['save_dir'] = save_dir

        # Initialize wandb logger
        logger = wandb.init(
            project='laplacian-encoder',
            dir=hparam_yaml['save_dir'],
            config=hparam_yaml,
        )
        # wandb_logger.watch(laplacian_encoder)   # TODO: Test overhead
    else:
        logger = None

    Trainer = AugmentedLagrangianTrainer

    # when you initialise the trainer, it builds the environment and collects experience
    trainer = Trainer(
        encoder_fn=encoder_fn,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        logger=logger,
        rng_key=rng_key,
        agent_collect_experience=Lstdq,
        **hparam_yaml,
    )

    env = trainer.env

    if hparam_yaml['relearn_eigvecs']:

        # learn EVs
        eigvec = trainer.train(return_eigvec=True)

        # save EVs to a CSV file
        with open('eigenvectors.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(eigvec)

    else:
        eigvec = np.genfromtxt('eigenvectors.csv', delimiter=',')

    # fit policy
    lspi = LspiAgent(eigenvectors=eigvec,
                     state_map=env.get_state_indices(),
                     actions=env.action_space,
                     source_of_samples=trainer.sars,
                     env=env,
                     seed=hparam_yaml['seed'])

    stopping_criteria = .001
    policy, toplot = lspi.learn(stopping_criteria)

    # TODO: collect experience using learned agent's policy
    # note: to really be sample efficient, can't need to see every state once -- otherwise tabular Q with experience
    # replay would be just as good. Need to be able to generalise to new data points -- this is IMPORTANT

    # Print training time
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))

    with open('out.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(toplot)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--relearn_eigvecs',
        type=bool,
        default=False,
        help='Relearn eigenvectors or load in pre-learned.'
    )

    parser.add_argument(
        "--exp_label",
        type=str,
        default=f'exp_label_{datetime.datetime.now()}',
        help="Experiment label",
    )

    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Raise the flag to use wandb offline."
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Raise the flag to save the model."
    )

    parser.add_argument(
        '--config_file',
        type=str,
        default='al.yaml',
        help='Configuration file to use.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save the model.'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size.'
    )
    parser.add_argument(
        '--discount',
        type=float,
        default=None,
        help='Lambda discount used for sampling states.'
    )
    parser.add_argument(
        '--total_train_steps',
        type=int,
        default=None,
        help='Number of training steps for laplacian encoder.'
    )
    parser.add_argument(
        '--max_episode_steps',
        type=int,
        default=None,
        help='Maximum trajectory length.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed for random number generators.'
    )
    parser.add_argument(
        '--env_name',
        type=str,
        default=None,
        help='Environment name.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate of the Adam optimizer used to train the laplacian encoder.'
    )
    parser.add_argument(
        '--hidden_dims',
        nargs='+',
        type=int,
        help='Hidden dimensions of the laplacian encoder.'
    )
    parser.add_argument(
        '--barrier_initial_val',
        type=float,
        default=None,
        help='Initial value for barrier coefficient in the quadratic penalty.'
    )
    parser.add_argument(
        '--lr_barrier_coefs',
        type=float,
        default=None,
        help='Learning rate of the barrier coefficient in the quadratic penalty.'
    )

    hyperparams = parser.parse_args()

    main(hyperparams)
