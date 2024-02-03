import numpy as np
from src.vfa.lstd import LspiAgent
from src.agent.agent import BehaviorAgent as Agent
from src.tools import timer_tools
from src.agent.episodic_replay_buffer import EpisodicReplayBuffer


class ValueFunctionApproximation:
    def __init__(self, env, n_samples, replay_buffer: EpisodicReplayBuffer, env_name):
        self.env = env
        self.n_samples = n_samples
        self.replay_buffer = replay_buffer
        self.env_name = env_name

    def fit(self, eigenvectors):
        pass

    def collect_experience(self, policy) -> None:

        agent = Agent(policy)

        # Collect trajectories from random actions
        print('Start collecting samples.')  # TODO: Use logging
        timer = timer_tools.Timer()
        total_n_steps = 0
        collect_batch = 10_000  # TODO: Check if necessary
        while total_n_steps < self.n_samples:
            n_steps = min(collect_batch,
                          self.n_samples - total_n_steps)
            steps = agent.collect_experience(self.env, n_steps)
            self.replay_buffer.add_steps(steps)
            total_n_steps += n_steps
            print(f'({total_n_steps}/{self.n_samples}) steps collected.')
        time_cost = timer.time_cost()
        print(f'Data collection finished, time cost: {time_cost}s')

        # Plot visitation counts
        min_visitation, max_visitation, visitation_entropy, max_entropy, visitation_freq = \
            self.replay_buffer.plot_visitation_counts(
                self.env.get_states(),
                self.env_name,
                self.env.grid.astype(bool),
            )
        time_cost = timer.time_cost()
        print(f'Visitation evaluated, time cost: {time_cost}s')
        print(f'Min visitation: {min_visitation}')
        print(f'Max visitation: {max_visitation}')
        print(f'Visitation entropy: {visitation_entropy}/{max_entropy}')


if __name__ == '__main__':
    # collect experience
    x=0

    # learn eigenvectors, initialised with existing eigenvectors if exist

    # learn VF with LSPI

import os
import yaml
from argparse import ArgumentParser
import random
import subprocess
import numpy as np

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
    algorithm = hparam_yaml['algorithm']
    rng_key = jax.random.PRNGKey(hparam_yaml['seed'])
    hidden_dims = hparam_yaml['hidden_dims']

    encoder_fn = generate_hk_module_fn(MLP, d, hidden_dims, hparam_yaml['activation'])

    optimizer = optax.adam(hparam_yaml['lr'])  # TODO: Add hyperparameter to config file

    replay_buffer = EpisodicReplayBuffer(
        max_size=hparam_yaml['n_samples'])  # TODO: Separate hyperparameter for replay buffer size (?)

    Trainer = AugmentedLagrangianTrainer

    trainer = Trainer(
        encoder_fn=encoder_fn,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        logger=None,
        rng_key=rng_key,
        **hparam_yaml,
    )
    # the loop is
    # collect random experience
    # learn evs
    # fit policy
    # collect guided experience
    # learn evs
    # fit policy
    ###########
    # when you initialise the trainer, it builds the environment and collects experience
    # initialise the trainer (above)
    # trainer.train()
    # for i in n_loops_to_do:
    #   trainer.train(reset_params=False)
    #   get learned evs
    #   fit lspi
    #   collect experience with policy
    # trainer.build_environment()
    # trainer.collect_experience()

    # todo
    # make sure training loop doesn't reset parameters each time - put in flag for resetting DONE?
    # get learned evs
    # make ev format work with lspi
    #######


    trainer.train()

    # Print training time
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "exp_label",
        type=str,
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
        default='barrier.yaml',
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

