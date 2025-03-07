from src.trainer import AugmentedLagrangianTrainer
from src.agent.episodic_replay_buffer import EpisodicReplayBuffer
from src.tools import timer_tools
from src.nets import generate_hk_module_fn, MLP

import optax
import jax


def main(encoder_net=MLP,
         d=11,
         hidden_dims=[256, 256, 256],
         activation='relu',
         n_samples=200000,
         seed=1234
         ):

    # Initialize timer
    timer = timer_tools.Timer()

    encoder_fn = generate_hk_module_fn(
        encoder_net,
        d,
        hidden_dims,
        activation,
    )

    replay_buffer = EpisodicReplayBuffer(max_size=n_samples)

    rng_key = jax.random.PRNGKey(seed)

    optimizer = optax.adam(0.001)

    trainer = AugmentedLagrangianTrainer(
        encoder_fn=encoder_fn,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        logger=None,
        rng_key=rng_key,
        env="GridRoom-16",
        obs_mode="xy",
        d=11,
        discount=0.9,
        batch_size=1024,

    )

    trainer.train()

    # Print training time
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))


if __name__ == '__main__':
    main()
