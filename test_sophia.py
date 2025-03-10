from src.trainer import AugmentedLagrangianTrainer
from src.agent.episodic_replay_buffer import EpisodicReplayBuffer
from src.tools import timer_tools
from src.nets import generate_hk_module_fn, MLP

import optax
import jax


def main(seed=1234):

    # Initialize timer
    timer = timer_tools.Timer()

    encoder_fn = generate_hk_module_fn(
        MLP,
        11,  # these have to be non keyword
        [256, 256, 256],
        'relu')
    replay_buffer = EpisodicReplayBuffer(max_size=200000)
    rng_key = jax.random.PRNGKey(seed)
    optimizer = optax.adam(0.001)

    trainer = AugmentedLagrangianTrainer(
        encoder_fn=encoder_fn,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        batch_size=1024,
        logger=None,
        rng_key=rng_key,
        seed=seed,

        env_name="GridRoom-16",
        obs_mode="xy",
        env_family="Grid-v0",

        discount=0.9,
        window_size=180,
        max_episode_steps=50,
        reduction_factor=1,
        eigval_precision_order=16,
        n_samples=200000,
        d=11,
        total_train_steps=20000,
        permute_step=20000,

        save_eig=False,
        use_wandb=False,
        barrier_initial_val=2.0,
        lr_barrier_coefs=1.0,
        min_barrier_coefs=0,
        max_barrier_coefs=10000,
        use_barrier_normalization=False,
        use_barrier_for_duals=False,
        lr_duals=0.0001,
        min_duals=-100,
        max_duals=100,
        lr_dual_velocities=0.1,
        error_update_rate=1,
        q_error_update_rate=0.1,
        normalize_graph_loss=False,
        print_freq=200,
        do_plot_eigenvectors=True,
        save_model=False,  # 37
    )

    trainer.train()
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))


if __name__ == '__main__':
    main()
