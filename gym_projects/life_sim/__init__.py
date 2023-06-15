from gymnasium.envs.registration import register

register(
    id='life_sim/LifeSim-v0',
    entry_point='life_sim.envs:LifeSim',
    max_episode_steps=1000,
)