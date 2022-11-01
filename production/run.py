from logger import *
from production.envs.production_env import *
from production.envs import ProductionEnv
from tensorforce.environments import Environment
from tensorforce.execution import Runner

# tf.set_random_seed(10)

timesteps = 10 ** 2  # Set time steps per episode
episodes =  1000#10 ** 3 # Set number of episodes

environment_production = Environment.create(environment='production.envs.ProductionEnv',
                                            max_episode_timesteps=timesteps)

final_reward = list()
max_reward = list()
rewards = list()

#for n in range(5):
# Tensorforce runner
runner = Runner(agent='D:\SimRL_onward - 16 machines - 2agents - version 14010420 - Hard High\config\ppo1.json', environment=environment_production)
runner.run(num_episodes=episodes)
    #if n >= 29:
#runner.agent.save('D:\SimRL_onward\model', 'runner_bal')
runner.close()

'''final_reward.append(float(np.mean(runner.episode_rewards[-20:], axis=0)))
average_rewards = [
    float(np.mean(runner.episode_rewards[n: n + 20], axis=0))
    for n in range(len(runner.episode_rewards) - 20)
]
max_reward.append(float(np.amax(average_rewards, axis=0)))
rewards.append(list(runner.episode_rewards))

    # mean_num_episodes = float(np.mean(num_episodes, axis=0))
mean_final_reward = float(np.mean(final_reward, axis=0))
mean_max_reward = float(np.mean(max_reward, axis=0))
# loss = mean_num_episodes - mean_final_reward - mean_max_reward
loss = -mean_final_reward - mean_max_reward
print("loss:   ",loss)

'''
environment_production.environment.statistics.update({'time_end': environment_production.environment.env.now})
export_statistics_logging(statistics=environment_production.environment.statistics,
                          parameters=environment_production.environment.parameters,
                          resources=environment_production.environment.resources)
