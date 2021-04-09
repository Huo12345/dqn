import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    env.render()
    x = env.step(env.action_space.sample())
    print(x)
env.close()
