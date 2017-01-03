import gym
# env = gym.make('Copy-v0')
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('CartPole-v0')
env = gym.make('Humanoid-v1')

env.reset()
env.render()
