import gym
import numpy as np
from RLbrain import PolicyGradient

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    print(env.action_space)
    print(env.observation_space)

    RL = PolicyGradient(
        n_actions = env.action_space.n,
        n_states= env.observation_space.shape[0],
        learning_rate= 0.3,
        reward_decay= 0.99
    )

    for i in range(400):
        observation = env.reset()
        action = RL.choose(observation)
        observation_, reward, done, _ = env.step(action)
        RL.transition(observation, action, reward)

        while True:
            env.render()
            observation = observation_
            action = RL.choose(observation)
            observation_, reward, done, _ = env.step(action)
            RL.transition(observation, action, reward)
            reward_in_this_episode = sum(RL.list_reward)
            print("Range:", i,"\t Reward:", reward_in_this_episode)

            if done:
               RL.learn()
               break


               

        
