import tensorforce
from tensorforce.environments import Environment
from tensorforce.agents import Agent

environment = Environment.create(environment='configs/cartpole.json')
agent = Agent.create(agent='configs/ppo.json', environment=environment)

for episode in range(100):
    episode_states = []
    episode_internals = []
    episode_actions = []
    episode_terminal = []
    episode_reward = []

    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    sum_rewards = 0.0 
    while not terminal:
        episode_states.append(states)
        episode_internals.append(internals)
        actions, internals = agent.act(states=states, internals=internals, independent=True)
        episode_actions.append(actions)
        states, terminal, reward = environment.execute(actions=actions)
        episode_terminal.append(terminal)
        episode_reward.append(reward)
        sum_rewards += reward
    print('Episode {}: {}'.format(episode, sum_rewards))

    agent.experience(states=episode_states, internals=episode_internals, actions=episode_actions, terminal=episode_terminal, reward=episode_reward)
    agent.update()

sum_rewards = 0.0
for _ in range(100):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward
print('Mean Eval Return:', sum_rewards/100.0)
agent.close()
environment.close()











