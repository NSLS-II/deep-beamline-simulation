from env.Beamline_RL import BeamlineModel, BeamlineEnvironment
from utils import runner
from tensorforce import Agent

# initialize the model we wish to use
BeamlineModel = BeamlineModel()

# initialize the environment we wish to use
environment = BeamlineEnvironment()

# initialize Tensorforce agent
agent = Agent.create(agent='ppo', environment=environment, batch_size=1)

# call the runner
runner(environment, agent,
    max_step_per_episode=1000,
    n_episodes=10000)
