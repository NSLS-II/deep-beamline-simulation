import math
from math import cos, sin, ceil, floor
import numpy as np
from numpy import arcsin
from numpy.linalg import norm
from .graph_utils import plot_duo, plot_multiple, plot_xy

from tensorforce.environments import Environment
import numpy as np

class BeamlineModel:
    '''
    Beamline model with aperature information
    '''
    def __init__(self):
        self.timestep = 0
        self.max_timestep = 1000

        self.horizontal_size = 0.01
        self.vertical_size = 0.01
        self.size = [self.horizontal_size, self.vertical_size]

        self.horizontal_position = 0
        self.vertical_position = 0
        self.position = [self.horizontal_position, self.vertical_position]
        # the items that will change each step
        self.observations = [self.horizontal_size, self.vertical_size, self.horizontal_position, self.vertical_position]

        self.pos_vec = [[], []]

    def update_size(self, new_hor_size, new_vert_size):
        '''
        Setter for size if needed
        '''
        self.horizontal_size = new_hor_size
        self.vertical_size = new_vert_size
        self.size = [self.horizontal_size, self.vertical_size]

    def update_position(self, new_hor_pos, new_vert_pos):
        '''
        Setter for positions if needed
        '''
        self.horizontal_position = new_hor_pos
        self.vertical_position = new_vert_pos
        self.position = [self.horizontal_position, self.vertical_position]

    def compute_timestep(self):
        '''
        This will compute the change in the sizes and positions at each timestep
        Depends on what formulas I will use
        '''
        self.horizontal_size = self.horizontal_size
        self.vertical_size = self.vertical_size
        self.horizontal_position = self.horizontal_position
        self.vertical_position = self.vertical_position
        self.pos_vec[0].append(self.horizontal_position)
        self.pos_vec[1].append(self.vertical_position)
        # create list of observations
        self.observations = [self.horizontal_size, self.vertical_size, self.horizontal_position, self.vertical_position]
        return self.observations

    def plot_graphs(self, save_figs=False, path=None):
        Series = [self.pos_vec[0], self.pos_vec[1]]
        labels = ["Horizontal position", "Vertical position"]
        xlabel = "time (s)"
        ylabel = "Distance from origin (m)"
        title = "Position vs time"
        plot_duo(Series, labels, xlabel, ylabel, title, save_fig=save_figs, path=path)

        Series = [self.size[0], self.size[1]]
        labels = ["Horizontal Size", "Vertical Size"]
        xlabel = "time (s)"
        ylabel = "Distance from origin (m)"
        title = "Position vs time"
        plot_duo(Series, labels, xlabel, ylabel, title, save_fig=save_figs, path=path)


class BeamlineEnvironment(Environment):
    '''
    Custom tensorforce environment to use beamline data 
    for reinforcement learning
    '''
    def __init__(self):
        super().__init__()
        self.BeamModel = BeamlineModel()
        self.NUM_POSITIONS = len(self.BeamModel.position)
        self.NUM_SIZE = len(self.BeamModel.size)
        
        self.max_step_per_episode = 5000
        self.finished = False
        self.episode_end = False
        self.STATES_SIZE = len(self.BeamModel.observations)

    def states(self):
        '''
        A state in the beamline model is dependent on
        all of the information for an aperture
        '''
        return dict(type='float', shape=(self.STATES_SIZE,))

    def actions(self):
        # defined the type of actions available to the agent
        # in this example there are 2 values the network should
        # control, each has a specified amount of values 
        return {
                "size":dict(type='int', num_values=self.NUM_SIZE),
                "positions":dict(type='int', num_values=self.NUM_POSITIONS)
        }

    def max_episode_timesteps(self):
        '''
        Docs state this is optional and should only be defined if
        the environment has a natural fixed max episode length
        .create(..., max_episode_timesteps=?)
        In this example we can assume the number of steps is 100
        '''
        return self.max_step_per_episode

    def terminal(self):
        # The simulation ends when the pinhole becomes bigger than 0.5 (the normal aperature opening is 1)
        # We only check one direction assuming horizontal_size = vertical size
        self.finished = self.BeamModel.horizontal_size < 0.5
        # End the episode when 
        self.episode_end = self.BeamModel.timestep > self.max_step_per_episode
        return self.finished or self.episode_end

    def reward(self):
        '''
        Reward at the moment is just adding or decreasing by one 
        if the simulation completes
        '''
        reward = None
        if self.finished:
            reward = 1
        else: 
            reward =- 1
        return reward

    def close(self):
        super().close()

    def reset(self):
        '''
        Resets the environment to wait for new episode to begin
        '''
        state = np.zeros(shape=(self.STATES_SIZE,))
        self.BeamModel = BeamlineModel()
        return state 

    def execute(self, actions):
        reward = 0
        nb_timesteps = 1
        for i in range(1, nb_timesteps + 1):
            #next_state = self.BeamlineModel.compute_timestep(actions, nb_timesteps)
            next_state = self.BeamModel.compute_timestep()
            reward += self.reward()
            if self.terminal():
                reward = reward / i
                break
        if i == nb_timesteps:
            reward = reward / nb_timesteps
        return next_state, self.terminal(), reward
