import os
import time
import itertools
import shutil
from collections import namedtuple
import numpy as np
from tensorforce import Agent
from env.graph_utils import plot_duo, plot_multiple

def write_to_txt_general(data, path):
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("env/" + str(path), cur_path)
    text_file = open(new_path, "w")
    n = text_file.write(str(data))
    text_file.close()

def write_pos_and_angles_to_txt(environment, path):
    write_to_txt_general(environment.BeamModel.position, path + "/positions.txt")
    write_to_txt_general(environment.BeamModel.size, path + "/sizes.txt")

def show_policy(thrust_vec, theta_vec, combination, title="Policy vs time"):
    plot_duo(
        Series=[thrust_vec, theta_vec],
        labels=["Thrust", "Theta"],
        xlabel="time (s)",
        ylabel="Force intensity (N)/Angle value (°)",
        title=title,
        save_fig=True,
        path="env",
        folder=str(combination),
        time=True,
    )
    '''
    plot_multiple(
        Series=[distances],
        labels=["TO-Distance"],
        xlabel="episodes",
        ylabel="TO-Distance (m)",
        title="Distance vs episodes",
        save_fig=True,
        path="env",
        folder=str(combination),
        time=False,
    )
    '''

def batch_information(
    i, result_vec, combination, total_combination, temp_time, number_batches
):
    if result_vec:

        print(
            "Combination {}/{}, Batch {}/{}, Best result: {},Time per batch {}s, Combination ETA: {}mn{}s, Total ETA: {}mn{}s".format(
                combination,
                total_combination,
                i,
                number_batches,
                int(result_vec[-1]),
                round(temp_time / i, 1),
                round(((temp_time * number_batches / i) - temp_time) // 60),
                round(((temp_time * number_batches / i) - temp_time) % 60),
                round(((temp_time * number_batches / i) * total_combination) // 60),
                round(((temp_time * number_batches / i) * total_combination) % 60),
            )
        )

def train_info(i, n_episodes, start_time, combination):
    temp_time = time.time() - start_time
    time_per_episode = temp_time / (i + 1)
    print(
        "combination : ",
        combination,
        "episode : ",
        i,
        "/",
        n_episodes,
        " time per episode",
        round(time_per_episode, 2),
        "seconds. ",
        "estimated time to finish",
        int((time_per_episode * n_episodes) - temp_time),
        "seconds.",
    )

def run(
    environment,
    agent,
    n_episodes,
    max_step_per_episode,
    combination,
    total_combination,
    batch,
    test=False,
):
    """
    Train agent for n_episodes
    """
    environment.BeamModel.max_step_per_episode = max_step_per_episode
    # score is only based on the reward and it's mean
    Score = namedtuple("Score", ["reward", "reward_mean"])
    score = Score([], [])

    start_time = time.time()
    for i in range(1, n_episodes + 1):
        # Variables initialization
        Episode = namedtuple("Episode", ["rewards", "position_values", "size_values"],)
        episode = Episode([], [], [])

        if total_combination == 1 and (
            i % 50 == 0
        ):  # Print training information every 50 episodes
            train_info(i, n_episodes, start_time, combination)

        # Initialize episode
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False

        while not terminal:  # While an episode has not yet terminated

            if test:  # Test mode (deterministic, no exploration)
                actions, internals = agent.act(
                    states=states, internals=internals, independent=True
                )
                states, terminal, reward = environment.execute(actions=actions)
            else:  # Train mode (exploration and randomness)
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

            episode.position_values.append(round(actions["positions"], 2))
            episode.size_values.append(round(actions["size"], 2))
            episode.rewards.append(reward)
            # if terminal and (i % 100 == 0):
            #     terminal_info(
            #         episode, states, actions,
            #     )
        score.reward.append(np.sum(episode.rewards))
        score.reward_mean.append(np.mean(score.reward))
        #score.distance.append(environment.FlightModel.Pos[0])
    if not (test):
        show_policy(
            episode.position_values,
            episode.size_values,
            combination,
            title="pvt_train_" + str(batch),
        )
    if test:
        show_policy(
            episode.position_values,
            episode.size_values,
            combination,
            title="pvt_" + str(batch),
        )
        if not os.path.exists(os.path.join("env", "Pos_and_angles", str(batch))):
            os.mkdir(os.path.join("env", "Pos_and_angles", str(batch)))
        write_pos_and_angles_to_txt(environment, "Pos_and_angles/" + str(batch))
    plot_multiple(
        Series=[score.reward, score.reward_mean],
        labels=["Reward", "Mean reward"],
        xlabel="time (s)",
        ylabel="Reward",
        title="Global Reward vs time",
        save_fig=True,
        path="env",
        folder=str(combination),
    )
    print(episode)

    return environment.BeamModel.position[0]

def runner(
    environment,
    agent,
    max_step_per_episode,
    n_episodes,
    n_episodes_test=1,
    combination=1,
    total_combination=1,
):

    result_vec = []
    start_time = time.time()
    number_batches = round(n_episodes / 100) + 1
    for i in range(1, number_batches):
        temp_time = time.time() - start_time
        batch_information(
            i, result_vec, combination, total_combination, temp_time, number_batches
        )
        # Train agent
        run(
            environment,
            agent,
            100,
            max_step_per_episode,
            combination=combination,
            total_combination=total_combination,
            batch=i,
        )
        print(environment.BeamModel.pos_vec)
        # Test Agent
        result_vec.append(
            run(
                environment,
                agent,
                n_episodes_test,
                max_step_per_episode,
                combination=combination,
                total_combination=total_combination,
                batch=i,
                test=True,
            )
        )
    environment.BeamModel.plot_graphs(save_figs=True, path="env")
    plot_multiple(
        Series=[result_vec],
        labels=["TO-Distance"],
        xlabel="episodes",
        ylabel="Distance (m)",
        title="TO-Distance vs episodes",
        save_fig=True,
        path="env",
        folder=str(combination),
        time=False,
    )
    agent.close()
    environment.close()
    save_distances(
        result_vec, combination, environment
    )  # saves distances results for each combination in a txt file.
    return environment.BeamModel.position[0]

def save_distances(result_vec, combination, environment):
    """
    Saves distances results in a txt in the current combination folder
    """
    if not os.path.exists(os.path.join("env", "Distances", str(combination))):
        os.mkdir(os.path.join("env", "Distances", str(combination)))
    write_to_txt_general(result_vec, "Distances/" + str(combination) + "/distances.txt")
    write_pos_and_angles_to_txt(environment, "Distances/" + str(combination))
