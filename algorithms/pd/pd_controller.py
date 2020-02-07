import numpy as np
from algorithms import logger
import copy as cp
import time


def learn(
        controller,
        env,
        nb_epochs=5,   # with default settings, perform 1M steps total
        nb_epoch_cycles=150,
        nb_rollout_steps=400,
        data_path_reward="",
        data_path_steps="",
        data_path_states="",
        data_path_times=""
):

    epochs_rewards = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_times = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_steps = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_states = []

    for epoch in range(nb_epochs):
        logger.info("======================== The {} epoch start !!! =========================".format(epoch))
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_times = []
        epoch_episode_states = []
        for i in range(nb_epoch_cycles):
            start_time = time.time()
            s = env.reset()
            episode_reward = 0.
            episode_step = 0
            episode_states = []
            for j in range(nb_rollout_steps):

                print('state', s[:6])
                a = 0
                s_, us, r, done, safe = env.step([(0, 0, 0, 0, 0, 0), ""])
                episode_reward += r
                episode_step += 1
                episode_states.append(
                    [cp.deepcopy(s), cp.deepcopy(a), np.array(cp.deepcopy(r)), cp.deepcopy(s_)])
                if done or j == nb_rollout_steps - 1 or safe is False:
                    print('Ep: %i | %s | %s | step: %i' % (
                        i, '---' if not done else 'done', 'unsafe' if not safe else 'safe', j))
                    break
                s = s_

            """ store data """
            duration = time.time() - start_time
            epoch_episode_rewards.append(episode_reward)
            epoch_episode_steps.append(episode_step)
            epoch_episode_times.append(cp.deepcopy(duration))
            epoch_episode_states.append(cp.deepcopy(episode_states))

            epochs_rewards[epoch, i] = episode_reward
            epochs_steps[epoch, i] = episode_step
            epochs_times[epoch, i] = cp.deepcopy(duration)

        epochs_states.append(cp.deepcopy(epoch_episode_states))

        # # save data
        np.save(data_path_reward, epochs_rewards)
        np.save(data_path_steps, epochs_steps)
        np.save(data_path_states, epochs_states)
        np.save(data_path_times, epochs_times)
