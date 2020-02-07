import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import numpy.matlib
from scipy.stats import multivariate_normal
import gym
from gym import wrappers

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge, Lambda, Activation
from keras.layers.merge import Add, Concatenate, concatenate
from keras.optimizers import Adam
import keras.backend as K
from keras import metrics

# from replay_buffer import ReplayBuffer
from replay_buffer_weight import ReplayBufferWeight

import argparse
import pprint as pp

from adInfoHRL_TD3_agent import adInfoHRLTD3

# ===========================
#   Agent Test
# ===========================
def test(sess, env_test, args, agent, result_name):

    episode_R = []

    #step_cnt = 0
    epi_cnt = 0
    total_step_cnt = 0
    test_iter = 0
    # return_step = 0
    # step_R = []
    T = 1000
    return_test = np.zeros((np.ceil(int(args['total_step_num']) / int(args['sample_step_num'])).astype('int') + 1))
    option_result = []
    state_action_pairs = []

    # sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)


    print('evaluating the deterministic policy...')
    for nn in range(int(args['test_num'])):
        state_test = env_test.reset()
        return_epi_test = 0
        # for t_test in range(int(args['max_episode_len'])):
        for t_test in range(T):
            if args['render_env']:
                env_test.render()

            action_test = agent.predict_actor(np.reshape(state_test, (1, agent.state_dim)))
            action_test = action_test.clip(env_test.action_space.low, env_test.action_space.high)

            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test[0])
            state_test = state_test2
            return_epi_test = return_epi_test + reward_test

            if terminal_test:
                break

        print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(nn), int(return_epi_test)))
        return_test[test_iter] = return_test[test_iter] + return_epi_test / float(args['test_num'])

    print('{:d}th test: return_test[test_iter] {:d}'.format(int(test_iter), int(return_test[test_iter])))
    result_filename_option = 'results/option/' + result_name + '_option.txt'
    result_filename_state_action = 'results/option/' + result_name + '_state_action_pairs.txt'

    print('sa.shape', np.asarray(state_action_pairs).shape)
    np.savetxt(result_filename_option, np.asarray(option_result))
    np.savetxt(result_filename_state_action, np.asarray(state_action_pairs))

    return  return_test #episode_R


def main(args):
    result_name = 'TD3_' + args['env'] + '_trial_idx_' + str(int(args['trial_idx']))

    for ite in range(int(args['trial_num'])):
        print('Trial Number:', ite)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

        with tf.Session(config=config) as sess:

            if args['change_seed']:
                rand_seed = 10 * ite
            else:
                rand_seed = 0

            np.random.seed(int(args['random_seed']) +int(rand_seed))
            tf.set_random_seed(int(args['random_seed']) + int(rand_seed))
            env = gym.make(args['env'])
            env.seed(int(args['random_seed'])+ int(rand_seed))

            if args['save_video']:
                try:
                    import pathlib
                    pathlib.Path("./Video/" + args['env']).mkdir(parents=True, exist_ok=True)
                    video_relative_path = "./Video/" + args['env'] + "/"

                    ## To save video of the first episode
                    env = gym.wrappers.Monitor(env, video_relative_path,
                                                    video_callable=lambda episode_id: episode_id == 0, force=True)
                    ## To save video of every episodes
                    # env_test = gym.wrappers.Monitor(env_test, video_relative_path, \
                    #    video_callable=lambda episode_id: episode_id%1==0, force =True)
                except:
                    print("Cannot create video directories. Video will not be saved.")

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            action_bound = env.action_space.high
            # Ensure action bound is symmetric
            assert (env.action_space.high[0] == -env.action_space.low[0])

            if args['method_name'] == 'TD3':
                from TD3_keras_agent import TD3
                agent = TD3(sess, env, state_dim, action_dim, action_bound, int(args['minibatch_size']),
                                tau=float(args['tau']),
                                actor_lr=float(args['actor_lr']),
                                critic_lr=float(args['critic_lr']),
                                gamma=float(args['gamma']),
                                hidden_dim=np.asarray(args['hidden_dim']),
                                )

            agent.load_model(iteration=int(args['load_model_iter']), expname=result_name)

            # if args['use_gym_monitor']:
            #     if not args['render_env']:
            #         env = wrappers.Monitor(
            #                 env, args['monitor_dir'], video_callable=False, force=True)
            #     else:
            #         env = wrappers.Monitor(env, args['monitor_dir'], video_callable=lambda episode_id: episode_id==0, force=True)

            test(sess, env, args, agent, result_name)

            # if args['use_gym_monitor']:
            #     env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRL agent')
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=100)
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.005)
    parser.add_argument('--hidden-dim', help='max size of the hidden layers', default=(400, 300))

    parser.add_argument('--save-model-num', help='number of time steps for saving the network models', default=10000)
    parser.add_argument('--load-model-iter', help='number of time steps for saving the network models', default=1000)
    parser.add_argument('--save-video', help='save the video or not', default=True)

    parser.add_argument('--test-num', help='number of episode for recording the return', default=1)


    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    # parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1001) #50000
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1001)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_TD3')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_TD3')
    parser.add_argument('--result-file', help='file name for storing results from multiple trials',
                        default='./results/trials/trials_TD3')
    parser.add_argument('--method-name', help='method name for recording the results', default='TD3')
    # parser.add_argument('--overwrite-result', help='flag for overwriting the trial file', default=True)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial-idx', help='index of trials', default=0)
    parser.add_argument('--change-seed', help='change the random seed to obtain different results', default=False)

    # parser.set_defaults(env='MountainCarContinuous-v0')
    #
    # parser.set_defaults(monitor_dir='./results/gym_adhrl_ddpg_mtcar')
    # parser.set_defaults(result_file='./results/trials/trials_AdInfoHRL_ddpg_mtcar.txt')
    # parser.set_defaults(hidden_dim=(64, 64))
    # parser.set_defaults(minibatch_size=128)
    parser.set_defaults(total_step_num=1000)
    parser.set_defaults(sample_step_num=200)
    parser.set_defaults(trial_num=1)
    parser.set_defaults(save_model_num=1000)
    parser.set_defaults(load_model_iter=201)
    # parser.set_defaults(load_model_iter=101)
    parser.set_defaults(trial_idx=1)

    # parser.set_defaults(env='HalfCheetah-v1')
    # parser.set_defaults(env='Reacher-v1')
    # parser.set_defaults(env='Swimmer-v1')
    parser.set_defaults(env='Ant-v1')
    # parser.set_defaults(env='Walker2d-v1')
    # parser.set_defaults(env='Hopper-v1')
    # parser.set_defaults(env='Humanoid-v1')
    # parser.set_defaults(max_episodes=201)
    # parser.set_defaults(trial_num=1)

    # parser.set_defaults(save_video=False)
    # parser.set_defaults(render_env=False)
    # parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(change_seed=True)
    parser.set_defaults(overwrite_result=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
