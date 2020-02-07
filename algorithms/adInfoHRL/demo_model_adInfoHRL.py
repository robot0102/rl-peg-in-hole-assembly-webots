import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import gym

import argparse
import pprint as pp


# ===========================
#   Agent Test
# ===========================
def test(sess, env_test, args, agent, result_name):
    epi_cnt = 0
    total_step_cnt = 0
    test_iter = 0
    T = 1000
    return_test = np.zeros((np.ceil(int(args['total_step_num']) / int(args['sample_step_num'])).astype('int') + 1))
    option_result = []
    option_cluster = []
    state_action_pairs = []
    states = []
    action_noise = float(args['action_noise'])

    print('evaluating the deterministic policy...')
    for nn in range(int(args['test_num'])):
        state_test = env_test.reset()
        return_epi_test = 0
        option_test = 0

        for t_test in range(T):
            if args['render_env']:
                env_test.render()

            if t_test % int(args['temporal_num']) == 0:

                option_test, _, _ = agent.max_option(np.reshape(state_test, (1, agent.state_dim)))
                option_test = option_test[0]

                option_one_hot_test = np.zeros((1, int(agent.option_num)))
                option_one_hot_test[0][option_test] = 1.
                print('option', option_test)
                print('t_test', t_test)

            option_result.append(option_test)

            action_test = agent.predict_actor_option(np.reshape(state_test, (1, agent.state_dim)), option_test)
            p_cluster_test = agent.predict_option(np.reshape(state_test, (1, agent.state_dim)), np.reshape(action_test, (1, agent.action_dim)))
            option_cluster_test = np.argmax(p_cluster_test)
            # print('option', option_test)
            # print('p_cluster_test', p_cluster_test, 'option_cluster', option_cluster_test)
            option_cluster.append(option_cluster_test)

            action_test = action_test.clip(env_test.action_space.low, env_test.action_space.high)
            state_action_i = np.hstack((np.reshape(state_test, (1, agent.state_dim)), np.reshape(action_test, (1, agent.action_dim))))
            # print(state_action_i.shape)
            if len(state_action_pairs) == 0:
                state_action_pairs = state_action_i
                states = np.reshape(state_test, (1, agent.state_dim))
            elif len(state_action_pairs) < 10000:
                state_action_pairs = np.vstack((state_action_pairs,state_action_i))
                states = np.vstack((states,np.reshape(state_test, (1, agent.state_dim))))

            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test[0])
            state_test = state_test2
            return_epi_test = return_epi_test + reward_test

            if terminal_test:
                break

        print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(nn), int(return_epi_test)))
        return_test[test_iter] = return_test[test_iter] + return_epi_test / float(args['test_num'])

    print('{:d}th test: return_test[test_iter] {:d}'.format(int(test_iter), int(return_test[test_iter])))
    result_filename_option = 'results/option/' + result_name + '_option.txt'
    result_filename_option_cluster = 'results/option/' + result_name + '_option_cluster.txt'
    result_filename_state_action = 'results/option/' + result_name + '_state_action_pairs.txt'
    result_filename_states = 'results/option/' + result_name + '_states.txt'

    print('sa.shape', np.asarray(state_action_pairs).shape)
    np.savetxt(result_filename_option, np.asarray(option_result))
    np.savetxt(result_filename_option_cluster, np.asarray(option_cluster))
    np.savetxt(result_filename_state_action, np.asarray(state_action_pairs))
    np.savetxt(result_filename_states, np.asarray(states))

    return  return_test #episode_R


def main(args):
    result_name = 'adInfoHRLTD3_' + args['env'] \
                + '_lambda_' + str(float(args['lambda'])) \
                + '_c_reg_' + str(float(args['c_reg'])) \
                + '_vat_noise_' + str(float(args['vat_noise'])) \
                + '_c_ent_' + str(float(args['c_ent'])) \
                + '_option_' + str(float(args['option_num'])) \
                + '_temporal_' + str(float(args['temporal_num'])) \
                + '_trial_idx_' + str(int(args['trial_idx']))

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
            if args['method_name'] == 'adInfoHRLTD3':
                from adInfoHRL_agent import adInfoHRLTD3
                agent = adInfoHRLTD3(sess, env, state_dim, action_dim, action_bound,
                                     #  , int(args['minibatch_size']),
                                     #  tau=float(args['tau']),
                                     #  actor_lr=float(args['actor_lr']),
                                     #  critic_lr=float(args['critic_lr']),
                                     #  gamma=float(args['gamma']),
                                     #  hidden_dim=np.asarray(args['hidden_dim']),
                                     #  entropy_coeff=float(args['lambda']),
                                     #  c_reg=float(args['c_reg']),
                                     option_num=int(args['option_num']),
                                     #  vat_noise=float(args['vat_noise'])
                                     )

            model_path = "./Model/adInfoHRL/" + args['env'] + '/'
            agent.load_model(iteration=int(args['load_model_iter']), expname=result_name, model_path=model_path)

            test(sess, env, args, agent, result_name)

            if args['use_gym_monitor']:
                 env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRL agent')
    parser.add_argument('--test-num', help='number of episode for recording the return', default=1)
    parser.add_argument('--temporal-num', help='frequency of the gating policy selection', default=3)
    parser.add_argument('--hard-sample-assignment', help='False means soft assignment', default=True)
    parser.add_argument('--option-num', help='number of options', default=5)
    parser.add_argument('--lambda', help='cofficient for the mutual information term', default=0.1)
    parser.add_argument('--c-reg', help='cofficient for regularization term', default=1.0)
    parser.add_argument('--c-ent', help='cofficient for regularization term', default=4.0)
    parser.add_argument('--vat-noise', help='noise for vat in clustering', default=0.04)
    parser.add_argument('--action-noise', help='parameter of the noise for exploration', default=0.2)
    parser.add_argument('--save-model-num', help='number of time steps for saving the network models', default=10000)
    parser.add_argument('--load-model-iter', help='number of iteration of the saved model', default=101)
    parser.add_argument('--save-video', help='save the video or not', default=True)
    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1001)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_adhrlTD3')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_adhrlTD3')
    parser.add_argument('--result-file', help='file name for storing results from multiple trials',
                        default='./results/trials/trials_adInfoHRLTD3')
    parser.add_argument('--method-name', help='method name for recording the results', default='adInfoHRLTD3')
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial-idx', help='index of trials', default=0)
    parser.add_argument('--change-seed', help='change the random seed to obtain different results', default=False)

    parser.set_defaults(total_step_num=1000)
    parser.set_defaults(sample_step_num=200)
    parser.set_defaults(save_model_num=1000)
    parser.set_defaults(load_model_iter=201)
    parser.set_defaults(trial_idx=2)
    parser.set_defaults(option_num=4)

    parser.set_defaults(env='Walker2d-v2')
    parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(render_env=False)

    parser.set_defaults(change_seed=True)
    parser.set_defaults(overwrite_result=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
