import os
print(os.getcwd())
import sys
project_path = '../../../../../'
sys.path.insert(0, project_path + 'code')
sys.path.insert(0, '/usr/local/webots/lib/python36')
print(sys.path)
from gym_webots import Atlas
import argparse
import datetime
import time
from utils.solver import utils, Solver

def main(env, reward_name = '', policy_name = 'TD3', state_noise = 0.0, seed = 0, load_policy_idx = ''):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default=policy_name)  # Policy name
    parser.add_argument("--env_name", default="WebotsAtlas-v1")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/ATD3_Atlas_new')

    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--render", default=False)

    parser.add_argument("--save_video", default=False)
    parser.add_argument("--video_size", default=(1920, 1080))
    parser.add_argument("--save_all_policy", default=False)
    parser.add_argument("--evaluate_Q_value", default=False)
    parser.add_argument("--load_policy_idx", default=load_policy_idx)
    parser.add_argument("--reward_name", default=reward_name,
                        help='Name of your method (default: )')  # Name of the method

    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--ini_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=seed, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for

    parser.add_argument("--expl_noise", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=state_noise, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    solver = Solver(args, env, project_path)
    if not args.eval_only:
        solver.train()
    else:
        video_dir = '{}/video_all/{}_{}_{}'.format(solver.result_path, solver.args.env_name,
                                                   solver.args.policy_name, solver.args.reward_name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        video_name = video_dir + '/{}_{}_{}.mp4'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            solver.file_name, solver.args.load_policy_idx)
        print(video_name)
        env.robot.movieStartRecording(file=video_name, width=1920, height=1080, codec=0,
                                      quality=50, acceleration=1, caption=False)
        solver.eval_only(is_reset=False)
        env.robot.movieStopRecording()
        time.sleep(5)
        print(env.robot.movieFailed())
        env.reset(is_eval_only=True)


if __name__ == "__main__":
    # env = Atlas(action_dim=6, obs_dim=22)
    # for i in range(5):
    #     env.run()
    #     env.reset()
    env = Atlas(action_dim=6, obs_dim=22)
    reward_name_vec = ['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    policy_name_vec = ['TD3', 'ATD3', 'ATD3_RNN']
    # r_vec = [0, 4, 4, 4]
    # p_vec = [0, 0, 1, 2]
    # for i in range(4):
    #     r = r_vec[i]
    #     p = p_vec[i]
    #     for load_policy_idx in ['']:
    #         main(env, reward_name=utils.connect_str_list(reward_name_vec[:r+1]),
    #              policy_name = policy_name_vec[p], load_policy_idx=load_policy_idx)
    r = 4
    p = 2
    main(env, reward_name=utils.connect_str_list(reward_name_vec[:r + 1]),
         policy_name=policy_name_vec[p])
    env.close()
