import os
print(os.getcwd())
import sys
project_path = './'
sys.path.append("/usr/local/webots/lib")
sys.path.insert(0, project_path + 'code')
print(sys.path)
from envs.env import ArmEnv
import argparse
import numpy as np
from code.utils.solver import utils, Solver


# def test_env(env):
#     env.reset()
#     state = np.random.rand(22)
#     print(env.set_robot(state) - state)
#     while True:
#         env.render()


def main(env, args):
    solver = Solver(args, env, project_path)
    if not args.eval_only:
        solver.train()
    else:
        solver.eval_only()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default='TD3')  # Policy name
    parser.add_argument("--env_name", default="Peg-in-hole-single_assembly")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/single_assembly')

    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--render", default=False)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--video_size", default=(600, 400))
    parser.add_argument("--save_all_policy", default=True)
    parser.add_argument("--load_policy", default=False)
    parser.add_argument("--load_policy_idx", default='100000')
    parser.add_argument("--evaluate_Q_value", default=False)
    parser.add_argument("--reward_name", default='r_s')
    parser.add_argument("--discount_low", default=0.99, type=float)  # Discount factor
    parser.add_argument("--discount_high", default=0.99, type=float)  # Discount factor

    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--ini_seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e3,
                        type=int)  # How many time steps purely random policy is run for

    parser.add_argument("--auxiliary_reward", default=False)
    parser.add_argument("--option_num", default=4, type=int)
    
    parser.add_argument("--option_buffer_size", default=5000, type=int)  # Batch size for both actor and critic
    parser.add_argument("--option_batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--policy_batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--critic_batch_size", default=400, type=int)  # Batch size for both actor and critic
    parser.add_argument("--upper_critic_batch_size", default=200, type=int)  # Batch size for both actor and critic

    parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=0, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.2, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_episode_steps", default=200, type=int)

    parser.add_argument("--average_steps", default=20, type=int)

    args = parser.parse_args()

    env = ArmEnv()
    policy_name_vec = ['Average_TD3']
    average_steps = [2]
    for policy_name in policy_name_vec:
        for num_steps in average_steps:
            args.average_steps = num_steps
            for i in range(0, 1):
                args.policy_name = policy_name
                args.seed = i
                main(env, args)
