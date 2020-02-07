import numpy as np
import roboschool, gym
import torch
import argparse
import os
import datetime
import utils
import TD3
import OurDDPG
import DDPG
import cv2
from scipy import signal


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    # print ("---------------------------------------"                      )
    # print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print ("---------------------------------------"                      )
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="RoboschoolWalker2d-v1")  # OpenAI gym environment name
    parser.add_argument("--method_name", default="early_stop",
                        help='Name of your method (default: )')  # Name of the method
    parser.add_argument("--eval_only", default=True)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3.1e5, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    result_path = "../../results"
    video_dir = '{}/video/{}_{}'.format(result_path, args.env_name, args.method_name)
    model_dir = '{}/models/TD3/{}_{}'.format(result_path, args.env_name, args.method_name)
    if args.save_models and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.save_video and not os.path.exists(video_dir):
        os.makedirs(video_dir)


    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)

    policy.load("%s" % (file_name), directory=model_dir)

    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = video_dir + '/{}_TD3_{}.mp4'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env_name)
        out_video = cv2.VideoWriter(video_name, fourcc, 60.0, (640, 480))
        print(video_name)

    human_joint_angle = utils.read_table()
    for i in range(1):
        obs = env.reset()
        done = False
        pre_foot_contact = 1
        foot_contact = 1
        foot_contact_vec = np.asarray([1, 1, 1])
        gait_num = 0
        joing_angle_list = []
        coe_list = []
        joint_angle = np.zeros((0,6))
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            utils.fifo_list(foot_contact_vec, obs[-2])
            if 0 == np.std(foot_contact_vec):
                foot_contact = np.mean(foot_contact_vec)
            if 1 == (foot_contact - pre_foot_contact):
                if joint_angle.shape[0] > 20:
                    gait_num += 1
                    if gait_num >= 2:
                        joint_angle_sampled = signal.resample(joint_angle[:-(foot_contact_vec.shape[0]-1)],
                                                              num=human_joint_angle.shape[0])
                        coefficient = utils.calc_cos_similarity(human_joint_angle,
                                                                joint_angle_sampled)
                        print('gait_num:', gait_num, 'time steps in a gait', joint_angle.shape[0],
                              'coefficient', coefficient)
                        coe_list.append(coefficient)
                        joing_angle_list.append(joint_angle_sampled)
                joint_angle = joint_angle[-(foot_contact_vec.shape[0]-1):]
            pre_foot_contact = foot_contact
            joing_angle_obs = np.zeros((1, 6))
            joing_angle_obs[0,:] = obs[8:20:2]
            joint_angle = np.r_[joint_angle, joing_angle_obs]
            if args.save_video:
                img = env.render(mode='rgb_array')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out_video.write(img)
            else:
                env.render()

        idx = np.argmax(np.asarray(coe_list))
        utils.plot_joint_angle(joing_angle_list[idx], human_joint_angle)
    env.close()
    if args.save_video:
        out_video.release()