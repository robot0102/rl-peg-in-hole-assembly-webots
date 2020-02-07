import pybullet_envs, roboschool, gym
import time
import cv2
env = gym.make("HumanoidFlagrunBulletEnv-v0")
# env._render_width = 1280
# env._render_height = 720
env.render(mode="human")
obs = env.reset()

for i in range(10000):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    # env.render('rgb_array')
    img = env.render(mode='human')
    # cv2.imshow('img', img)
    cv2.waitKey(1)
    # print(img.shape)
