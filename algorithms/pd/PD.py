import numpy as np


class PD:
    def __init__(self):
        # self.kp = np.array([0.01, 0.01, 0.001, 0.0001, 0.0001, 0.005])
        # self.kd = np.array([0.005, 0.005, 0.001, 0.0001, 0.0001, 0.005])
        # self.kp = np.array([0.01, 0.01, 0.005, 0.0001, 0.0001, 0.005])
        # self.kd = np.array([0.005, 0.005, 0.005, 0.0001, 0.0001, 0.005])
        self.kp = np.array([0.007, 0.007, 0.03, 0.0001, 0.0001, 0.005])
        self.kd = np.array([0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005])
        # self.kp = np.array([0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005])
        # self.kd = np.array([0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005])
        self.uk = np.array([0, 0, 0, 0, 0, 0])
        self.uk_1 = np.array([0, 0, 0, 0, 0, 0])
        self.yk = np.array([0, 0, 0, 0, 0, 0])
        self.ek = np.array([0, 0, 0, 0, 0, 0])
        self.ek_1 = np.array([0, 0, 0, 0, 0, 0])
        self.ek_2 = np.array([0, 0, 0, 0, 0, 0])

    def cal(self, s, rk):
        yk = np.array([s[6], s[7], s[8], s[9], s[10], s[11]])
        self.ek = rk - yk
        # discrete PD algorithm
        self.uk = self.uk_1 + self.kp * (self.ek - self.ek_1) + self.kd * (self.ek - 2 * self.ek_1 + self.ek_2)
        # renew variables
        self.ek_2 = self.ek_1
        self.ek_1 = self.ek
        self.uk_1 = self.uk
        action = self.uk
        for i in range(6):
            action[i] = round(action[i], 4)
        return action

    def clear(self):
        # self.kp = np.array([0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005])
        # self.kd = np.array([0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005])
        # self.kp = np.array([0.01, 0.01, 0.005, 0.0001, 0.0001, 0.005])
        # self.kd = np.array([0.005, 0.005, 0.005, 0.0001, 0.0001, 0.005])
        self.kp = np.array([0.007, 0.007, 0.03, 0.0001, 0.0001, 0.005])
        self.kd = np.array([0.01, 0.01, 0.05, 0.0001, 0.0001, 0.005])
        self.uk = np.array([0, 0, 0, 0, 0, 0])
        self.uk_1 = np.array([0, 0, 0, 0, 0, 0])
        self.yk = np.array([0, 0, 0, 0, 0, 0])
        self.ek = np.array([0, 0, 0, 0, 0, 0])
        self.ek_1 = np.array([0, 0, 0, 0, 0, 0])
        self.ek_2 = np.array([0, 0, 0, 0, 0, 0])