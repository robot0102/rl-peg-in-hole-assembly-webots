# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Env_robot_control
   Description :  The class for real-world experiments to control the ABB robot,
                    which base on the basic class connect finally
   Author :       Zhimin Hou
   date：         18-1-9
-------------------------------------------------
   Change Activity:
                   18-1-9
-------------------------------------------------
"""
from gym import spaces
import numpy as np
import copy as cp


class env_search_control(object):

    def __init__(self):
        """state and Action Parameters"""
        self.observation_dim = 12
        self.action_dim = 5
        self.state = np.zeros(self.observation_dim)
        self.next_state = np.zeros(self.observation_dim)
        self.action = np.zeros(self.action_dim)
        self.reward = 1.
        self.add_noise = True
        self.pull_terminal = False
        self.fuzzy_control = True
        self.step_max = 50
        self.step_max_pos = 15
        self.max_action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The desired force and moments :: get the force"""
        """action = [0, 1, 2, 3, 4]"""
        self.desired_force_moment = np.array([[0, 0, -40, 0, 0, 0],
                                              [0, 0, -40, 0, 0, 0],
                                              [0, 0, -40, 0, 0, 0],
                                              [0, 0, -40, 0, 0, 0],
                                              [0, 0, -40, 0, 0, 0]])

        """The force and moment"""
        self.max_force_moment = [50, 5]
        self.safe_force_search = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """parameters for search phase"""
        self.kp = np.array([0.01, 0.01, 0.0015])
        self.kd = np.array([0.005, 0.005, 0.0002])
        self.kr = np.array([0.015, 0.015, 0.015])
        self.kv = 0.5
        self.k_former = 0.9
        self.k_later = 0.2

        """information for action and state"""
        self.high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 5])
        self.low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -5])
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high)

        """build a fuzzy control system"""
        self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
                                high_output=np.array([0.022, 0.022, 0.015, 0.015, 0.015, 0.015]))

    def reset(self):

        # judge whether need to pull the peg up
        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()
        if Position_0[2] < self.robot_control.start_pos[2]:
            print("++++++++++++++++++++++ The pegs need to be pull up !!! +++++++++++++++++++++++++")
            self.pull_peg_up()

        """add randomness for the initial position and orietation"""
        state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3),
                                np.random.uniform(-0.3, 0.3), 0., 0., 0.])
        if self.add_noise:
            initial_pos = self.robot_control.start_pos + state_noise[0:3]
            inital_euler = self.robot_control.start_euler + state_noise[3:6]
            print("add noise to the initial position")
        else:
            initial_pos = self.robot_control.start_pos
            inital_euler = self.robot_control.start_euler

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MoveToolTo(initial_pos, inital_euler, 10)

        """Get the max force and moment"""
        myForceVector = self.robot_control.GetFCForce()
        max_fm = np.array([max(abs(myForceVector[0:3])), max(abs(myForceVector[3:6]))])

        safe_or_not = all(max_fm < self.max_force_moment)
        if safe_or_not is not True:
            exit("The pegs can't move for the exceed force!!!")

        done = self.positon_control()

        print("++++++++++++++++++++++++++++ Reset Finished !!! +++++++++++++++++++++++++++++")
        self.state = self.get_state()
        return self.get_obs(self.state), done

    def step(self, action, step_num):

        """choose one action from the different actions vector"""
        done = False
        force_desired = self.desired_force_moment[action, :]
        self.reward = -0.1
        force = self.state[:6]
        state = self.state[6:]

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if self.fuzzy_control:
            self.kp = self.fc.get_output(force)[:3]
            self.kr = self.fc.get_output(force)[3:]

        if step_num == 0:
            setPosition = self.kp * force_error[:3]
            self.former_force_error = force_error
        elif step_num == 1:
            setPosition = self.kp * force_error[:3]
            self.last_setPosition = setPosition
            self.last_force_error = force_error
        else:
            setPosition = self.last_setPosition + self.kp * (force_error[:3] - self.last_force_error[:3]) + \
                          self.kd * (force_error[:3] - 2 * self.last_force_error[:3] + self.former_force_error[:3])
            self.last_setPosition = setPosition
            self.former_force_error = self.last_force_error
            self.last_force_error = force_error

        """Get the euler"""
        setEuler = self.kr * force_error[3:6]

        # set the velocity of robot
        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.max_force_moment)

        movePosition = np.zeros(self.action_dim + 1)

        movePosition[2] = setPosition[2]
        if action < 2:
            movePosition[action] = setPosition[action]
        else:
            movePosition[action + 1] = setEuler[action - 2]

        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', force)
            self.reward = -1
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """Move and rotate the pegs"""
            self.robot_control.MoveToolTo(state[:3] + movePosition[:3], state[3:] + movePosition[3:], setVel)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('setPosition: ', setPosition)
            print('setEuLer: ', setEuler)
            print('force', force)

        if state[2] < self.robot_control.final_pos[2]:
            print("+++++++++++++++++++++++++++++ The Search Phase Finished!!! ++++++++++++++++++++++++++++")
            self.reward = 1 - step_num / self.step_max
            done = True

        self.next_state = self.get_state()
        return self.get_obs(self.next_state), self.reward, done, self.safe_or_not

    def get_state(self):
        force = self.robot_control.GetFCForce()
        position, euler, T = self.robot_control.GetCalibTool()

        self.state[:6] = force
        self.state[6:9] = position
        self.state[9:12] = euler
        return self.state

    def get_obs(self, current_state):
        state = cp.deepcopy(current_state)

        if state[9] > 0 and state[9] < 180:
            state[9] -= 180
        elif state[9] < 0 and state[9] > -180:
            state[9] += 180
        else:
            pass

        # normalize the state
        scale = self.high - self.low
        final_state = (state - self.low) / scale
        return final_state

    def positon_control(self):
        step_num = 0
        pos_error = np.zeros(3)
        while True:
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            force = self.robot_control.GetFCForce()
            print('Force', force)

            """Get the current position and euler"""
            Tw_p = np.dot(Tw_t, self.robot_control.T_tt)

            pos_error[2] = self.robot_control.target_pos[2] - Tw_p[2, 3] - 130

            if step_num == 0:
                setPostion = self.k_former * pos_error
                self.former_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            elif step_num == 1:
                setPostion = self.k_former * pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 10)
            else:
                setPostion = self.k_former * pos_error
                # setPostion = self.k_later * (pos_error - self.last_pos_error)
                # self.k_later * (pos_error - 2 * self.last_pos_error + self.former_pos_error)
                self.former_pos_error = self.last_pos_error
                self.last_pos_error = pos_error
                self.robot_control.MoveToolTo(Position + setPostion, Euler, 0.5)

            max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
            safe_or_not = any(max_abs_F_M > self.safe_force_search)

            if safe_or_not:
                print("Position Control finished!!!")
                return True

            if step_num > self.step_max_pos:
                print("Position Control failed!!!")
                return True
            step_num += 1

    def pull_peg_up(self):
        Vel_up = 5
        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MoveToolTo(Position + np.array([0., 0., 1]), Euler, Vel_up)

            """finish or not"""
            if Position[2] > self.robot_control.start_pos[2]:
                print("=====================Pull up the pegs finished!!!======================")
                self.pull_terminal = True
                break
        return self.pull_terminal
