import numpy as np
import copy as cp
from baselines.deepq.assembly.Connect_Finall import Robot_Control
from gym import spaces
from .fuzzy_control import fuzzy_control


class env_search_control(object):
    def __init__(self, step_max=100, fuzzy=False, add_noise=False):

        self.observation_dim = 12
        self.action_dim = 6

        """ state """
        self.state = np.zeros(self.observation_dim)
        self.next_state = np.zeros(self.observation_dim)
        self.init_state = np.zeros(self.observation_dim)

        """ action """
        self.action = np.zeros(self.action_dim)
        self.action_low_bound = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
        self.action_high_bound = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.fuzzy_control = fuzzy

        """ reward """
        self.step_max = step_max
        self.step_max_pos = 15
        self.reward = 1.

        """setting"""
        self.add_noise = add_noise  # or True
        self.pull_terminal = False

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

        """The desired force and moments :: get the force"""
        self.desired_forces_moments = np.array([[0, 0, -50, 0, 0, 0],
                                                [0, 0, -80, 0, 0, 0],
                                                [0, 0, 60, 0, 0, 0],
                                                [0, 0, -80, 0, 0, 0],
                                                [0, 0, -90, 0, 0, 0],
                                                [0, 0, -100, 0, 0, 0]])
        self.desired_force_moment = self.desired_forces_moments[0, :]
        self.pull_desired_force_moment = self.desired_forces_moments[2, :]

        """The force and moment"""
        self.max_force_moment = [70, 5]
        self.safe_force_search = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """parameters for search phase"""
        self.kp = np.array([0.025, 0.025, 0.002])
        self.kd = np.array([0.005, 0.005, 0.0002])
        self.kr = np.array([0.02, 0.02, 0.02])
        self.kv = 0.5
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6

        """information for action and state"""
        self.state_high = np.array([50, 50, 0, 5, 5, 6, 1453, 70, 995, 5, 5, 6])
        self.state_low = np.array([-50, -50, -50, -5, -5, -6, 1456, 76, 985, -5, -5, -6])
        self.terminated_state = np.array([30, 30, 30, 2, 2, 2])
        self.action_space = spaces.Box(low=self.action_low_bound, high=self.action_high_bound,
                                       shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high,
                                            shape=(self.observation_dim,), dtype=np.float32)

        """fuzzy parameters"""
        self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
                                high_output=np.array([0.03, 0.03, 0.004, 0.03, 0.03, 0.03]))

        """ Setting for initial position and orientation """
        self.set_initial_pos = np.array([1453.2509, 73.2577, 1000])
        self.set_initial_euler = np.array([179.8938, 0.9185, 1.0311])

        """ setting for insertion phase """
        self.set_search_start_pos = np.array([1453.2509, 73.2577, 995.8843])
        self.set_search_start_euler = np.array([179.8938, 0.9185, 1.0311])

        self.set_search_goal_pos = np.array([1453.2509, 73.2577, 990])
        self.set_search_goal_euler = np.array([179.8938, 0.9185, 1.0311])

        """ setting for insertion phase """
        self.set_insert_start_pos = np.array([1453.2509, 73.2577, 990])
        self.set_insert_start_euler = np.array([179.8938, 0.9185, 1.0311])

        self.set_insert_goal_pos = np.array([1453.2509, 73.2577, 980])
        self.set_insert_goal_euler = np.array([179.8938, 0.9185, 1.0311])
        self.set_insert_goal = np.array([1453.2509, 73.2577, 980, 179.8938, 0.9185, 1.0311])

        """ random number generator """
        self.rng = np.random.RandomState(5)

    """reset initial state"""
    def reset(self):
        """reset the initial parameters"""

        self.pull_terminal = False
        self.kp = np.array([0.025, 0.025, 0.002])
        self.kd = np.array([0.005, 0.005, 0.0002])
        self.kr = np.array([0.02, 0.02, 0.02])
        self.kv = 0.5
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6
        self.desired_force_moment = self.desired_forces_moments[0, :]

        """pull peg up"""
        self.__pull_peg_up()

        if self.pull_terminal:
            pass
        else:
            exit("+++++++++++++++++++The pegs didn't move the init position!!!+++++++++++++++++++++")

        self.robot_control.MovelineTo(self.set_initial_pos, self.set_initial_euler, 20)

        if self.add_noise:
            """add randomness for the initial position and orietation"""
            state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0., 0., 0., 0.])

            initial_pos = self.set_search_start_pos + state_noise[0:3]
            inital_euler = self.set_search_start_euler + state_noise[3:6]
        else:
            initial_pos = self.set_search_start_pos
            inital_euler = self.set_search_start_euler

        self.robot_control.MovelineTo(initial_pos, inital_euler, 5)

        """Get the max force and moment"""
        myForceVector = self.robot_control.GetFCForce()
        max_fm = np.array([max(abs(myForceVector[0:3])), max(abs(myForceVector[3:6]))])
        self.safe_or_not = all(max_fm < self.max_force_moment)

        if self.safe_or_not is not True:
            self.__pull_peg_up()
            exit("++++++++++++++++++The pegs can't move for the exceed force!!!+++++++++++++++++++")

        self.state = self.__get_state()
        print("++++++++++++++++++++++++++++ Reset Finished !!! +++++++++++++++++++++++++++++")
        print('initial state :::::', self.state)

        return self.code_state(self.state), self.state, self.pull_terminal

    """execute action by robot"""
    def step(self, action, step_num):
        """clip the action"""
        # action = np.clip(action, self.action_low_bound, self.action_high_bound)

        """get the PD basic action"""
        movePosition, setVel = self.__expert_action(step_num)
        executeAction = movePosition + movePosition * action[0]

        # print('executeAction', executeAction)

        max_abs_F_M = np.array([max(abs(self.state[0:3])), max(abs(self.state[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.max_force_moment)

        """move robot"""
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_Force_Moment:::', self.state[:6])
            print("++++++++++++++++++++++++ The force is too large and pull up !!!!!+++++++++++++++++++++++")
            self.__pull_peg_up()
        else:
            """Move and rotate the pegs"""
            self.robot_control.MovelineTo(self.state[6:9] + executeAction[:3], self.state[9:12] + executeAction[3:], setVel)
            print('setPosition:::', executeAction[:3])
            print('setEuLer: ', executeAction[3:])
            print('force', self.state[:6])
            print('state', self.state[6:])

        self.next_state = self.__get_state()
        reward, done = self.__get_reward(self.next_state, step_num)

        return self.code_state(self.next_state), self.next_state, reward, done, self.safe_or_not, executeAction

    """ normalize state """
    def code_state(self, current_state):
        state = cp.deepcopy(current_state)

        if state[9] > 0 and state[9] < 180:
            state[9] -= 180
        elif state[9] < 0 and state[9] > -180:
            state[9] += 180
        else:
            pass

        """normalize the state"""
        scale = self.state_high - self.state_low
        final_state = (state - self.state_low) / scale

        return final_state

    """ ilqg cost """
    def get_running_cost(self, x, u):

        target_x = self.code_state(self.set_insert_goal)
        return (x[8] - target_x[2]) + 1/2 * np.linalg.norm(u)

    """ denormalize state """
    def decode_state(self, obs):
        """denormalize the state"""

        scale = self.state_high - self.state_low
        state = obs * scale + self.state_low

        return state

    """ setting for seed """
    def seed(self, seed=None):
        """Seed the environment"""

        if seed is not None:
            self.rng = np.random.RandomState(seed)

    """ Get the expert action by PD controller"""
    def __expert_action(self, step_num):

        force_desired = self.desired_force_moment
        force = self.state[:6]

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.max_force_moment)

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        """insert phase"""
        if self.state[8] < self.set_insert_start_pos[2]:
            print("+++++++++++++++++++++++++++++++++ The insert Phase !!!! +++++++++++++++++++++++++++++++++")
            self.kp = np.array([0.002, 0.002, 0.015])
            self.kd = np.array([0.0002, 0.0002, 0.0002])
            self.desired_force_moment = self.desired_forces_moments[1, :]
        else:
            if self.fuzzy_control:
                self.kp = self.fc.get_output(force)[:3]
                self.kr = self.fc.get_output(force)[3:]
            print("+++++++++++++++++++++++++++++++++ The search phase !!!! +++++++++++++++++++++++++++++++++")

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

        """Set the velocity of robot"""
        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)
        movePosition = np.zeros(6)
        movePosition[:3] = setPosition
        movePosition[3:6] = setEuler

        return movePosition, setVel

    """Get the expert action by PD controller"""
    def __pd_controller(self):

        force_desired = self.pull_desired_force_moment
        force = self.state[:6]

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.max_force_moment)

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        setPosition = self.kp * force_error[:3]
        self.former_force_error = force_error

        """Get the euler"""
        setEuler = self.kr * force_error[3:6]

        """Set the velocity of robot"""
        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)

        movePosition = np.zeros(6)
        movePosition[:3] = setPosition
        movePosition[3:6] = setEuler

        return movePosition, setVel

    """get reward"""
    def __get_reward(self, state, step_num):
        done = False
        force = state[:6]
        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_force_moment:', force)
            self.reward = -1 + state[8]/self.robot_control.set_insert_goal_pos[2]
            print("-------------------------------- The force is too large!!! -----------------------------")
        else:
            """consider force and moment"""
            self.reward = -0.1

        if state[8] < self.robot_control.set_insert_goal_pos[2]:
            print("+++++++++++++++++++++++++++++ The Assembly Phase Finished!!! ++++++++++++++++++++++++++++")
            self.reward = 1 - step_num / self.step_max
            done = True

        return self.reward, done

    """get the current state"""
    def __get_state(self):

        force = self.robot_control.GetFCForce()
        position, euler, T = self.robot_control.GetCalibTool()

        self.state[:6] = force
        self.state[6:9] = position
        self.state[9:12] = euler

        return self.state

    """execute position control"""
    def __positon_control(self, target_position):

        E_z = np.zeros(30)
        action = np.zeros((30, 3))
        """Move by a little step"""
        for i in range(30):

            myForceVector = self.robot_control.GetFCForce()

            if max(abs(myForceVector[0:3])) > 5:
                exit("The pegs can't move for the exceed force!!!")

            """"""
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()

            E_z[i] = target_position[2] - Position[2]

            if i < 3:
                action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
                vel_low = self.kv * abs(E_z[i])
            else:
                # action[i, :] = np.array([0., 0., action[i-1, 2] + self.Kp_z_0*(E_z[i] - E_z[i-1])])
                action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
                vel_low = min(self.kv * abs(E_z[i]), 0.5)

            self.robot_control.MovelineTo(Position + action[i, :], Euler, vel_low)
            # print(action[i, :])

            if abs(E_z[i]) < 0.001:
                print("The pegs reset successfully!!!")
                self.init_state[0:6] = myForceVector
                self.init_state[6:9] = Position
                self.init_state[9:12] = Euler
                break

        return self.init_state

    """pull peg up"""
    def __pull_peg_up(self):
        Vel_up = 20

        Position, Euler, T = self.robot_control.GetCalibTool()
        self.robot_control.MovelineTo(Position + np.array([0., 0., 0.2]), Euler, Vel_up)
        self.robot_control.MovelineTo(Position + np.array([0., 0., 0.5]), Euler, Vel_up)

        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MovelineTo(Position + np.array([0., 0., 1.]), Euler, Vel_up)

            """finish or not"""
            if Position[2] > self.set_initial_pos[2]:
                self.pull_terminal = True
                print("++++++++++++++++++++++++++ Pull up the pegs finished!!! ++++++++++++++++++++++++")
                break
