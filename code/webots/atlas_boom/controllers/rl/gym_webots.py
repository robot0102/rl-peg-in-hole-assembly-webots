import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import cv2
import time
from controller import Robot, Supervisor


class Atlas(gym.Env):
    """
        Y axis is the vertical axis.
        Base class for Webots actors in a Scene.
        These environments create single-player scenes and behave like normal Gym environments, if
        you don't use multiplayer.
    """

    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    joints_at_limit_cost = -0.2  # discourage stuck joints

    episode_reward = 0

    frame = 0
    _max_episode_steps = 1000

    initial_y = None
    body_xyz = None
    joint_angles = None
    joint_exceed_limit = False
    ignore_frame = 1


    def __init__(self, action_dim, obs_dim):

        self.robot = Supervisor()
        solid_def_names = self.read_all_def()
        self.def_node_field_list = self.get_all_fields(solid_def_names)

        self.robot_node = self.robot.getFromDef('Atlas')

        self.boom_base = self.robot.getFromDef('BoomBase')
        self.boom_base_trans_field = self.boom_base.getField("translation")
        self.timeStep = int(self.robot.getBasicTimeStep() * self.ignore_frame) # ms
        self.find_and_enable_devices()

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def read_all_def(self, file_name ='../../worlds/atlas_change_foot.wbt'):
        no_solid_str_list = ['HingeJoint', 'BallJoint', 'Hinge2Joint', 'Shape', 'Group', 'Physics']
        with open(file_name) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        def_str_list = []
        for x in content:
            if 'DEF' in x:
                def_str_list.append(x)
        for sub_str in no_solid_str_list:
            for i in range(len(def_str_list)):
                if sub_str in def_str_list[i]:
                    def_str_list[i] = 'remove_def'
        solid_def_names = []
        for def_str in def_str_list:
            if 'remove_def' != def_str:
                def_str_temp_list = def_str.split()
                solid_def_names.append(def_str_temp_list[def_str_temp_list.index('DEF') + 1])
        print(solid_def_names)
        print('There are duplicates: ',len(solid_def_names) != len(set(solid_def_names)))
        return solid_def_names

    def get_all_fields(self, solid_def_names):
        def_node_field_list = []
        for def_name in solid_def_names:
            def_node = self.robot.getFromDef(def_name)
            node_trans_field = def_node.getField("translation")
            node_rot_field = def_node.getField("rotation")
            # print(def_name)
            node_ini_trans = node_trans_field.getSFVec3f()
            node_ini_rot = node_rot_field.getSFRotation()

            def_node_field_list.append({'def_node': def_node,
                                        'node_trans_field': node_trans_field,
                                        'node_rot_field': node_rot_field,
                                        'node_ini_trans': node_ini_trans,
                                        'node_ini_rot': node_ini_rot})
        return def_node_field_list

    def reset_all_fields(self):
        for def_node_field in self.def_node_field_list:
            def_node_field['node_trans_field'].setSFVec3f(def_node_field['node_ini_trans'])
            def_node_field['node_rot_field'].setSFRotation(def_node_field['node_ini_rot'])

    def find_and_enable_devices(self):
        # inertial unit
        self.inertial_unit = self.robot.getInertialUnit("inertial unit")
        self.inertial_unit.enable(self.timeStep)

        # gps
        self.gps = self.robot.getGPS("gps")
        self.gps.enable(self.timeStep)

        # foot sensors

        self.fsr = [self.robot.getTouchSensor("RFsr"), self.robot.getTouchSensor("LFsr")]
        for i in range(len(self.fsr)):
            self.fsr[i].enable(self.timeStep)

        # all motors
        # motor_names = [# 'HeadPitch', 'HeadYaw',
        #                'LLegUay', 'LLegLax', 'LLegKny',
        #                'LLegLhy', 'LLegMhx', 'LLegUhz',
        #                'RLegUay', 'RLegLax', 'RLegKny',
        #                'RLegLhy', 'RLegMhx', 'RLegUhz',
        #                ]
        motor_names = [  # 'HeadPitch', 'HeadYaw',
             'BackLbz', 'BackMby', 'BackUbx', 'NeckAy',
             'LLegLax', 'LLegMhx', 'LLegUhz',
             'RLegLax', 'RLegMhx', 'RLegUhz',
        ]
        self.motors = []
        for i in range(len(motor_names)):
            self.motors.append(self.robot.getMotor(motor_names[i]))

        # leg pitch motors
        self.legPitchMotor = [self.robot.getMotor('RLegLhy'),
                              self.robot.getMotor('RLegKny'),
                              self.robot.getMotor('RLegUay'),
                              self.robot.getMotor('LLegLhy'),
                              self.robot.getMotor('LLegKny'),
                              self.robot.getMotor('LLegUay')]

        for i in range(len(self.legPitchMotor)):
            self.legPitchMotor[i].enableTorqueFeedback(self.timeStep)

        # leg pitch sensors
        self.legPitchSensor =[self.robot.getPositionSensor('RLegLhyS'),
                              self.robot.getPositionSensor('RLegKnyS'),
                              self.robot.getPositionSensor('RLegUayS'),
                              self.robot.getPositionSensor('LLegLhyS'),
                              self.robot.getPositionSensor('LLegKnyS'),
                              self.robot.getPositionSensor('LLegUayS')]
        for i in range(len(self.legPitchSensor)):
            self.legPitchSensor[i].enable(self.timeStep)


    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for n, j in enumerate(self.legPitchMotor):
            max_joint_angle = j.getMaxPosition()
            min_joint_angle = j.getMinPosition()
            mean_angle = 0.5 * (max_joint_angle + min_joint_angle)
            half_range_angle = 0.5 * (max_joint_angle - min_joint_angle)
            j.setPosition(mean_angle + half_range_angle * float(np.clip(a[n], -1, +1)))

            # joint_angle = self.read_joint_angle(joint_idx=n)
            # torque = 0.5 * j.getMaxTorque() * float(np.clip(a[n], -1, +1))
            # if joint_angle > max_joint_angle:
            #     j.setPosition(max_joint_angle - 0.1)
            #     # j.setTorque(-1.0 * abs(torque))
            # elif joint_angle < min_joint_angle:
            #     j.setPosition(min_joint_angle + 0.1)
            #     # j.setTorque(abs(torque))
            # else:
            #     j.setTorque(torque)


    def read_joint_angle(self, joint_idx):
        joint_angle = self.legPitchSensor[joint_idx].getValue() % (2.0 * np.pi)
        if joint_angle > np.pi:
            joint_angle -= 2.0 * np.pi
        max_joint_angle = self.legPitchMotor[joint_idx].getMaxPosition()
        min_joint_angle = self.legPitchMotor[joint_idx].getMinPosition()
        if joint_angle > max_joint_angle + 0.05 or joint_angle < min_joint_angle - 0.05:
            self.joint_exceed_limit = True
        return joint_angle

    def calc_state(self):
        joint_states = np.zeros(2*len(self.legPitchMotor))
        # even elements [0::2] position, scaled to -1..+1 between limits
        for r in range(6):
            joint_angle = self.read_joint_angle(joint_idx=r)
            if r in [0, 3]:
                joint_states[2 * r] = (-joint_angle - np.deg2rad(35)) / np.deg2rad(80)
            elif r in [1, 4]:
                joint_states[2 * r] = 1 - joint_angle / np.deg2rad(75)
            elif r in [2, 5]:
                joint_states[2 * r] = -joint_angle / np.deg2rad(45)
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        for r in range(6):
            if self.joint_angles is None:
                joint_states[2 * r + 1] = 0.0
            else:
                joint_states[2 * r + 1] = 0.5 * (joint_states[2*r] - self.joint_angles[r])

        self.joint_angles = np.copy(joint_states[0::2])
        self.joint_speeds = joint_states[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(joint_states[0::2]) > 0.99)

        self.joint_torques = np.zeros(len(self.legPitchMotor))
        for i in range(len(self.legPitchMotor)):
            self.joint_torques[i] = self.legPitchMotor[i].getTorqueFeedback() \
                                    / self.legPitchMotor[i].getAvailableTorque()

        if self.body_xyz is None:
            self.body_xyz = np.asarray(self.gps.getValues())
            self.body_speed = np.zeros(3)
        else:
            self.body_speed = (np.asarray(self.gps.getValues()) - self.body_xyz) / (self.timeStep * 1e-3)
            self.body_xyz = np.asarray(self.gps.getValues())

        body_local_speed = np.copy(self.body_speed)
        body_local_speed[0], body_local_speed[2] = self.calc_local_speed()

        # print('speed: ', np.linalg.norm(self.body_speed))
        y = self.body_xyz[1]
        if self.initial_y is None:
            self.initial_y = y

        self.body_rpy = self.inertial_unit.getRollPitchYaw()
        '''
        The roll angle indicates the unit's rotation angle about its x-axis, 
        in the interval [-π,π]. The roll angle is zero when the InertialUnit is horizontal, 
        i.e., when its y-axis has the opposite direction of the gravity (WorldInfo defines 
        the gravity vector).

        The pitch angle indicates the unit's rotation angle about is z-axis, 
        in the interval [-π/2,π/2]. The pitch angle is zero when the InertialUnit is horizontal, 
        i.e., when its y-axis has the opposite direction of the gravity. 
        If the InertialUnit is placed on the Robot with a standard orientation, 
        then the pitch angle is negative when the Robot is going down, 
        and positive when the robot is going up.

        The yaw angle indicates the unit orientation, in the interval [-π,π], 
        with respect to WorldInfo.northDirection. 
        The yaw angle is zero when the InertialUnit's x-axis is aligned with the north direction, 
        it is π/2 when the unit is heading east, and -π/2 when the unit is oriented towards the west. 
        The yaw angle can be used as a compass.
        '''

        more = np.array([
            y - self.initial_y,
            0, 0,
            0.3 * body_local_speed[0], 0.3 * body_local_speed[1], 0.3 * body_local_speed[2],
            # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            self.body_rpy[0] / np.pi, self.body_rpy[1] / np.pi], dtype=np.float32)

        self.feet_contact = np.zeros(2)
        for j in range(len(self.fsr)):
            self.feet_contact[j] = self.fsr[j].getValue()

        return np.clip(np.concatenate([more] + [joint_states] + [self.feet_contact]), -5, +5)

    def calc_local_speed(self):
        '''
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second,
        this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        '''
        direction_r = self.body_xyz - np.asarray(self.boom_base_trans_field.getSFVec3f())
        # print('robot_xyz: {}, boom_base_xyz: {}'.format(self.body_xyz,
        #                                                 np.asarray(self.boom_base_trans_field.getSFVec3f())))
        direction_r = direction_r[[0, 2]] / np.linalg.norm(direction_r[[0, 2]])
        direction_t = np.dot(np.asarray([[0, 1],
                                  [-1, 0]]), direction_r.reshape((-1, 1)))
        return np.dot(self.body_speed[[0, 2]], direction_t), np.dot(self.body_speed[[0, 2]], direction_r)

    def alive_bonus(self, y, pitch):
        return +1 if abs(y) > 0.5 and abs(pitch) < 1.0 else -1

    def render(self, mode='human'):
        file_name = 'img.jpg'
        self.robot.exportImage(file=file_name, quality=100)
        return cv2.imread(file_name, -1)

    def step(self, action):
        for i in range(self.ignore_frame):
            self.apply_action(action)
            simulation_state = self.robot.step(self.timeStep)
        state = self.calc_state()  # also calculates self.joints_at_limit
        # state[0] is body height above ground, body_rpy[1] is pitch
        alive = float(self.alive_bonus(state[0] + self.initial_y,
                                       self.body_rpy[1]))

        progress, _ = self.calc_local_speed()
        # print('progress: {}'.format(progress))

        feet_collision_cost = 0.0


        '''
        let's assume we have DC motor with controller, and reverse current braking
        '''
        electricity_cost = self.electricity_cost * float(np.abs(
            self.joint_torques * self.joint_speeds).mean())
        electricity_cost += self.stall_torque_cost * float(np.square(self.joint_torques).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]

        self.episode_reward += progress

        self.frame += 1

        done = (-1 == simulation_state) or (self._max_episode_steps <= self.frame) \
               or (alive < 0) or (not np.isfinite(state).all())
        # print('frame: {}, alive: {}, done: {}, body_xyz: {}'.format(self.frame, alive, done, self.body_xyz))
        # print('state_{} \n action_{}, reward_{}'.format(state, action, sum(rewards)))
        return state, sum(rewards), done, {}

    def run(self):
        # Main loop.
        for i in range(self._max_episode_steps):
            action = np.random.uniform(-1, 1, 6)
            state, reward, done, _ = self.step(action)
            # print('state_{} \n action_{}, reward_{}'.format(state, action, reward))
            if done:
                break

    def reset(self, is_eval_only = False):
        self.initial_y = None
        self.body_xyz = None
        self.joint_angles = None
        self.frame = 0
        self.episode_reward = 0
        self.joint_exceed_limit = False

        for i in range(100):
            for j in self.motors:
                j.setPosition(0)
            for k in range(len(self.legPitchMotor)):
                j = self.legPitchMotor[k]
                j.setPosition(0)
            self.robot.step(self.timeStep)
        self.robot.simulationResetPhysics()
        self.reset_all_fields()
        for i in range(10):
            self.robot_node.moveViewpoint()
            self.robot.step(self.timeStep)
        if is_eval_only:
            time.sleep(1)
        return self.calc_state()

