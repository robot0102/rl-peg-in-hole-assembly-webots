# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Connect
   Description :   The class for control the robot; especially, the force sensor need to be calibrated firstly
   and test the detected force and moment then.
   Author :       Zhimin Hou
   date：         18-1-7
-------------------------------------------------
   Change Activity:
                   18-1-7
-------------------------------------------------
"""
import socket
import time
import numpy as np
HOST = '192.168.125.1'
PORT = 1502


__all__ = ('Robot_Control',)


class Robot_Control(object):

    def __init__(self):
        """force direction + Z(down), +Y(left), +X(far from ABB)"""
        """The Matrix of Peg and hole"""
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        """The hole in world::::Tw_h=T*Tt_p"""
        self.Tw_h = np.array([[9.99584307e-01, - 1.76482665e-02, 2.27980822e-02, 5.36925589e+02],
                              [1.75973675e-02, 9.99842198e-01, 2.43130956e-03, - 4.00176302e+01],
                              [-2.28373930e-02, - 2.02911265e-03, 9.99737134e-01, 6.16471626e+01],
                              [0.0, 0.0, 0.0, 1.0]])

        """The single hole in world"""
        self.Tw_h_single = np.array([[9.99588567e-01, -1.75915494e-02, 2.26546690e-02,  5.46446162e+02],
                                     [1.75403761e-02, 9.99843140e-01, 2.45559349e-03, -1.04633584e+02],
                                     [-2.26943131e-02, -2.05721177e-03, 9.99740334e-01, 6.28998272e+01],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        """The peg in tool"""
        # self.Tt_p = np.array([[0.992324636460983, -0.123656366940134, -0.000957450195690770, 1.41144492938517],
        #                      [-0.123650357439767, -0.992313999324900, 0.00485764354118817, -0.848850965882200],
        #                      [-0.00155076978095379, -0.00469357455783674, -0.999987743210303, 129.363218736061],
        #                      [0, 0, 0, 1]])

        """tool with single peg """
        self.tool_single_peg = np.array([[1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, -1, 158],
                                        [0, 0, 0, 1]])

        """Second version: we don't need to know epirical position ::::::: Theoritical matrix"""
        self.T_tt = np.array([[1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 130],
                              [0, 0, 0, 1]])

        # """prediction"""
        # self.target_pos = np.array([539.89459, -39.68952, 191.5599])
        # self.target_euler = np.array([179.4948,  0.8913,  -5.9635])
        #
        # """start position for calibrate force sensor"""
        # self.start_pos = np.array([536.2698, -38.8847, 398.0346])
        # self.start_euler = np.array([179.8924, -1.195000e-01, -7.109200])
        #
        # """setting for initial single_assembly"""
        # self.set_initial_pos = np.array([539.8634, -39.6989, 200.0000])
        # self.set_initial_euler = np.array([179.8417,  0.9112, -5.5042])
        #
        # """setting for search phase"""
        # self.set_search_start_pos = np.array([539.8634, -39.6989, 194.7765])
        # self.set_search_start_euler = np.array([179.8417,  0.9112, -5.5042])
        #
        # self.set_search_goal_pos = np.array([539.8549, -39.466, 191.0095])
        # self.set_search_goal_euler = np.array([179.4898, 0.8922, -6.541])
        #
        # """setting for insertion phase"""
        # self.set_insert_start_pos = np.array([539.8549, -39.466, 190.0095])
        # self.set_insert_start_euler = np.array([179.4898, 0.8922, -6.541])
        #
        # self.set_insert_goal_pos = np.array([539.8549, -39.466, 188.0095])
        # self.set_insert_goal_euler = np.array([179.4898, 0.8922, -6.541])
        #
        # """final position and euler"""
        # self.final_pos = np.array([539.88427, -38.68679, 190.03184])
        # self.final_euler = np.array([179.88444, 1.30539, 0.21414])
        # self.final_force = np.array([0.9143, -11.975, -5.3944, 0., 0.0426, -0.1369])

        """ Experiments in Tanjin :: IRB 1600 """
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

        self.set_insert_goal_pos = np.array([1453.2509, 73.2577, 985])
        self.set_insert_goal_euler = np.array([179.8938, 0.9185, 1.0311])

        """final position and euler"""
        self.final_pos = np.array([539.88427, -38.68679, 190.03184])
        self.final_euler = np.array([179.88444, 1.30539, 0.21414])
        self.final_force = np.array([0.9143, -11.975, -5.3944, 0., 0.0426, -0.1369])

        """setting for connect"""
        self.Connect()

    """Connect with robot"""
    def Connect(self):
        self.s.connect((HOST, PORT))

    """Disconnect"""
    def DisConnect(self):
        self.s.close()

    """move line to target offset based on tool coordinate"""
    def MovelineTo_tool_base(self, T, pos_offset, euler_offset, vel):
        """T_tar=T*Tool_peg*T_offset*inv(Tool_peg)"""
        T_offset = self.EulerToMatrix(pos_offset, euler_offset)
        T_tar = np.dot(T, self.tool_single_peg)
        T_tar = np.dot(T_tar, T_offset)
        tempT = np.linalg.inv(self.tool_single_peg)
        T_tar = np.dot(T_tar, tempT)
        pos_tar, euler_tar = self.MatrixToEuler(T_tar)
        pos_cur, euler_cur = self.MatrixToEuler(T)

        """base coordinate"""
        pos_tar1 = pos_cur + pos_offset
        euler_tar1 = euler_cur + euler_offset

        self.SetTarget_1(pos_tar, euler_tar, vel[0])

    """move line to target"""
    def MovelineTo(self, position, euler, vel):

        Q = self.EulerToQuternion(euler)
        swrite = '#SetTar_1 ' + '*' + '[' + str('%0.3f' % position[0]) + ',' + str('%0.3f' % position[1]) + ',' + str(
            '%0.3f' % position[2]) + ']' + '!' + '[' + str('%0.3f' % Q[0]) + ',' + str('%0.3f' % Q[1]) + \
                 ',' + str('%0.3f' % Q[2]) + ',' + str('%0.3f' % Q[3]) + ']' + '!' + str('%0.3f' % vel) + '@'

        self.s.send(swrite.encode())

        recvbuf = ''
        time_out = 0
        while recvbuf.find('MotionFinish') == -1:
            recvbuf = self.s.recv(2048).decode()
            time_out += 1

        print('--------------Move Finish!!!!-------------------')

    """Calibrate the force sensor"""
    def GetCalibTool(self):
        recvbuf = ''
        Euler = np.zeros(3, dtype=float)
        Position = np.zeros(3, dtype=float)

        self.s.send('#GetCalibPar@'.encode())

        time_out = 0
        while len(recvbuf) == 0 and time_out < 20:
            recvbuf += self.s.recv(2048).decode()
            time_out += 1
            # time.sleep(0.001)

        if time_out < 20 and recvbuf[len(recvbuf) - 4:len(recvbuf)] == 'Pose':
            print('Get Pose Success!!!')
        else:
            result = False
            print('Get Pose Fail!')
            return result

        px = recvbuf.find('PX')
        endnum_px = recvbuf.find('*', px)
        py = recvbuf.find('PY')
        endnum_py = recvbuf.find('*', py)
        pz = recvbuf.find('PZ')
        endnum_pz = recvbuf.find('*', pz)
        ex = recvbuf.find('EX')
        endnum_ex = recvbuf.find('*', ex)
        ey = recvbuf.find('EY')
        endnum_ey = recvbuf.find('*', ey)
        ez = recvbuf.find('EZ')
        endnum_ez = recvbuf.find('*', ez)

        Position[0] = round(float(recvbuf[(px + 2):endnum_px]), 4)
        Position[1] = round(float(recvbuf[(py + 2):endnum_py]), 4)
        Position[2] = round(float(recvbuf[(pz + 2):endnum_pz]), 4)
        Euler[0] = round(float(recvbuf[(ex + 2):endnum_ex]), 4)
        Euler[1] = round(float(recvbuf[(ey + 2):endnum_ey]), 4)
        Euler[2] = round(float(recvbuf[(ez + 2):endnum_ez]), 4)
        T = self.EulerToMatrix(Position, Euler)

        return Position, Euler, T

    """Get force and moment"""
    def GetFCForce(self):
        recvbuf = ''
        myForceVector = np.zeros(6, dtype=float)
        self.s.send('#GetFCForce@'.encode())

        time_out = 0
        while len(recvbuf) == 0 and time_out < 20:
            recvbuf = self.s.recv(2048).decode()
            time_out += 1
            time.sleep(0.01)

        if time_out < 20 and recvbuf[len(recvbuf) - 5:len(recvbuf)] == 'Force':
            print('Get Force Success!!!')
        else:
            result = False
            print('Get Force Fail!')
            return result

        fx = recvbuf.find('Fx')
        endnum_fx = recvbuf.find('*', fx)
        fy = recvbuf.find('Fy')
        endnum_fy = recvbuf.find('*', fy)
        fz = recvbuf.find('Fz')
        endnum_fz = recvbuf.find('*', fz)
        tx = recvbuf.find('Tx')
        endnum_tx = recvbuf.find('*', tx)
        ty = recvbuf.find('Ty')
        endnum_ty = recvbuf.find('*', ty)
        tz = recvbuf.find('Tz')
        endnum_tz = recvbuf.find('*', tz)
        myForceVector[0] = round(float(recvbuf[(fx + 2):endnum_fx]), 4)
        myForceVector[1] = round(float(recvbuf[(fy + 2):endnum_fy]), 4)
        myForceVector[2] = round(float(recvbuf[(fz + 2):endnum_fz]), 4)
        myForceVector[3] = round(float(recvbuf[(tx + 2):endnum_tx]), 4)
        myForceVector[4] = round(float(recvbuf[(ty + 2):endnum_ty]), 4)
        myForceVector[5] = round(float(recvbuf[(tz + 2):endnum_tz]), 4)

        return myForceVector

    """move line to the target"""
    def MoveToolTo(self, position, euler, vel):

        # change euler to quter
        Q = self.EulerToQuternion(euler)

        Filecounter = 0
        # send the code head
        swrite = '#FileHead@'
        self.s.send(swrite.encode())

        # The send module
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'MODULE movproc' + '@'
        self.s.send(swrite.encode())

        # The target point
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'CONST robtarget Target_1000:=[[' + str('%0.5f' %position[0]) + ',' + \
                 str('%0.5f' %position[1]) + ',' + str('%0.5f' %position[2]) + '],' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + '[' + str('%0.5f' %Q[0]) + ',' + str('%0.5f' %Q[1]) +\
                 ',' + str('%0.5f' %Q[2]) + ',' + str('%0.5f' %Q[3]) + '],' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + '[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,0]];' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'PROC Path_10()' + chr(10) + '@'
        self.s.send(swrite.encode())

        # SingArea \Wrist
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'SingArea  ' + chr(92) + 'Wrist;' + '@'
        self.s.send(swrite.encode())

        # ConfL \off
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'ConfL ' + chr(92) + 'Off;' + '@'
        self.s.send(swrite.encode())

        # Move intruction
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'MoveL Target_1000,userspeed' + chr(92) + 'V:=' +\
                  str('%.5f' %vel) + ',z100,Tool0' + chr(92) + 'WObj:=wobj0;' + chr(10) + '@'
        self.s.send(swrite.encode())

        # Move Finish
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + ' MovtionFinish;' + '@'
        self.s.send(swrite.encode())

        # Error_Move_Finish
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ERROR' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + ' MovtionFinish;' + '@'
        self.s.send(swrite.encode())

        # The finish of code
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ENDPROC' + '@'
        self.s.send(swrite.encode())

        # The finish of Module
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ENDMODULE' + '@'
        self.s.send(swrite.encode())

        # Send the finish of the document
        Filecounter += 1
        swrite = '#FileEnd@'
        self.s.send(swrite.encode())

        # Receive the message of the robot and send the instruct to control the robot
        recvbuf = ''
        time_out = 0
        while recvbuf.find('Receive Over!') == -1 and time_out < 20:
            recvbuf = self.s.recv(2048).decode()
            time.sleep(0.01)
            time_out += 1
        if time_out < 20:
            pass
        else:
            print('Receive Fail for Time Out!\n')

        self.s.send('#WorkStart@'.encode())

        # Wait the message of the finish signal of the motion work
        recvbuf = ''
        time_out = 0
        while recvbuf.find('MotionFinish') == -1 and time_out < 100:
            recvbuf = self.s.recv(2048).decode()
            time_out += 1
            time.sleep(0.01)

        if time_out < 20:
            pass
        else:
            print('Move Tool Fail for Time Out!\n')

    """calibration the force controller"""
    def CalibFCforce(self):

        # self.MovelineTo(self.start_pos, self.start_euler, 20)
        # self.MovelineTo(np.array([1590, 82.58, 1421]), np.array([0.02717, -0.73902, -0.00856, -0.67308]), 10)
        swrite = '#CalibFCForce@"'
        self.s.send(swrite.encode())
        CalibResult = True

        return CalibResult

    """Euler to matrix"""
    def EulerToMatrix(self, position, euler):
        euler = euler * np.pi/180
        Wx = euler[0]
        Wy = euler[1]
        Wz = euler[2]
        Mz = np.array([[np.cos(Wz), -np.sin(Wz), 0, 0],
                       [np.sin(Wz), np.cos(Wz), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        My = np.array([[np.cos(Wy), 0, np.sin(Wy), 0],
                       [0, 1, 0, 0],
                       [-np.sin(Wy), 0, np.cos(Wy), 0],
                       [0, 0, 0, 1]])
        Mx = np.array([[1, 0, 0, 0],
                       [0, np.cos(Wx), -np.sin(Wx), 0],
                       [0, np.sin(Wx), np.cos(Wx), 0],
                       [0, 0, 0, 1]])
        Matrix_three = np.dot(np.dot(Mz, My), Mx)  # Mz * My * Mx
        Unit = np.array([[1, 0, 0, position[0]],
                         [0, 1, 0, position[1]],
                         [0, 0, 1, position[2]],
                         [0, 0, 0, 1]])
        Matrix = np.dot(Unit, Matrix_three)
        return Matrix

    """Euler to Quternion"""
    def EulerToQuternion(self, euler):
        euler = euler * np.pi/180
        x1 = np.cos(euler[1]) * np.cos(euler[2])
        x2 = np.cos(euler[1]) * np.sin(euler[2])
        x3 = -np.sin(euler[1])

        y1 = -np.cos(euler[0]) * np.sin(euler[2]) + np.sin(euler[0]) * np.sin(euler[1]) * np.cos(euler[2])
        y2 = np.cos(euler[0]) * np.cos(euler[2]) + np.sin(euler[0]) * np.sin(euler[1]) * np.sin(euler[2])
        y3 = np.sin(euler[0]) * np.cos(euler[1])

        z1 = np.sin(euler[0]) * np.sin(euler[2]) + np.cos(euler[0]) * np.sin(euler[1]) * np.cos(euler[2])
        z2 = -np.sin(euler[0]) * np.cos(euler[2]) + np.cos(euler[0]) * np.sin(euler[1]) * np.sin(euler[2])
        z3 = np.cos(euler[0]) * np.cos(euler[1])

        Q = np.zeros(4, dtype=float)
        Q[0] = np.sqrt(x1 + y2 + z3 + 1)/2
        if y3 > z2:
            Q[1] = np.sqrt(x1 - y2 - z3 + 1)/2
        else:
            Q[1] = -np.sqrt(x1 - y2 - z3 + 1)/2

        if z1 > x3:
            Q[2] = np.sqrt(y2 - x1 - z3 + 1)/2
        else:
            Q[2] = -np.sqrt(y2 - x1 - z3 + 1)/2

        if x2 > y1:
            Q[3] = np.sqrt(z3 - x1 - y2 + 1)/2
        else:
            Q[3] = -np.sqrt(z3 - x1 - y2 + 1)/2
        return Q

    """calculate the traslation Matrix T with the position and euler"""
    def MatrixToEuler(self, T):
        Position = np.zeros(3, dtype=float)
        Position[0] = T[0, 3]
        Position[1] = T[1, 3]
        Position[2] = T[2, 3]

        Euler = np.zeros(3, dtype=float)
        Euler[2] = np.arctan2(T[1, 0], T[0, 0])
        Euler[1] = np.arctan2(-T[2, 0], np.cos(Euler[2]) * T[0, 0] + np.sin(Euler[2]) * T[1, 0])
        Euler[0] = np.arctan2(np.sin(Euler[2]) * T[0, 2] - np.cos(Euler[2]) * T[1, 2],
                         -np.sin(Euler[2]) * T[0, 1] + np.cos(Euler[2]) * T[1, 1])

        Euler = Euler * 180 / np.pi
        return Position, Euler

    """move line to target, "SetTarget_1" to send move command """
    def SetTarget_1(self, position, euler, vel):
        Q = self.EulerToQuternion(euler)
        swrite = '#SetTar_1 ' + '*' + '[' + str('%0.3f' % position[0]) + ',' + str(
            '%0.3f' % position[1]) + ',' + str(
            '%0.3f' % position[2]) + ']' + '!' + '[' + str('%0.3f' % Q[0]) + ',' + str('%0.3f' % Q[1]) + \
                 ',' + str('%0.3f' % Q[2]) + ',' + str('%0.3f' % Q[3]) + ']' + '!' + str('%0.2f' % vel) + '@'
        # print(swrite)
        self.s.send(swrite.encode())
        # print('-------------------------SetTarget_1--------------------------')
        # Wait the message of the finish signal of the motion work
        time_out = 0
        recvbuf = ''

        if self.MonitorFlag.value == 1:
            """ Monitor Process Open """
            while self.MotionFinishFlag.value == 0 and time_out < 10000:
                time_out += 1
                time.sleep(0.1)

            self.MotionFinishFlag.value = 0

            if time_out > 10000:
                print('Time out for MotionFinish signal !')
            else:
                print('---------------------Monitor Motion Finish!!!!------------------------\n')
        else:
            """ Monitor Process Off """
            while recvbuf.find('MotionFinish') == -1:
                recvbuf = self.s.recv(2048).decode()
                time_out += 1
                # time.sleep(0.01)
            print('---------------------Motion Finish!!!!------------------------\n')


"""calibrate the force sensor and moment"""
def Test_calibrate():

    """define a robot_control class"""
    Controller = Robot_Control()

    """Calibrate the force sensor"""
    # print('===================== Calibrate the force-moment sensor =========================')
    # done = Controller.CalibFCforce()
    #
    # print('=================================================================================')

    """ Test the calibrated force sensor"""
    # print('===================== show the result of F/T sensor =============================')
    # force = Controller.GetFCForce()
    # print(force)
    # print('=================================================================================')

    """Align peg and hole"""
    # position_t2, eulerang_t2 = Controller.Align_PegHole()
    # print('======================== Position and Force Information =========================')
    # print('position', position_t2)
    # print('eulerang', eulerang_t2)
    # force = Controller.GetFCForce()
    # print('force', force)
    # print('=================================================================================')

    """used to search the initial position and euler; please note the other code"""
    print('======================== Position and Force Information =========================')
    position, euler, T = Controller.GetCalibTool()
    print('position', position)
    print('eulerang', euler)
    Controller.MovelineTo(Controller.set_initial_pos + [0., 0., 50.], Controller.set_initial_euler + [0., 0., 0.], 20)
    # Controller.MovelineTo(Controller.set_search_start_pos, Controller.set_search_start_euler, 10)
    # Controller.MovelineTo(position + [0., 0., 2], euler + [0., 0., 0.], 10)
    # force = Controller.GetFCForce()
    # print('force', force)
    print('=================================================================================')

    # # Controller.MoveToolTo(position + [0, 0, 0], euler + [0, 0, 0.], 20)
    # Controller.MovelineTo(Controller.set_search_pos+[0., 0., 0], Controller.set_search_euler + [-0., 0., -0.], 5)
    # force = Controller.GetFCForce()
    # print(force)

    # position, euler, T = Controller.GetCalibTool()
    # print('position', position)
    # print('eulerang', euler)
    # T_z = Controller.EulerToMatrix(np.array([0., 0., 0.]), np.array([0., 0., -90]))
    # T_final = np.dot(T, T_z)
    #
    # pos, euler = Controller.MatrixToEuler(T_final)
    # print(pos)
    # Controller.MoveToolTo(pos, euler, 5)

    # Controller.MoveJointTo([0., 0., 0., .0, 0., 10], 5)
    # position, euler, T = Controller.GetCalibTool()
    # print('position', position)
    # print('euler', euler)
    #
    # force = Controller.GetFCForce()
    # print('force', force)
    # print('=================================================================================')

    # print('===================calculate the initial position and orientation================')
    # print('T', T)
    # T = np.array(T)
    # print('Tw_h', np.dot(T, Controller.T_tt))

    """move to last initial position and orientation"""
    # print('================= Move to the initial position and orientation ==================')
    # Controller.MoveToolTo(Controller.start_insert_pos, Controller.start_insert_euler, 5)
    # position, euler, T = Controller.GetCalibTool()
    # print('position', position)
    # print('euler', euler)
    # force = Controller.GetFCForce()
    # print('force', force)
    # print('=================================================================================')

    """Get the position and orientation of pegs"""
    # print(np.dot(T, Controller.tool_single_peg))


if __name__ == "__main__":
    Test_calibrate()