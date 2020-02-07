# -*- coding: utf-8 -*-
"""
# @Time    : 23/10/18 9:10 PM
# @Author  : ZHIMIN HOU
# @FileName: fuzzy_control.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import argparse
import numpy as np
import copy as cp
import skfuzzy.control as ctrl


class fuzzy_control(object):

    # the input is six forces and moments
    # the output is the hyperpapermeters [Kpz, kpx, kpy, krx, kry, krz]
    def __init__(self,
                 low_input=np.array([-1, -1, -1, -1, -1, -1]),
                 high_input=np.array([1, 1, 1, 1, 1, 1]),
                 # low_input=np.array([-40, -40, -40, -5, -5, -5]),
                 # high_input=np.array([40, 40, 0, 5, 5, 5]),
                 low_output=np.array([0., 0., 0., 0., 0., 0.]),
                 high_output=np.array([0.015, 0.015, 0.02, 0.015, 0.015, 0.015])):

        self.low_input = low_input
        self.high_input = high_input
        self.low_output = low_output
        self.high_output = high_output
        self.num_input = 5
        self.num_output = 3
        self.num_mesh = 21

        self.sim_kpx = self.build_fuzzy_kpx()
        self.sim_kpy = self.build_fuzzy_kpy()
        self.sim_kpz = self.build_fuzzy_kpz()

        self.sim_krx = self.build_fuzzy_krx()
        self.sim_kry = self.build_fuzzy_kry()
        self.sim_krz = self.build_fuzzy_krz()
        # self.sim_kpx, self.sim_kpy, self.sim_kpz, \
        # self.sim_krx, self.sim_kry, self.sim_krz = self.build_fuzzy_system()

    def get_output(self, force):

        self.sim_kpx.input['fx'] = force[0]
        self.sim_kpx.input['my'] = force[4]
        self.sim_kpx.compute()
        kpx = self.sim_kpx.output['kpx']

        index_3 = force[1]
        index_4 = force[3]
        self.sim_kpy.input['fy'] = index_3
        self.sim_kpy.input['mx'] = index_4
        self.sim_kpy.compute()
        kpy = self.sim_kpy.output['kpy']

        index_5 = force[0]
        index_6 = force[1]
        self.sim_kpz.input['fx'] = index_5
        self.sim_kpz.input['fy'] = index_6
        self.sim_kpz.compute()
        kpz = self.sim_kpz.output['kpz']

        index_7 = force[1]
        index_8 = force[3]
        self.sim_krx.input['fy'] = index_7
        self.sim_krx.input['mx'] = index_8
        self.sim_krx.compute()
        krx = self.sim_krx.output['krx']

        index_9 = force[0]
        index_10 = force[4]
        self.sim_kry.input['fx'] = index_9
        self.sim_kry.input['my'] = index_10
        self.sim_kry.compute()
        kry = self.sim_kry.output['kry']

        index_11 = force[5]
        index_12 = force[3]
        self.sim_krz.input['mz'] = index_11
        self.sim_krz.input['mx'] = index_12
        self.sim_krz.compute()
        krz = self.sim_krz.output['krz']

        # index_1 = (force[0] - self.low_input[0])/(self.high_input[0] - self.low_input[0]) * self.num_mesh
        # index_2 = (force[4] - self.low_input[4])/(self.high_input[4] - self.low_input[4]) * self.num_mesh
        # self.sim_kpx.input['fx'] = index_1
        # self.sim_kpx.input['my'] = index_2
        # self.sim_kpx.compute()
        # kpx = self.sim_kpx.output['kpx']
        #
        # index_3 = (force[1] - self.low_input[1]) / (self.high_input[1] - self.low_input[1]) * self.num_mesh
        # index_4 = (force[3] - self.low_input[3]) / (self.high_input[3] - self.low_input[3]) * self.num_mesh
        # self.sim_kry.input['fy'] = index_3
        # self.sim_kry.input['mx'] = index_4
        # self.sim_kry.compute()
        # kpy = self.sim_kry.output['kpy']
        #
        # index_5 = (force[0] - self.low_input[0]) / (self.high_input[0] - self.low_input[0]) * self.num_mesh
        # index_6 = (force[1] - self.low_input[1]) / (self.high_input[1] - self.low_input[1]) * self.num_mesh
        # self.sim_kpz.input['fx'] = index_5
        # self.sim_krz.input['fy'] = index_6
        # self.sim_krz.compute()
        # kpz = self.sim_krz.output['kpz']
        #
        # index_7 = (force[1] - self.low_input[1]) / (self.high_input[1] - self.low_input[1]) * self.num_mesh
        # index_8 = (force[3] - self.low_input[3]) / (self.high_input[3] - self.low_input[3]) * self.num_mesh
        # self.sim_krx.input['fy'] = index_7
        # self.sim_krx.input['mx'] = index_8
        # self.sim_krx.compute()
        # krx = self.sim_krx.output['krx']
        #
        # index_9 = (force[0] - self.low_input[0]) / (self.high_input[0] - self.low_input[0]) * self.num_mesh
        # index_10 = (force[4] - self.low_input[4]) / (self.high_input[4] - self.low_input[4]) * self.num_mesh
        # self.sim_kry.input['fy'] = index_9
        # self.sim_kry.input['mx'] = index_10
        # self.sim_kry.compute()
        # kry = self.sim_kry.output['kry']
        #
        # index_11 = (force[5] - self.low_input[5]) / (self.high_input[5] - self.low_input[5]) * self.num_mesh
        # index_12 = (force[3] - self.low_input[3]) / (self.high_input[3] - self.low_input[3]) * self.num_mesh
        # self.sim_krz.input['mz'] = index_11
        # self.sim_krz.input['mx'] = index_12
        # self.sim_krz.compute()
        # krz = self.sim_krz.output['krx']
        return [round(kpx, 5), round(kpy, 5), round(kpz, 5,), round(krx, 5), round(kry, 5), round(krz, 5)]

    def plot_rules(self):

        self.unsampled = []

        fontsize=22
        for i in range(6):
            self.unsampled.append(np.linspace(self.low_input[i], self.high_input[i], 21))

        fig = plt.figure(figsize=(20, 12), dpi=1000)
        plt.title('Fuzzy Rules', fontsize=30)
        # plt.title('Fuzzy Rules')

        plt.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=0.99, wspace=0.02, hspace=0.02)

        """kpx"""
        upsampled_x = self.unsampled[0]
        upsampled_y = self.unsampled[4]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)

        for i in range(21):
            for j in range(21):
                self.sim_kpx.input['fx'] = x[i, j]
                self.sim_kpx.input['my'] = y[i, j]
                self.sim_kpx.compute()
                z[i, j] = self.sim_kpx.output['kpx']

        ax_1 = plt.subplot(231, projection='3d')
        surf = ax_1.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        ax_1.view_init(20, 235)
        ax_1.set_zlabel("$K_p^x$", fontsize=22, labelpad=fontsize)
        ax_1.set_xlabel("    $V_{F_x}$", fontsize=22, labelpad=fontsize)
        ax_1.set_ylabel("    $V_{M_y}$", fontsize=22, labelpad=fontsize)
        ax_1.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_1.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_1.set_zticks([0.004, 0.008, 0.012, 0.016])
        plt.tick_params(labelsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Loop through the system 21*21 times to collect the control surface
        upsampled_x = self.unsampled[1]
        upsampled_y = self.unsampled[3]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)
        for i in range(21):
            for j in range(21):
                self.sim_kpy.input['fy'] = x[i, j]
                self.sim_kpy.input['mx'] = y[i, j]
                self.sim_kpy.compute()
                z[i, j] = self.sim_kpy.output['kpy']

        ax_2 = plt.subplot(232, projection='3d')
        # ax = fig.add_subplot(232, projection='3d')
        surf = ax_2.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        ax_2.view_init(20, 235)
        ax_2.set_zlabel("$K_p^y$", fontsize=22, labelpad=fontsize)
        ax_2.set_xlabel("    $V_{F_y}$", fontsize=22, labelpad=fontsize)
        ax_2.set_ylabel("    $V_{M_x}$", fontsize=22, labelpad=fontsize)
        ax_2.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_2.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_2.set_zticks([0.004, 0.008, 0.012, 0.016])
        plt.tick_params(labelsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        upsampled_x = self.unsampled[0]
        upsampled_y = self.unsampled[1]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)
        for i in range(21):
            for j in range(21):
                self.sim_kpz.input['fx'] = x[i, j]
                self.sim_kpz.input['fy'] = y[i, j]
                self.sim_kpz.compute()
                z[i, j] = self.sim_kpz.output['kpz']

        ax_3 = plt.subplot(233, projection='3d')
        # ax = fig.add_subplot(233, projection='3d')
        surf = ax_3.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        ax_3.view_init(20, 235)
        # ax.set_xticks([-40, -20, 0, 20, 40])
        z = np.zeros_like(x)
        ax_3.set_zlabel("$K_p^z$", fontsize=22, labelpad=fontsize)
        ax_3.set_xlabel("    $V_{F_x}$", fontsize=22, labelpad=fontsize)
        ax_3.set_ylabel("    $V_{F_y}$", fontsize=22, labelpad=fontsize)
        ax_3.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_3.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_3.set_zticks([0.004, 0.008, 0.012, 0.016,0.02])
        plt.tick_params(labelsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        upsampled_x = self.unsampled[1]
        upsampled_y = self.unsampled[3]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        for i in range(21):
            for j in range(21):
                self.sim_krx.input['fy'] = x[i, j]
                self.sim_krx.input['mx'] = y[i, j]
                self.sim_krx.compute()
                z[i, j] = self.sim_krx.output['krx']

        ax_4 = plt.subplot(234, projection='3d')
        # ax = fig.add_subplot(234, projection='3d')
        surf = ax_4.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        ax_4.view_init(20, 235)

        """kry"""
        ax_4.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_4.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_4.set_zlabel("$K_p^{rx}$", fontsize=22, labelpad=fontsize)
        ax_4.set_xlabel("  $V_{F_y}$", fontsize=22, labelpad=fontsize)
        ax_4.set_ylabel("    $V_{M_x}$", fontsize=22, labelpad=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        upsampled_x = self.unsampled[0]
        upsampled_y = self.unsampled[4]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        z = np.zeros_like(x)
        for i in range(21):
            for j in range(21):
                self.sim_kry.input['fx'] = x[i, j]
                self.sim_kry.input['my'] = y[i, j]
                self.sim_kry.compute()
                z[i, j] = self.sim_kry.output['kry']

        ax_5 = plt.subplot(235, projection='3d')
        # ax = fig.add_subplot(235, projection='3d')
        surf = ax_5.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        ax_5.view_init(20, 235)
        z = np.zeros_like(x)
        ax_5.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_5.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_5.set_zlabel("$K_p^{ry}$", fontsize=22, labelpad=fontsize)
        ax_5.set_xlabel("  $V_{F_x}$", fontsize=22, labelpad=fontsize)
        ax_5.set_ylabel("    $V_{M_y}$", fontsize=22, labelpad=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        upsampled_x = self.unsampled[3]
        upsampled_y = self.unsampled[5]
        x, y = np.meshgrid(upsampled_x, upsampled_y)
        for i in range(21):
            for j in range(21):
                self.sim_krz.input['mx'] = x[i, j]
                self.sim_krz.input['mz'] = y[i, j]
                self.sim_krz.compute()
                z[i, j] = self.sim_krz.output['krz']

        ax_6 = plt.subplot(236, projection='3d')
        surf = ax_6.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                               linewidth=0.4, antialiased=True)
        ax_6.view_init(20, 235)
        ax_6.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_6.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax_6.set_zlabel("$K_p^{rz}$", fontsize=22, labelpad=fontsize)
        ax_6.set_xlabel("  $V_{M_x}$", fontsize=22, labelpad=fontsize)
        ax_6.set_ylabel("    $V_{M_z}$", fontsize=22, labelpad=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.colorbar(ax_1, ax=[ax_1, ax_2, ax_3, ax_4, ax_5, ax_6])
        plt.savefig('fuzzy_rules.pdf')
        # plt.show()


    def build_fuzzy_kpx(self):
        fx_universe = np.linspace(self.low_input[0], self.high_input[0], self.num_input)
        my_universe = np.linspace(self.low_input[4], self.high_input[4], self.num_input)

        fx = ctrl.Antecedent(fx_universe, 'fx')
        my = ctrl.Antecedent(my_universe, 'my')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fx.automf(names=input_names)
        my.automf(names=input_names)

        kpx_universe = np.linspace(self.low_output[0], self.high_output[0], self.num_output)
        kpx = ctrl.Consequent(kpx_universe, 'kpx')

        output_names_3 = ['nb', 'ze', 'pb']
        kpx.automf(names=output_names_3)

        rule_kpx_0 = ctrl.Rule(antecedent=((fx['nb'] & my['ze']) |
                                           (fx['nb'] & my['ns']) |
                                           (fx['pb'] & my['ze']) |
                                           (fx['pb'] & my['ps'])),
                               consequent=kpx['pb'], label='rule kpx pb')
        rule_kpx_1 = ctrl.Rule(antecedent=((fx['ns'] & my['ze']) |
                                           (fx['ns'] & my['ns']) |
                                           (fx['ns'] & my['nb']) |
                                           (fx['nb'] & my['nb']) |
                                           (fx['pb'] & my['pb']) |
                                           (fx['ps'] & my['ps']) |
                                           (fx['ps'] & my['pb']) |
                                           (fx['ps'] & my['ze'])),
                               consequent=kpx['ze'], label='rule kpx ze')
        rule_kpx_2 = ctrl.Rule(antecedent=((fx['ze'] & my['ze']) |
                                           (fx['ze'] & my['ps']) |
                                           (fx['ze'] & my['ns']) |
                                           (fx['ze'] & my['pb']) |
                                           (fx['ze'] & my['nb']) |
                                           (fx['nb'] & my['ps']) |
                                           (fx['nb'] & my['pb']) |
                                           (fx['pb'] & my['ns']) |
                                           (fx['pb'] & my['nb']) |
                                           (fx['ns'] & my['ps']) |
                                           (fx['ns'] & my['pb']) |
                                           (fx['ps'] & my['nb']) |
                                           (fx['ps'] & my['ns'])),
                               consequent=kpx['nb'], label='rule kpx nb')
        system_kpx = ctrl.ControlSystem(rules=[rule_kpx_2, rule_kpx_1, rule_kpx_0])
        sim_kpx = ctrl.ControlSystemSimulation(system_kpx, flush_after_run=self.num_mesh * self.num_mesh + 1)

        """kpx"""
        # upsampled_x = self.unsampled[0]
        # upsampled_y = self.unsampled[4]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         sim_kpx.input['fx'] = x[i, j]
        #         sim_kpx.input['my'] = y[i, j]
        #         sim_kpx.compute()
        #         z[i, j] = sim_kpx.output['kpx']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)
        # plt.show()
        return sim_kpx

    def build_fuzzy_kpy(self):
        fy_universe = np.linspace(self.low_input[1], self.high_input[1], self.num_input)
        mx_universe = np.linspace(self.low_input[3], self.high_input[3], self.num_input)

        fy = ctrl.Antecedent(fy_universe, 'fy')
        mx = ctrl.Antecedent(mx_universe, 'mx')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fy.automf(names=input_names)
        mx.automf(names=input_names)

        kpy_universe = np.linspace(self.low_output[1], self.high_output[1], self.num_output)
        kpy = ctrl.Consequent(kpy_universe, 'kpy')

        output_names_3 = ['nb', 'ze', 'pb']
        kpy.automf(names=output_names_3)

        rule_kpy_0 = ctrl.Rule(antecedent=((fy['nb'] & mx['ns']) |
                                           (fy['nb'] & mx['ze']) |
                                           (fy['pb'] & mx['ze']) |
                                           (fy['pb'] & mx['ps'])),
                               consequent=kpy['pb'], label='rule_kpy_pb')
        rule_kpy_1 = ctrl.Rule(antecedent=((fy['ns'] & mx['ze']) |
                                           (fy['ns'] & mx['ns']) |
                                           (fy['ns'] & mx['nb']) |
                                           (fy['ps'] & mx['ps']) |
                                           (fy['ps'] & mx['pb']) |
                                           (fy['ps'] & mx['ze']) |
                                           (fy['nb'] & mx['nb']) |
                                           (fy['pb'] & mx['pb'])),
                               consequent=kpy['ze'], label='rule_kpy_ze')
        rule_kpy_2 = ctrl.Rule(antecedent=((fy['ze']) |
                                           (fy['nb'] & mx['ps']) |
                                           (fy['nb'] & mx['pb']) |
                                           (fy['pb'] & mx['ns']) |
                                           (fy['pb'] & mx['nb']) |
                                           (fy['ns'] & mx['ps']) |
                                           (fy['ns'] & mx['pb']) |
                                           (fy['ps'] & mx['nb']) |
                                           (fy['ps'] & mx['ns'])),
                               consequent=kpy['nb'], label='rule_kpy_nb')
        system_kpy = ctrl.ControlSystem(rules=[rule_kpy_0, rule_kpy_1, rule_kpy_2])
        sim_kpy = ctrl.ControlSystemSimulation(system_kpy, flush_after_run=self.num_mesh * self.num_mesh + 1)

        """kpx"""
        # upsampled_x = self.unsampled[1]
        # upsampled_y = self.unsampled[3]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         sim_kpy.input['fy'] = x[i, j]
        #         sim_kpy.input['mx'] = y[i, j]
        #         sim_kpy.compute()
        #         z[i, j] = sim_kpy.output['kpy']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)
        # plt.show()
        return sim_kpy

    def build_fuzzy_kpz(self):

        fy_universe = np.linspace(self.low_input[1], self.high_input[1], self.num_input)
        fx_universe = np.linspace(self.low_input[0], self.high_input[0], self.num_input)

        fy = ctrl.Antecedent(fy_universe, 'fy')
        fx = ctrl.Antecedent(fx_universe, 'fx')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fy.automf(names=input_names)
        fx.automf(names=input_names)

        kpz_universe = np.linspace(self.low_output[2], self.high_output[2], self.num_output)
        kpz = ctrl.Consequent(kpz_universe, 'kpz')

        output_names_3 = ['nb', 'ze', 'pb']
        kpz.automf(names=output_names_3)

        rule_kpz_0 = ctrl.Rule(antecedent=((fx['ze'] & fy['ze']) |
                                               (fx['ze'] & fy['ns']) |
                                               (fx['ns'] & fy['ze']) |
                                               (fx['ze'] & fy['ps']) |
                                               (fx['ps'] & fy['ze'])),
                                   consequent=kpz['pb'], label='rule_kpz_pb')
        rule_kpz_1 = ctrl.Rule(antecedent=((fx['ns'] & fy['ns']) |
                                           (fx['ps'] & fy['ps']) |
                                           (fx['ns'] & fy['ps']) |
                                           (fx['ps'] & fy['ns'])),
                               consequent=kpz['ze'], label='rule_kpz_ze')
        rule_kpz_2 = ctrl.Rule(antecedent=((fx['nb']) |
                                           (fx['pb']) |
                                           (fy['nb']) |
                                           (fy['pb'])),
                               consequent=kpz['nb'], label='rule_kpz_nb')
        system_kpz = ctrl.ControlSystem(rules=[rule_kpz_0, rule_kpz_1, rule_kpz_2])
        sim_kpz = ctrl.ControlSystemSimulation(system_kpz, flush_after_run=self.num_mesh * self.num_mesh + 1)

        """kpx"""
        # upsampled_x = self.unsampled[0]
        # upsampled_y = self.unsampled[1]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)

        # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         sim_kpz.input['fx'] = x[i, j]
        #         sim_kpz.input['fy'] = y[i, j]
        #         sim_kpz.compute()
        #         z[i, j] = sim_kpz.output['kpz']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)
        # plt.show()

        return sim_kpz

    def build_fuzzy_krx(self):

        fy_universe = np.linspace(self.low_input[1], self.high_input[1], self.num_input)
        mx_universe = np.linspace(self.low_input[3], self.high_input[3], self.num_input)

        fy = ctrl.Antecedent(fy_universe, 'fy')
        mx = ctrl.Antecedent(mx_universe, 'mx')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fy.automf(names=input_names)
        mx.automf(names=input_names)

        krx_universe = np.linspace(self.low_output[3], self.high_output[3], 3)
        krx = ctrl.Consequent(krx_universe, 'krx')

        output_names_2 = ['nb', 'ze', 'pb']
        krx.automf(names=output_names_2)

        rule_krx_0 = ctrl.Rule(antecedent=((mx['nb'] & fy['ze']) |
                                           (mx['nb'] & fy['ns']) |
                                           (mx['pb'] & fy['ze']) |
                                           (mx['pb'] & fy['ps'])),
                               consequent=krx['pb'], label='rule_krx_pb')
        rule_krx_1 = ctrl.Rule(antecedent=((mx['ze']) |
                                           (mx['ns']) |
                                           (mx['ps']) |
                                           (mx['nb'] & fy['nb']) |
                                           (mx['nb'] & fy['ps']) |
                                           (mx['nb'] & fy['pb']) |
                                           (mx['pb'] & fy['pb']) |
                                           (mx['pb'] & fy['ns']) |
                                           (mx['pb'] & fy['nb'])),
                               consequent=krx['nb'], label='rule_krx_ze')
        system_krx = ctrl.ControlSystem(rules=[rule_krx_0, rule_krx_1])
        sim_krx = ctrl.ControlSystemSimulation(system_krx, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # upsampled_x = self.unsampled[1]
        # upsampled_y = self.unsampled[3]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         sim_krx.input['fy'] = x[i, j]
        #         sim_krx.input['mx'] = y[i, j]
        #         sim_krx.compute()
        #         z[i, j] = sim_krx.output['krx']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)
        # plt.show()
        return sim_krx

    def build_fuzzy_kry(self):
        fx_universe = np.linspace(self.low_input[0], self.high_input[0], self.num_input)
        my_universe = np.linspace(self.low_input[4], self.high_input[4], self.num_input)

        fx = ctrl.Antecedent(fx_universe, 'fx')
        my = ctrl.Antecedent(my_universe, 'my')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']
        fx.automf(names=input_names)
        my.automf(names=input_names)

        kry_universe = np.linspace(self.low_output[4], self.high_output[4], 3)
        kry = ctrl.Consequent(kry_universe, 'kry')

        output_names_2 = ['nb', 'ze', 'pb']
        kry.automf(names=output_names_2)

        rule_kry_0 = ctrl.Rule(antecedent=((my['nb'] & fx['ze']) |
                                           (my['nb'] & fx['ns']) |
                                           (my['pb'] & fx['ze']) |
                                           (my['pb'] & fx['ps'])),
                               consequent=kry['pb'], label='rule_kry_pb')
        rule_kry_1 = ctrl.Rule(antecedent=((my['ze']) |
                                           (my['ns']) |
                                           (my['ps']) |
                                           (my['nb'] & fx['nb']) |
                                           (my['pb'] & fx['pb']) |
                                           (my['nb'] & fx['ps']) |
                                           (my['pb'] & fx['ns']) |
                                           (my['nb'] & fx['pb']) |
                                           (my['pb'] & fx['nb'])),
                               consequent=kry['nb'], label='rule_kry_nb')
        system_kry = ctrl.ControlSystem(rules=[rule_kry_0, rule_kry_1])
        sim_kry = ctrl.ControlSystemSimulation(system_kry, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # upsampled_x = self.unsampled[1]
        # upsampled_y = self.unsampled[3]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # # Loop through the system 21*21 times to collect the control surface
        # for i in range(21):
        #     for j in range(21):
        #         sim_kry.input['fx'] = x[i, j]
        #         sim_kry.input['my'] = y[i, j]
        #         sim_kry.compute()
        #         z[i, j] = sim_kry.output['kry']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)
        # plt.show()
        return sim_kry

    def build_fuzzy_krz(self):

        mx_universe = np.linspace(self.low_input[3], self.high_input[3], self.num_input)
        mz_universe = np.linspace(self.low_input[5], self.high_input[5], self.num_input)

        mx = ctrl.Antecedent(mx_universe, 'mx')
        mz = ctrl.Antecedent(mz_universe, 'mz')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']

        mx.automf(names=input_names)
        mz.automf(names=input_names)

        krz_universe = np.linspace(self.low_output[5], self.high_output[5], 3)
        krz = ctrl.Consequent(krz_universe, 'krz')
        output_names_2 = ['nb', 'ze', 'pb']
        krz.automf(names=output_names_2)

        rule_krz_0 = ctrl.Rule(antecedent=((mz['nb'] & mx['ze']) |
                                           (mz['nb'] & mx['ps']) |
                                           (mz['nb'] & mx['ns']) |
                                           (mz['pb'] & mx['ns']) |
                                           (mz['pb'] & mx['ze']) |
                                           (mz['pb'] & mx['ps'])),
                               consequent=krz['pb'], label='rule_krz_pb')
        rule_krz_1 = ctrl.Rule(antecedent=((mz['ze']) |
                                           (mz['ns']) |
                                           (mz['ps']) |
                                           (mz['nb'] & mx['nb']) |
                                           (mz['nb'] & mx['pb']) |
                                           (mz['pb'] & mx['pb']) |
                                           (mz['pb'] & mx['nb'])),
                               consequent=krz['nb'], label='rule_krz_nb')
        system_krz = ctrl.ControlSystem(rules=[rule_krz_0, rule_krz_1])
        sim_krz = ctrl.ControlSystemSimulation(system_krz, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # upsampled_x = self.unsampled[3]
        # upsampled_y = self.unsampled[5]
        # x, y = np.meshgrid(upsampled_x, upsampled_y)
        # z = np.zeros_like(x)
        #
        # for i in range(21):
        #     for j in range(21):
        #         sim_krz.input['mx'] = x[i, j]
        #         sim_krz.input['mz'] = y[i, j]
        #         sim_krz.compute()
        #         z[i, j] = sim_krz.output['krz']
        #
        # """ Plot the result in pretty 3D with alpha blending"""
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
        #                        linewidth=0.4, antialiased=True)
        # plt.show()
        return sim_krz

    def build_fuzzy_system(self):

        # Sparse universe makes calculations faster, without sacrifice accuracy.
        # Only the critical points are included here; making it higher resolution is
        # unnecessary.
        """============================================================="""
        low_force = self.low_input
        high_force = self.high_input
        num_input = self.num_input

        fx_universe = np.linspace(low_force[0], high_force[0], num_input)
        fy_universe = np.linspace(low_force[1], high_force[1], num_input)
        fz_universe = np.linspace(low_force[2], high_force[2], num_input)

        mx_universe = np.linspace(low_force[3], low_force[3], num_input)
        my_universe = np.linspace(low_force[4], low_force[4], num_input)
        mz_universe = np.linspace(low_force[5], low_force[5], num_input)

        """Create the three fuzzy variables - two inputs, one output"""
        fx = ctrl.Antecedent(fx_universe, 'fx')
        fy = ctrl.Antecedent(fy_universe, 'fy')
        fz = ctrl.Antecedent(fz_universe, 'fz')

        mx = ctrl.Antecedent(mx_universe, 'mx')
        my = ctrl.Antecedent(my_universe, 'my')
        mz = ctrl.Antecedent(mz_universe, 'mz')

        input_names = ['nb', 'ns', 'ze', 'ps', 'pb']

        fx.automf(names=input_names)
        fy.automf(names=input_names)
        fz.automf(names=input_names)
        mx.automf(names=input_names)
        my.automf(names=input_names)
        mz.automf(names=input_names)

        """============================================================="""
        """Create the outputs"""
        kpx_universe = np.linspace(self.low_output[0], self.high_output[0], self.num_output)
        kpy_universe = np.linspace(self.low_output[1], self.high_output[1], self.num_output)
        kpz_universe = np.linspace(self.low_output[2], self.high_output[2], self.num_output)

        krx_universe = np.linspace(self.low_output[3], self.high_output[3], 3)
        kry_universe = np.linspace(self.low_output[4], self.high_output[4], 3)
        krz_universe = np.linspace(self.low_output[5], self.high_output[5], 3)

        kpx = ctrl.Consequent(kpx_universe, 'kpx')
        kpy = ctrl.Consequent(kpy_universe, 'kpy')
        kpz = ctrl.Consequent(kpz_universe, 'kpz')

        krx = ctrl.Consequent(krx_universe, 'krx')
        kry = ctrl.Consequent(kry_universe, 'kry')
        krz = ctrl.Consequent(krz_universe, 'krz')

        output_names_3 = ['nb', 'ze', 'pb']

        # Here we use the convenience `automf` to populate the fuzzy variables with
        # terms. The optional kwarg `names=` lets us specify the names of our Terms.

        kpx.automf(names=output_names_3)
        kpy.automf(names=output_names_3)
        kpz.automf(names=output_names_3)

        krx.automf(names=output_names_3)
        kry.automf(names=output_names_3)
        krz.automf(names=output_names_3)

        # define the rules for the desired force fx and my
        # ===============================================================
        rule_kpx_0 = ctrl.Rule(antecedent=((fx['nb'] & my['ze']) |
                                           (fx['nb'] & my['ns']) |
                                           (fx['pb'] & my['ze']) |
                                           (fx['pb'] & my['ps'])),
                               consequent=kpx['pb'], label='rule kpx pb')
        rule_kpx_1 = ctrl.Rule(antecedent=((fx['ns'] & my['ze']) |
                                           (fx['ns'] & my['ns']) |
                                           (fx['ns'] & my['nb']) |
                                           (fx['nb'] & my['nb']) |
                                           (fx['pb'] & my['pb']) |
                                           (fx['ps'] & my['ps']) |
                                           (fx['ps'] & my['pb']) |
                                           (fx['ps'] & my['ze'])),
                               consequent=kpx['ze'], label='rule kpx ze')
        rule_kpx_2 = ctrl.Rule(antecedent=((fx['ze'] & my['ze']) |
                                           (fx['ze'] & my['ps']) |
                                           (fx['ze'] & my['ns']) |
                                           (fx['ze'] & my['pb']) |
                                           (fx['ze'] & my['nb']) |
                                           (fx['nb'] & my['ps']) |
                                           (fx['nb'] & my['pb']) |
                                           (fx['pb'] & my['ns']) |
                                           (fx['pb'] & my['nb']) |
                                           (fx['ns'] & my['ps']) |
                                           (fx['ns'] & my['pb']) |
                                           (fx['ps'] & my['nb']) |
                                           (fx['ps'] & my['ns'])),
                               consequent=kpx['nb'], label='rule kpx nb')
        system_kpx = ctrl.ControlSystem(rules=[rule_kpx_2, rule_kpx_1, rule_kpx_0])
        sim_kpx = ctrl.ControlSystemSimulation(system_kpx, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # define the rules for the desired force fy and mz
        # ===============================================================
        rule_kpy_0 = ctrl.Rule(antecedent=((fy['nb'] & mx['ns']) |
                                           (fy['nb'] & mx['ze']) |
                                           (fy['pb'] & mx['ze']) |
                                           (fy['pb'] & mx['ps'])),
                               consequent=kpy['pb'], label='rule_kpy_pb')
        rule_kpy_1 = ctrl.Rule(antecedent=((fy['ns'] & mx['ze']) |
                                           (fy['ns'] & mx['ns']) |
                                           (fy['ns'] & mx['nb']) |
                                           (fy['ps'] & mx['ps']) |
                                           (fy['ps'] & mx['pb']) |
                                           (fy['ps'] & mx['ze']) |
                                           (fy['nb'] & mx['nb']) |
                                           (fy['pb'] & mx['pb'])),
                               consequent=kpy['ze'], label='rule_kpy_ze')
        rule_kpy_2 = ctrl.Rule(antecedent=((fy['ze']) |
                                           (fy['nb'] & mx['ps']) |
                                           (fy['nb'] & mx['pb']) |
                                           (fy['pb'] & mx['ns']) |
                                           (fy['pb'] & mx['nb']) |
                                           (fy['ns'] & mx['ps']) |
                                           (fy['ns'] & mx['pb']) |
                                           (fy['ps'] & mx['nb']) |
                                           (fy['ps'] & mx['ns'])),
                               consequent=kpy['nb'], label='rule_kpy_nb')
        system_kpy = ctrl.ControlSystem(rules=[rule_kpy_0, rule_kpy_1, rule_kpy_2])
        sim_kpy = ctrl.ControlSystemSimulation(system_kpy, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # ===============================================================
        rule_kpz_0 = ctrl.Rule(antecedent=((fx['ze'] & fy['ze']) |
                                           (fx['ze'] & fy['ns']) |
                                           (fx['ns'] & fy['ze']) |
                                           (fx['ze'] & fy['ps']) |
                                           (fx['ps'] & fy['ze'])),
                               consequent=kpz['pb'], label='rule_kpz_pb')
        rule_kpz_1 = ctrl.Rule(antecedent=((fx['ns'] & fy['ns']) |
                                           (fx['ps'] & fy['ps']) |
                                           (fx['ns'] & fy['ps']) |
                                           (fx['ps'] & fy['ns'])),
                               consequent=kpz['ze'], label='rule_kpz_ze')
        rule_kpz_2 = ctrl.Rule(antecedent=((fx['nb']) |
                                           (fx['pb']) |
                                           (fy['nb']) |
                                           (fy['pb'])),
                               consequent=kpz['nb'], label='rule_kpz_nb')
        system_kpz = ctrl.ControlSystem(rules=[rule_kpz_0, rule_kpz_1, rule_kpz_2])
        sim_kpz = ctrl.ControlSystemSimulation(system_kpz, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # ===============================================================
        rule_krx_0 = ctrl.Rule(antecedent=((mx['nb'] & fy['ze']) |
                                           (mx['nb'] & fy['ns']) |
                                           (mx['pb'] & fy['ze']) |
                                           (mx['pb'] & fy['ps'])),
                               consequent=krx['pb'], label='rule_krx_pb')
        rule_krx_1 = ctrl.Rule(antecedent=((mx['ze']) |
                                           (mx['ns']) |
                                           (mx['ps']) |
                                           (mx['nb'] & fy['nb']) |
                                           (mx['nb'] & fy['ps']) |
                                           (mx['nb'] & fy['pb']) |
                                           (mx['pb'] & fy['pb']) |
                                           (mx['pb'] & fy['ns']) |
                                           (mx['pb'] & fy['nb'])),
                               consequent=krx['nb'], label='rule_krx_ze')
        system_krx = ctrl.ControlSystem(rules=[rule_krx_0, rule_krx_1])
        sim_krx = ctrl.ControlSystemSimulation(system_krx, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # ===============================================================
        rule_kry_0 = ctrl.Rule(antecedent=((my['nb'] & fx['ze']) |
                                           (my['nb'] & fx['ns']) |
                                           (my['pb'] & fx['ze']) |
                                           (my['pb'] & fx['ps'])),
                               consequent=kry['pb'], label='rule_kry_pb')
        rule_kry_1 = ctrl.Rule(antecedent=((my['ze']) |
                                           (my['ns']) |
                                           (my['ps']) |
                                           (my['nb'] & fx['nb']) |
                                           (my['pb'] & fx['pb']) |
                                           (my['nb'] & fx['ps']) |
                                           (my['pb'] & fx['ns']) |
                                           (my['nb'] & fx['pb']) |
                                           (my['pb'] & fx['nb'])),
                               consequent=kry['nb'], label='rule_kry_nb')
        system_kry = ctrl.ControlSystem(rules=[rule_kry_0, rule_kry_1])
        sim_kry = ctrl.ControlSystemSimulation(system_kry, flush_after_run=self.num_mesh * self.num_mesh + 1)

        # ===============================================================
        rule_krz_0 = ctrl.Rule(antecedent=((mz['nb'] & mx['ze']) |
                                           (mz['nb'] & mx['ps']) |
                                           (mz['nb'] & mx['ps']) |
                                           (mz['pb'] & mx['ns']) |
                                           (mz['pb'] & mx['ze']) |
                                           (mz['pb'] & mx['ps'])),
                               consequent=krz['pb'], label='rule_krz_pb')
        rule_krz_1 = ctrl.Rule(antecedent=((mz['ze']) |
                                           (mz['ns']) |
                                           (mz['ps']) |
                                           (mz['nb'] & mx['nb']) |
                                           (mz['pb'] & mx['pb']) |
                                           (mz['nb'] & mx['pb']) |
                                           (mz['pb'] & mx['nb'])),
                               consequent=krz['nb'], label='rule_krz_nb')
        system_krz = ctrl.ControlSystem(rules=[rule_krz_0, rule_krz_1])
        sim_krz = ctrl.ControlSystemSimulation(system_krz, flush_after_run=self.num_mesh * self.num_mesh + 1)

        return sim_kpx, sim_kpy, sim_kpz, sim_krx, sim_kry, sim_krz


if __name__ == "__main__":

    fuzzy_system = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]), high_output=np.array([0.02, 0.02, 0.025, 0.015, 0.015, 0.015]))
    fuzzy_system.plot_rules()

    # kp = fuzzy_system.get_output(np.array([-20, -20, -30, 0, 0.9, 0.7]))[:3]
    # fuzzy_system.build_fuzzy_kpx()