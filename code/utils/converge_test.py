import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2

class Figure_2D():
    def __init__(self, width=10, height=7.5, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        plt.rcParams.update({'font.size': 12, 'font.serif': 'Times New Roman'})
        self.axes = self.fig.add_subplot(111)

    def update_plot(self, y_vec, legend_vec=None, block = False):
        if legend_vec is None:
            legend_vec = ['True Q', 'TD3 y', 'TD3 Q_a', 'TD3 Q_b',
                          'ATD3 y', 'ATD3 Q_a', 'ATD3 Q_b', 'ATD3 Q_m']
        self.axes.clear()
        self.line_list = self.axes.plot(y_vec)
        self.axes.legend(legend_vec, loc='lower center', ncol=4, bbox_to_anchor=(0.50, 1.0), frameon=False)
        self.axes.figure.canvas.draw_idle()
        plt.show(block=block)
        plt.pause(0.0001)
        plt.ylabel('Q-value')
        plt.xlabel('Time steps')

class Figure_3D():
    def __init__(self, width=15, height=7.5, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        plt.rcParams.update({'font.size': 12, 'font.serif': 'Times New Roman'})
        self.axes = []
        for i in range(6):
            self.axes.append(self.fig.add_subplot(2, 3, i+1))
        # self.ax = self.fig.add_subplot(111, projection='3d')

    def init_plot(self, z_mat, legend_vec=None, block = False):
        if legend_vec is None:
            legend_vec = ['True Q', 'TD3 y', 'TD3 Q_a', 'TD3 Q_b',
                          'ATD3 y', 'ATD3 Q_a', 'ATD3 Q_b', 'ATD3 Q_m']
        self.fig.tight_layout()
        for i in range(z_mat.shape[1]):
            self.axes[i].clear()
            self.axes[i].plot(z_mat[:,i,:])
            self.axes[i].set_ylabel('Q-value: {}'.format(i))
            self.axes[i].set_xlabel('Time steps')
        self.fig.legend(legend_vec, loc='lower center', ncol=8, bbox_to_anchor=(0.50, 0.96), frameon=False)
        plt.show(block=block)
        plt.pause(0.0001)

    def update_plot(self, z_mat, legend_vec=None, block = False):
        self.fig.tight_layout()
        for i in range(z_mat.shape[1]):
            self.axes[i].clear()
            self.axes[i].plot(z_mat[:,i,:])
            self.axes[i].set_ylabel('Q-value: {}'.format(i))
            self.axes[i].set_xlabel('Time steps')
            self.axes[i].set_xticks([0, 0, z_mat.shape[0]])
            self.axes[i].set_yticks([0, np.min(z_mat[:,i,:]), np.max(z_mat[:,i,:])])
        plt.show(block=block)
        plt.pause(0.0001)

def fig_to_img(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

save_video = False
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('Q.mp4', fourcc, 60.0, (1000, 750))

gamma = 0.99
tau = 0.5
t_freq = 2
alpha = 0.05
beta = 0.3
step_num = 100
Q_a_ATD3 = np.zeros(step_num)
Q_b_ATD3 = np.zeros(step_num)
for i in range(step_num):
    Q_a_ATD3[i] = np.random.uniform(0, 2 * (step_num), 1)
    Q_b_ATD3[i] = np.random.uniform(0, 2 * (step_num), 1)
Q_a_TD3 = np.copy(Q_a_ATD3)
Q_t_a_ATD3 = np.copy(Q_a_ATD3)
Q_t_a_TD3 = np.copy(Q_a_ATD3)
Q_b_TD3 = np.copy(Q_b_ATD3)
Q_t_b_ATD3 = np.copy(Q_b_ATD3)
Q_t_b_TD3 = np.copy(Q_b_ATD3)

Q_m_ATD3 = 0.5 * (Q_a_ATD3 + Q_b_ATD3)

Q_true = np.zeros(step_num)
for i in range(step_num):
    Q_true_temp = 0.0
    for t in range(i, step_num):
        Q_true_temp += gamma**(t-i)
    Q_true[i] = Q_true_temp

fig_2d = Figure_2D()
y_vec = np.asarray([Q_true[0], 1.0 + min(Q_a_TD3[1], Q_b_TD3[1]), Q_a_TD3[0], Q_b_TD3[0],
                    1.0 + min(Q_a_ATD3[1], Q_b_ATD3[1]), Q_a_ATD3[0],
                    Q_b_ATD3[0], Q_m_ATD3[0]]).reshape((1, -1))
fig_2d.update_plot(y_vec=y_vec)


# fig_3d = Figure_3D()
# z_mat = np.expand_dims(np.c_[Q_true, 1.0 + np.min(np.c_[Q_a_TD3, Q_b_TD3], axis=-1), Q_a_TD3, Q_b_TD3,
#                     1.0 + np.min(np.c_[Q_a_ATD3, Q_b_ATD3], axis=-1), Q_a_ATD3,
#                     Q_b_ATD3, Q_m_ATD3], axis=0)
# fig_3d.init_plot(z_mat)
for r in range(1000):
    y_TD3 = np.ones(step_num)
    y_TD3[:(step_num-1)] = np.ones(step_num-1) + gamma * np.min(np.c_[Q_t_a_TD3[1:], Q_t_b_TD3[1:]], axis=-1)
    y_ATD3 = np.ones(step_num)
    y_ATD3[:(step_num-1)] = np.ones(step_num-1) + gamma * np.min(np.c_[Q_t_a_ATD3[1:], Q_t_b_ATD3[1:]], axis=-1)

    Q_a_TD3 = Q_a_TD3 + alpha * step_num**0.5 * (y_TD3 - Q_a_TD3)
    Q_b_TD3 = Q_b_TD3 + alpha * step_num**0.5 * (y_TD3 - Q_b_TD3)

    d_ab = Q_a_ATD3 - Q_b_ATD3
    Q_a_ATD3 = Q_a_ATD3 + alpha * np.arange(1, step_num+1)**0.5 * (y_ATD3 - Q_a_ATD3 + beta * d_ab)
    Q_b_ATD3 = Q_b_ATD3 + alpha * np.arange(1, step_num+1)**0.5 * (y_ATD3 - Q_b_ATD3 - beta * d_ab)
    Q_m_ATD3 = 0.5 * (Q_a_ATD3 + Q_b_ATD3)
    if 0 == r % t_freq:
        Q_t_a_TD3 = (1 - tau) * Q_t_a_TD3 + tau * Q_a_TD3
        Q_t_b_TD3 = (1 - tau) * Q_t_b_TD3 + tau * Q_b_TD3

        Q_t_a_ATD3 = (1 - tau) * Q_t_a_ATD3 + tau * Q_a_ATD3
        Q_t_b_ATD3 = (1 - tau) * Q_t_b_ATD3 + tau * Q_b_ATD3

    print('e_a_TD3: {}, e_m_ATD3: {}'.format(Q_a_TD3[0] - Q_true[0], Q_m_ATD3[0] - Q_true[0]))
    # print('Q_a_TD3: {}, Q_m_ATD3: {}'.format(Q_a_TD3, Q_m_ATD3))
    y_vec = np.r_[y_vec,
                  np.asarray([Q_true[0], y_TD3[0], Q_a_TD3[0], Q_b_TD3[0],
                              y_ATD3[0], Q_a_ATD3[0], Q_b_ATD3[0], Q_m_ATD3[0]]).reshape((1, -1))]
    # z_mat = np.r_[z_mat,np.expand_dims(np.c_[Q_true, y_TD3, Q_a_TD3, Q_b_TD3,
    #                 y_ATD3, Q_a_ATD3,
    #                 Q_b_ATD3, Q_m_ATD3], axis=0)]

    # fig_3d.update_plot(z_mat)
    fig_2d.update_plot(y_vec)
    if save_video:
        img = fig_to_img(fig_2d.fig)
        out_video.write(img)

if save_video:
    out_video.release()

fig_2d.update_plot(y_vec, block = True)


