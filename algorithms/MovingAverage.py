import numpy as np


class MA:
    def __init__(self, period):
        self.counter = 0
        self.period = period
        self.holder = [[], [], [], [], [], []]

    def cal(self, ft):
        self.counter += 1
        self.holder[0].append(ft[0])
        self.holder[1].append(ft[1])
        self.holder[2].append(ft[2])
        self.holder[3].append(ft[3])
        self.holder[4].append(ft[4])
        self.holder[5].append(ft[5])
        average = [0, 0, 0, 0, 0, 0]
        if self.counter < self.period:
            return ft
        else:
            for i in range(6):
                for j in range(self.period):
                    average[i] += self.holder[i][j]
                average[i] = average[i] / self.period
                self.holder[i].pop(0)
            return average

    def clear(self):
        self.counter = 0
        self.holder = [[], [], [], [], [], []]
