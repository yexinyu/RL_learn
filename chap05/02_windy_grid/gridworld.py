# -*- coding: UTF-8 -*-

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Grid(object):
    def __init__(self, x: int = None,
                 y: int = None,
                 type: int = 0,
                 reward: int = 0.0,
                 value: float = 0.0):  # value 属性备用
        self.x = x  # 坐标x
        self.y = y
        self.type = value  # 类别值(0:空；1:障碍或边界)
        self.reward = reward  # 该格子的即时奖励
        self.value = value  # 该格子的价值，暂没用上
        self.name = None  # 该格子的名称
        self._update_name()

    def _update_name(self):
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "name:{4}, x:{0}, y:{1}, type:{2}, value{3}".format(self.x,
                                                                   self.y,
                                                                   self.type,
                                                                   self.reward,
                                                                   self.value,
                                                                   self.name
                                                                   )


class GridMatrix(object):
    """格子矩阵，通过不同的设置，模拟不同的格子世界环境
    """

    def __init__(self, n_width: int,  # 水平方向格子数
                 n_height: int,  # 竖直方向格子数
                 default_type: int = 0,  # 默认类型
                 default_reward: float = 0.0,  # 默认即时奖励值
                 default_value: float = 0.0  # 默认价值（这个有点多余）
                 ):
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,
                                       y,
                                       self.default_type,
                                       self.default_reward,
                                       self.default_value))

    def get_grid(self, x, y=None):
        """获取一个格子信息
        args: 坐标信息，由x,y表示或仅有一个类型为tuple的x表示
        return: grid object
        """
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert (xx >= 0 and yy >= 0 and xx < self.n_width and yy < self.n_height), \
            "任意坐标值应在合理区间"
        index = yy * self.n_width + xx
        return self.grids[index]
