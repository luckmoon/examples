#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/1 上午2:15
# @Author  : wulala
# @Project : examples
# @File    : t_tensor.py
# @Software: PyCharm

import torch

a = torch.Tensor(4)
print(a)  # tensor([ 0.0000, -0.0000, -0.0000, -0.0000])

b = torch.Tensor([1])
print(b)  # tensor([1.])

c = torch.Tensor([6, 4])
print(c)  # tensor([6., 4.])
print(c.size(0))  # 2

print(10//3)  # 3

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x.shape)  # [3,3]
x = x.narrow(1, 0, 1)
print(x.shape)  # [2,3]
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
print(x)

