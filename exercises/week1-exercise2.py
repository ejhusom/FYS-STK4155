#!/usr/bin/env python3
# ==================================================================
# File:     week1-exercise2.py
# Author:   Erik Johannes Husom
# Created:  2019-08-27
# ------------------------------------------------------------------
# Description:
# Exercise 2 of week 1 in FYS-STK4155.
# ==================================================================
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100, 1)
y = 5*x*x + 0.1*np.random.randn(100, 1)

design_matrix = np.

plt.figure()
plt.plot(x, y, '.')
plt.show()
