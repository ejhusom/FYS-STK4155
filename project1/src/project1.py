#!/usr/bin/env python3
# ============================================================================
# File:     project1.py
# Author:   Erik Johannes Husom
# Created:  2019-09-11
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
from RegressionResampling import *

def ex_a(model):

    beta, z_tilde = model.regression(model.X, model.z, lmd=0)
    model.print_error_analysis(model.z, z_tilde)


def ex_b(model):

    model.cross_validation()


if __name__ == '__main__': 
    project1 = RegressionResampling()

    ex_a(project1)
    ex_b(project1)

