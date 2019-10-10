#!/usr/bin/env python3
# ============================================================================
# File:     analysis.py
# Author:   Erik Johannes Husom
# Created:  2019-10-10
# ----------------------------------------------------------------------------
# Description:
# 
# ============================================================================
import sys
import pandas as pd

# Add machine learning methods to path
sys.path.append('../../methods')

from Classification import *

def read_data(filename):
    """
    Read data from file. File must be placed in certain directory:
    
    project/
    | src/
    | | analysis.py
    | data/
    | | file
    """
    
    if filename.endswith(('.xls', '.xlsx'):
        df = pd.read_excel('../data/' + filename)
    else:
        print('File must be in Excel-format!')
        sys.exit(1)
        


