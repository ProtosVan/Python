# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, math
from PIL import Image
import numpy as np

def listtranspose(alist):
    temp = np.array(np.matrix(alist).transpose())
    result = []
    for i in temp:
        result.append(list(i))
    return result