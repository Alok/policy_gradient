#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.initializations import glorot_uniform, zero
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


