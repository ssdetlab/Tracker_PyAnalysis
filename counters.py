#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

import config
from config import *


COUNTERS      = ["Pixels/chip", "Clusters/chip", "Seeds",       "Good Tracks", "Selected Tracks"]
counters_cols = [ROOT.kBlack,   ROOT.kBlue,      ROOT.kGreen+2, ROOT.kRed,      ROOT.kOrange ]

counters_x_trg = array.array('d')
counters_y_val = {}

def init_global_counters():
    for counter in COUNTERS: counters_y_val.update({counter:array.array('d')})

def append_global_counters():
    for counter in COUNTERS:
        counters_y_val[counter].append(0)

def set_global_counter(counter,idx,val):
    counters_y_val[counter][idx] = val

# def fill_global_counters(trg,event_counters):
#     counters_x_trg.apend(trg)
#     for counter,value in event_counters.items(): counters_y_val[counter].append(value)

# def init_event_counters():
#     event_counters = {}
#     for counter in COUNTERS: event_counters.update({counter:-1})
#     return event_counters
