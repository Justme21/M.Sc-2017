#!/usr/bin/python

class DrivingLearner():
    def __init__(self,look_back):
        self.state = ()
        self.prev_states = []

        self.look_back = look_back
        self.e = .05
        self.gam = .99
        self.alph = 1

        self.reward = 0

        self.learnable = False

        self.act_index = 0

        self.q_values = {}



    def initialise(self):
        self.reward = 0
        self.prev_states = []

    def getMaxAction(self):
        action = None
        top_q = None
        act = None

        if self.state
