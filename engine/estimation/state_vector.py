import numpy as np

class StateVector:
    def __init__(self, num_amb):
        self.num_amb = num_amb
        self.dim = 5 + num_amb
        self.x = np.zeros((self.dim, 1))
        self.P = np.eye(self.dim) * 100.0
