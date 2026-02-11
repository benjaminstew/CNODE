"""
Implementation of barycentric interpolation of a function approximated using 
a Lagrange polynomial basis constrained at Chebyshev collocation nodes. 
"""
import numpy as np 

class BarycentricInterpolation:

    def __init__(self, start_time, end_time, num_nodes):
        self.start_time = start_time
        self.end_time = end_time
        self.num_nodes = num_nodes
        self.collocation_grid = self.generate_chebyshev_nodes_second_kind()
    
    def generate_chebyshev_nodes_second_kind(self):
        '''
        Generates self.num_nodes Chebyshev nodes of the second kind in the interval [self.start_time, self.end_time].
        '''
        i = np.arange(0, self.num_nodes, step=1, dtype=np.float32) # n equispaced nodes 0 -> n-1
        nodes = np.cos(i * np.pi / (self.num_nodes - 1)) # descending order 1 -> -1 as i ascends 0 -> n-1 
        nodes = 0.5 * (self.end_time - self.start_time) * nodes + 0.5 * (self.start_time + self.end_time) # scale nodes from [-1, 1] to [st, et]
        
        return np.sort(nodes, axis=0) # sort in ascending order a -> b 
    
    def compute_chebyshev_second_kind_barycentric_weights(self):
        '''
        Returns array of self.num_nodes barycentric weights for the self.num_nodes-sized Chebyshev (2nd kind) 
        collocation grid using the closed-form expression in https://people.maths.ox.ac.uk/trefethen/barycentric.pdf.
        '''
        ws = np.ones(self.num_nodes, dtype=np.float32)
        for i in range(self.num_nodes):
            if i % 2 != 0:
                ws[i] *= -1
            if i == 0 or i == (self.num_nodes-1):
                ws[i] *= 0.5

        return ws

    def compute_derivative_matrix(self):
        '''
        Compute the derivative matrix for the approximated function. 
        Formula taken from original paper https://arxiv.org/pdf/2502.15642. 
        '''
        D = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        ws = self.compute_chebyshev_second_kind_barycentric_weights()
        # Compute off-diagonals
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    D[i, j] = (ws[j] / ws[i]) * (1 / (self.collocation_grid[i] - self.collocation_grid[j]))
        # Compute diagonals as negative sum of off-diagonals of same row (requires all off-diagonals to be first filled)
        for i in range(self.num_nodes):
            D[i, i] = -np.sum(D[i, :])

        return D
    