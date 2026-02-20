import numpy as np 

class BarycentricInterpolation:
    """
    Implementation of barycentric interpolation of Lagrange polynomials, when they are used as a basis for a direct collocation 
    approximation of the solution to an ODE. Uses Chebyshev collocation nodes of the second kind.
    """
    def __init__(self, start_time, end_time, num_nodes):
        self.start_time = start_time
        self.end_time = end_time
        self.num_nodes = num_nodes
        self.collocation_grid = self.generate_chebyshev_nodes_second_kind()

    #-----------------DIRECT COLLOCATION METHODS-----------------#
    
    def generate_chebyshev_nodes_second_kind(self):
        '''Returns a sorted array of self.num_nodes Chebyshev nodes of the second kind in the interval [self.start_time, self.end_time].'''
        i = np.arange(0, self.num_nodes, step=1, dtype=np.float32) # n equispaced nodes 0 -> n-1
        nodes = np.cos(i * np.pi / (self.num_nodes - 1)) # descending order 1 -> -1 as i ascends 0 -> n-1 
        nodes = 0.5 * (self.end_time - self.start_time) * nodes + 0.5 * (self.start_time + self.end_time) # scale nodes from [-1, 1] to [st, et]
        
        return np.sort(nodes, axis=0) # sort in ascending order a -> b 
    
    def compute_chebyshev_second_kind_barycentric_weights(self):
        '''
        Returns array of barycentric weights for the self.num_nodes-sized Chebyshev (2nd kind) 
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
        Returns the Jacobian of the approximated ODE solution as ndarray. 
        Formula taken from original paper https://arxiv.org/pdf/2502.15642. 
        '''
        D = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        ws = self.compute_chebyshev_second_kind_barycentric_weights()
        # Compute off-diagonals
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    D[i, j] = (ws[j] / ws[i]) * (1 / (self.collocation_grid[i] - self.collocation_grid[j]))
        # Compute diagonals as negative sum of off-diagonals of same row (requires all off-diagonals to be filled first)
        for i in range(self.num_nodes):
            D[i, i] = -np.sum(D[i, :])

        return D
    
    #-----------------INTEGRATED RESIDUAL METHODS-----------------#
    
    def compute_clenshaw_curtis_quadrature_weights(self):
        ''''
        Returns array of self.num_nodes Clenshaw-Curtis quadrature weights. 
        Formula taken from https://personal.math.vt.edu/embree/math5466/lecture23.pdf.
        Algorithm to calculate the weights using FFT and an aprroximation of the Lagrange interpolating polynomials 
        as trigonometric interpolation is taken from https://people.maths.ox.ac.uk/trefethen/publication/PDF/2008_127.pdf
        '''
        num_intervals = self.num_nodes - 1 
        if num_intervals == 0:
            return np.array([self.end_time - self.start_time])

        k = np.arange(0, num_intervals + 1)

        # Fourier coefficients: v_k = 2 / (1 - k^2) for even k, 0 for odd k
        v = np.zeros(num_intervals + 1)
        v[0::2] = 2 / (1 - k[0::2]**2)

        # Construct symmetric vector for IFFT (length 2N)
        V = np.concatenate([v, v[num_intervals-1:0:-1]])

        # Weights via IFFT (take first N+1 entries)
        w = np.real(np.fft.ifft(V))[:num_intervals + 1]

        # Interior weights need factor of 2 (c_j = 2 for j=1,...,N-1; c_0 = c_N = 1)
        w[1:num_intervals] *= 2

        # Scale from [-1,1] to [start_time, end_time]
        w *= (self.end_time - self.start_time) / 2.0

        return w
