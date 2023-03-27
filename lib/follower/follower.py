### A score follower following a HMM representation of the state space 
from typing import * 
import numpy as np 


class ScoreFollowingHMM: 
    """A class to model the musical state HMM.
    
    """
    # Problem 1
    def __init__(self, chord_sequence, num_statistics, threshold = 0.001):
        """Initialize a HMM to model the evolution of polyphonic 
        score-audio states.

        Parameters:
            chord_sequence ((N,) list): A list of hidden/chord states from time t=0 to T = N-1
                (in the form (onset, offset, [list of MIDI note values]))
            num_statistics (int): The number of statistics to use (D in the paper)
            threshold (float): The threshold below which to assume probabilities
                are zero.
        """
        self.chord_sequence = chord_sequence    # The chord sequence to parse 
        self.D = num_statistics
        self.A = self._construct_state_transition(chord_sequence, 0.9) 
        
    def _construct_state_transition(self, chord_sequence, p):
        """Construct a state transition matrix based off of the 
        negative binomial assumption of remaining in a state for
        extended periods of time.
        
        Parameters:
            chord_sequence ((N,) list): A list of hidden/chord states from time t=0 to T = N-1
                (in the form (onset, offset, [list of MIDI note values]))
            p (float): Probability of self loop 
        """
        # Get all unique chords 
        unique_chords = set([tuple(chord) for _,_,chord in chord_sequence])    # All unique chords 
        
        # Get which chords follow which other chords 
        sequences = {('START',):[chord_sequence[0][-1]]}
        prev_chord = (('START',))
        for _,_,cur_chord in chord_sequence:
            cur_chord = tuple(cur_chord)
            if cur_chord not in sequences:
                sequences[cur_chord] = []
        
            sequences[prev_chord].append(cur_chord)
            prev_chord = cur_chord

        if prev_chord not in sequences:
            sequences[prev_chord] = []
        sequences[prev_chord].append(('STOP',))

        # Get indexes of these things 
        chord_indexes = {chord:i for i,chord in enumerate(sequences.keys())}

        # Build the coefficient matrix 
        self.N = len(chord_indexes)  # Append on 2 states afor silence
        A = np.zeros((N,N))

        # Populate with proper probabilities
        for chord in chord_indexes:
            A[chord_indexes[chord], chord_indexes[chord]] = p 
            for following_chords in sequences[chord]:
                for new_chord in following_chords:
                    A[chord_indexes[new_chord], chord_indexes[chord]] = 1-p 
        
        # Normalize the matrx to be column stochastic. 
        return A / A.sum(axis=0)
        
    # Problem 2
    def _forward(self, y):
        """
        Compute the scaled forward probability matrix and scaling factors.
        
        Parameters
        ----------
        y : ndarray of shape (T,m) 
            The sequence of observations

        s_r_probs: ndarray of shape (T,m):
        
        Returns
        -------
        alpha : ndarray of shape (T, n)
            The scaled forward probability matrix
        """
        # Initialize algorithm
        T, m = y.shape 
        self.N = self.A.shape[0]
        alpha_init = np.ones(self.N)    # alpha_{-1}(.) = 1 (concentraetd unit masses on initial state of model)
        
        alpha_vec = np.empty((T, self.N)) 
        alpha_vec[0] = self.A @ alpha_init * ???


        
        

    # Problem 3
    def _backward(self, z, c):
        """
        Compute the scaled backward probability matrix.
        
        Parameters
        ----------
        z : ndarray of shape (T,) 
            The sequence of observations
        c : ndarray of shape (T,)
            The scaling factors from the forward pass

        Returns
        -------
        beta : ndarray of shape (T, n)
            The scaled backward probability matrix
        """
        T = len(z)          # The number of timesteps/observations so far 
        n = len(self.pi)    # The number of (hidden) states

        beta = np.empty((T,n))
        beta[T-1] = c[T-1] * np.ones(n)   # Set the last step to just be the coefficients 

        for t in range(2,T+1):
            h = beta[T-t+1] * self.B[z[T-t+1]] # Used for inner part of sum (actually a Hadamard product)
            beta[T-t] = self.A.T @ h           # Used to actually calculate sum 
                                               # (sum is the def of matrix-vector mult, but with A transposed)
            beta[T-t] *= c[T-t]                # Rescale
        
        # Return the scaled backward probability matrix
        return beta

    
    # Problem 4
    def _xi(self, z, alpha, beta, c):
        """
        Compute the xi and gamma probabilities.

        Parameters
        ----------
        z : ndarray of shape (T,)
            The observation sequence
        alpha : ndarray of shape (T, n)
            The scaled forward probability matrix from the forward pass
        beta : ndarray of shape (T, n)
            The scaled backward probability matrix from the backward pass
        c : ndarray of shape (T,)
            The scaling factors from the forward pass

        Returns
        -------
        xi : ndarray of shape (T-1, n, n)
            The xi probability array
        gamma : ndarray of shape (T, n)
            The gamma probability array
        """
        T = len(z)          # The number of timesteps/observations so far 
        n = len(self.pi)    # The number of (hidden) states

        xi = np.empty((T-1,n,n))
        for t in range(T-1):
            xi[t] = alpha[t].reshape((n,1)) * self.A.T * self.B[z[t+1]] * beta[t+1] 

        gamma = np.empty((T,n))
        for t in range(T):
            gamma[t] = alpha[t]*beta[t]/c[t]
        
        return xi, gamma
            
    # Problem 5
    def _estimate(self, y, xi, gamma):
        """
        Estimate better parameter values and update self.pi, self.A, and
        self.B in place.

        Parameters
        ----------
        y : ndarray of shape (T,)
            The observation sequence
        xi : ndarray of shape (T-1, n, n)
            The xi probability array
        gamma : ndarray of shape (T, n)
            The gamma probability array
        """
        # Update the pi (state distribution) vector 
        self.pi = gamma[0]

        # Update the A (state transition) matrix 
        xi_t = np.zeros_like(xi)        # Weird transposition step
        for i in range(len(xi)):
            xi_t[i] = xi[i].T
        self.A = np.sum(xi_t,axis=0)/np.sum(gamma[:-1],axis=0)

        # Update the B (observation probability) matrix
        for i in range(len(self.B)):
            mask = (z == i)
            self.B[i] = (np.sum(gamma[mask], axis=0))/np.sum(gamma, axis=0)
    
    # Problem 6
    def fit(self, stats_vec):
        """
        Fit the HMM model parameters to a given set of statistics.

        Parameters
        ----------
        z : ndarray of shape (T,)
            Observation sequence on which to train the model.
        pi : Stochastic ndarray of shape (n,)
            Initial state distribution
        A : Stochastic ndarray of shape (n, n)
            Initial state transition matrix
        B : Stochastic ndarray of shape (m, n)
            Initial state observation matrix
        max_iter : int
            The maximum number of iterations to take
        tol : float
            The convergence threshold for change in log-probability
        """
        ### Initialize self.pi, self.A, self.B
        self.pi = pi 
        self.A = A 
        self.B = B 

        ### Run the iteration
        # Calculate log-likelihood P(z|(pi,A,B))
        alpha, c = self._forward(z)
        self.log_prob = -np.sum(np.log(c))       # From page 236 of lab spec

        for _ in range(max_iter):
            # Run backward pass 
            beta = self._backward(z, c)

            # Compute xi and gamma probabilities 
            xi, gamma = self._xi(z, alpha, beta, c)

            # Update model parameters 
            self._estimate(z, xi, gamma)

            # Compute log P(z|theta) according to new parameters 
            alpha, c = self._forward(z)
            log_prob_new = -np.sum(np.log(c))

            # Check for convergence 
            if np.abs(log_prob_new - self.log_prob) < tol:
                break 
            self.log_prob = log_prob_new 
        