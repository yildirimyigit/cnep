import numpy as np
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:
    def __init__(self, n_components=5, covariance_type='full'):
        """
        Initializes a Gaussian Mixture Model (GMM) for 1D trajectories

        Args:
            n_components (int): The number of Gaussian components in the mixture.
            covariance_type (str): Type of covariance matrix. Consider 'full', 'diag', etc.
                                   Refer to sklearn's GaussianMixture documentation.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmm = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type
        )

    def fit(self, x, Y):
        """
        Trains the GMM on 1D trajectories.

        Args:
            x (array-like):  The phase variable (time axis).
            Y (array-like):  Demonstration trajectories (shape: n_trajectories, n_time_steps)
        """
        data = np.hstack((x.reshape(-1, 1), Y.T))  # Combine phase and trajectories
        self.gmm.fit(data)

    def generate_conditioned(self, x, via_points, tolerance=0.1):
        """Generates trajectories conditioned on via-points using a sampling-based approach."""
        via_point_times, via_point_positions = zip(*via_points)
        via_point_indices = np.searchsorted(x, via_point_times)  # Find nearest indices

        while True:  # Rejection sampling loop
            new_trajectory = self.gmm.sample(len(x))[0][:, 1] 

            satisfactory = True
            for idx, pos in zip(via_point_indices, via_point_positions):
                # Check closeness to via-point (using absolute distance here)
                if abs(new_trajectory[idx] - pos) > tolerance:  
                    satisfactory = False
                    break

            if satisfactory:
                return new_trajectory
