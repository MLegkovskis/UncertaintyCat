# tests/test_monte_carlo.py

import unittest
import numpy as np
from modules.monte_carlo import monte_carlo_simulation

class TestMonteCarloSimulation(unittest.TestCase):
    def test_simulation_output(self):
        # Define a simple model and problem for testing
        def model(X):
            return [X[0] + X[1]]

        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Uniform', 'params': [0, 1]},
                {'type': 'Uniform', 'params': [0, 1]}
            ]
        }

        N = 10
        data = monte_carlo_simulation(N, model, problem)
        
        # Check if the output data has the correct length
        self.assertEqual(len(data), N)
        # Check if 'Y' column exists
        self.assertIn('Y', data.columns)
        # Check if 'Y' values are within expected range
        self.assertTrue((data['Y'] >= 0).all() and (data['Y'] <= 2).all())

if __name__ == '__main__':
    unittest.main()
