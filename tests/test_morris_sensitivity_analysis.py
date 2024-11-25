# tests/test_morris_sensitivity_analysis.py

import unittest
from modules.morris_sensitivity_analysis import run_morris_analysis_for_dimensionality_reduction

class TestMorrisSensitivityAnalysis(unittest.TestCase):
    def test_run_morris_analysis(self):
        # Define a simple model

        def function_of_interest(X):
            E, F, L, I = X
            Y = F * L**3 / (3 * E * I)
            return [Y]

        # Problem definition for the cantilever beam model
        problem = {
            'num_vars': 4,
            'names': ['E', 'F', 'L', 'I'],
            'distributions': [
                {'type': 'Beta', 'params': [0.9, 3.1, 2.8e7, 4.8e7]},  # E
                {'type': 'LogNormalMuSigma', 'params': [3.0e4, 9.0e3, 15.0e3]},  # F
                {'type': 'Uniform', 'params': [250., 260.]},  # L
                {'type': 'Beta', 'params': [2.5, 4, 310., 450.]}  # I
            ]
        }

        model = function_of_interest        

        N = 100  # Use a small N for testing

        # Run the function
        results = run_morris_analysis_for_dimensionality_reduction(N, model, problem)

        # Unpack results
        non_influential_indices, mu_star_values, sigma_values, fig = results

        # Assertions
        self.assertIsInstance(non_influential_indices, list)
        self.assertEqual(len(mu_star_values), problem['num_vars'])
        self.assertEqual(len(sigma_values), problem['num_vars'])
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main()
