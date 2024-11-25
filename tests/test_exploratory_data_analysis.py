# tests/test_exploratory_data_analysis.py

import unittest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for testing
from modules.exploratory_data_analysis import exploratory_data_analysis
from unittest.mock import patch

class TestExploratoryDataAnalysis(unittest.TestCase):
    @patch('modules.exploratory_data_analysis.call_groq_api')
    def test_eda_runs(self, mock_call_groq_api):
        # Set up the mock to return a fixed markdown string
        mock_call_groq_api.return_value = "Mocked AI response"

        # Create dummy data
        data = pd.DataFrame({
            'x1': [1, 2, 3, 4],
            'x2': [4, 3, 2, 1],
            'Y': [5, 5, 5, 5]
        })

        N = 4
        model = lambda X: [X[0] + X[1]]
        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Uniform', 'params': [0, 1]},
                {'type': 'Uniform', 'params': [0, 1]}
            ]
        }
        model_code_str = 'def model(X): return [X[0] + X[1]]'

        # Run the function (we won't check the plots here)
        exploratory_data_analysis(data, N, model, problem, model_code_str)

        # If no exceptions are raised, the test passes
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
