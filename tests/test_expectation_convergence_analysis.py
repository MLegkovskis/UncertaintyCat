# tests/test_expectation_convergence_analysis.py

import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
from modules.expectation_convergence_analysis import expectation_convergence_analysis
import streamlit as st

class TestExpectationConvergenceAnalysis(unittest.TestCase):
    @patch('modules.expectation_convergence_analysis.call_groq_api')
    def test_expectation_convergence_analysis_runs(self, mock_call_groq_api):
        # Mock the API response
        mock_call_groq_api.return_value = "Mocked AI response"

        # Define a simple model
        def model(X):
            return [X[0] * 2 + X[1]]

        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Normal', 'params': [0, 1]},
                {'type': 'Normal', 'params': [0, 1]}
            ]
        }
        model_code_str = 'def model(X): return [X[0] * 2 + X[1]]'
        N_samples = 100  # Use a small number for testing

        # Initialize st.session_state
        st.session_state.clear()

        # Run the function
        expectation_convergence_analysis(model, problem, model_code_str, N_samples=N_samples)

        # Check if the expected keys are in session state
        self.assertIn('expectation_response_markdown', st.session_state)
        self.assertIn('expectation_fig', st.session_state)

if __name__ == '__main__':
    unittest.main()
