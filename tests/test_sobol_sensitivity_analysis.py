# tests/test_sobol_sensitivity_analysis.py

import unittest
from unittest.mock import patch
import numpy as np  # Add this import
import matplotlib
matplotlib.use('Agg')
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis
import streamlit as st

class TestSobolSensitivityAnalysis(unittest.TestCase):
    @patch('modules.sobol_sensitivity_analysis.call_groq_api')
    def test_sobol_sensitivity_analysis_runs(self, mock_call_groq_api):
        # Mock the API response
        mock_call_groq_api.return_value = "Mocked AI response"

        # Define a simple model
        def model(X):
            return [np.sin(X[0]) * X[1]]

        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Uniform', 'params': [0, np.pi]},
                {'type': 'Uniform', 'params': [0, 1]}
            ]
        }
        model_code_str = 'def model(X): return [np.sin(X[0]) * X[1]]'
        N = 64  # Use a small power of 2 for testing

        # Initialize st.session_state
        st.session_state.clear()

        # Run the function
        sobol_sensitivity_analysis(N, model, problem, model_code_str)

        # Check if the expected keys are in session state
        self.assertIn('sobol_response_markdown', st.session_state)
        self.assertIn('sobol_fig', st.session_state)

if __name__ == '__main__':
    unittest.main()
