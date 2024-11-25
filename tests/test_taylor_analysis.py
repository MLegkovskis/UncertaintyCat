# tests/test_taylor_analysis.py

import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
from modules.taylor_analysis import taylor_analysis
import streamlit as st

class TestTaylorAnalysis(unittest.TestCase):
    @patch('modules.taylor_analysis.call_groq_api')
    def test_taylor_analysis_runs(self, mock_call_groq_api):
        # Mock the API response
        mock_call_groq_api.return_value = "Mocked AI response"

        # Define a simple model
        def model(X):
            return [X[0] ** 2 + X[1] ** 2]

        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Uniform', 'params': [1, 2]},  # Changed from Normal to Uniform
                {'type': 'Uniform', 'params': [1, 2]}   # Changed from Normal to Uniform
            ]
        }
        model_code_str = 'def model(X): return [X[0] ** 2 + X[1] ** 2]'

        # Initialize st.session_state
        st.session_state.clear()

        # Run the function
        taylor_analysis(model, problem, model_code_str)

        # Check if the expected keys are in session state
        self.assertIn('taylor_response_markdown', st.session_state)
        self.assertIn('taylor_fig', st.session_state)

if __name__ == '__main__':
    unittest.main()
