# tests/test_hsic_analysis.py

import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
from modules.hsic_analysis import hsic_analysis
import streamlit as st

class TestHSICAnalysis(unittest.TestCase):
    @patch('modules.hsic_analysis.call_groq_api')
    def test_hsic_analysis_runs(self, mock_call_groq_api):
        # Mock the API response
        mock_call_groq_api.return_value = "Mocked AI response"

        # Define a simple model
        def model(X):
            return [X[0] * X[1]]

        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Uniform', 'params': [-1, 1]},
                {'type': 'Uniform', 'params': [-1, 1]}
            ]
        }
        model_code_str = 'def model(X): return [X[0] * X[1]]'

        # Initialize st.session_state
        st.session_state.clear()

        # Run the function
        hsic_analysis(model, problem, model_code_str)

        # Check if the expected keys are in session state
        self.assertIn('hsic_response_markdown', st.session_state)
        self.assertIn('hsic_fig', st.session_state)

if __name__ == '__main__':
    unittest.main()
