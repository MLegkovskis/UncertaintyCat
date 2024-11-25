# tests/test_correlation_analysis.py

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from modules.correlation_analysis import correlation_analysis
import streamlit as st

class TestCorrelationAnalysis(unittest.TestCase):
    @patch('modules.correlation_analysis.call_groq_api')
    def test_correlation_analysis_runs(self, mock_call_groq_api):
        # Mock the API response
        mock_call_groq_api.return_value = "Mocked AI response"

        # Define a simple model
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
        model_code_str = 'def model(X): return [X[0] + X[1]]'

        # Initialize st.session_state
        st.session_state.clear()

        # Run the function (we won't check the plots here)
        correlation_analysis(model, problem, model_code_str)

        # Check if the expected keys are in session state
        self.assertIn('correlation_response_markdown', st.session_state)
        self.assertIn('correlation_fig', st.session_state)

if __name__ == '__main__':
    unittest.main()
