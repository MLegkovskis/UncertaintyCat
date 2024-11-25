# tests/test_ml_analysis.py

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from modules.ml_analysis import ml_analysis
import streamlit as st

class TestMLAnalysis(unittest.TestCase):
    @patch('modules.ml_analysis.call_groq_api')
    def test_ml_analysis_runs(self, mock_call_groq_api):
        # Mock the API response
        mock_call_groq_api.return_value = "Mocked AI response"

        # Create dummy data
        np.random.seed(42)
        data = pd.DataFrame({
            'x1': np.random.normal(0, 1, 100),
            'x2': np.random.normal(0, 1, 100),
            'Y': np.random.normal(0, 1, 100)
        })
        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Normal', 'params': [0, 1]},
                {'type': 'Normal', 'params': [0, 1]}
            ]
        }
        model_code_str = 'def model(X): return [X[0] + X[1]]'

        # Initialize st.session_state
        st.session_state.clear()

        # Run the function
        ml_analysis(data, problem, model_code_str)

        # Check if the expected keys are in session state
        self.assertIn('ml_response_markdown', st.session_state)
        self.assertIn('ml_shap_summary_fig', st.session_state)
        self.assertIn('ml_dependence_fig', st.session_state)

if __name__ == '__main__':
    unittest.main()
