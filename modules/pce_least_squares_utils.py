import streamlit as st
import openturns as ot
import numpy as np
import plotly.graph_objects as go
import pickle
from modules.openturns_utils import get_ot_distribution, get_ot_model, SuperChaosResult
from modules.statistical_utils import problem_to_python_code

def get_variable_names(distribution):
    """Extract variable names from OpenTURNS distribution."""
    dimension = distribution.getDimension()
    names = []
    for i in range(dimension):
        names.append(distribution.getMarginal(i).getDescription()[0])
    return names

def pce_sobol(
    train_sample_size,
    validation_sample_size,
    model,
    problem,
    model_code_str,
    language_model="llama-3.3-70b-versatile",
    basis_size_factor=1.0,
    use_model_selection=True
):
    """
    Perform PCE analysis and compute Sobol indices.
    """
    try:
        # Get the input distribution and the model
        distribution = get_ot_distribution(problem)
        model_g = get_ot_model(model, problem)
        
        # Create samples
        inputTrain = distribution.getSample(train_sample_size)
        outputTrain = model_g(inputTrain)
        
        # Create PCE basis
        dimension = distribution.getDimension()
        marginalList = []
        for i in range(dimension):
            marginal = distribution.getMarginal(i)
            # Create orthogonal basis for each marginal distribution
            marginalPolynomial = ot.StandardDistributionPolynomialFactory(marginal)
            marginalList.append(marginalPolynomial)
        
        multivariateBasis = ot.OrthogonalProductPolynomialFactory(marginalList)
        basisSize = int(basis_size_factor * train_sample_size)
        
        # Create and run PCE algorithm
        if use_model_selection:
            selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
        else:
            selectionAlgorithm = ot.PenalizedLeastSquaresAlgorithmFactory()
            
        projectionStrategy = ot.LeastSquaresStrategy(
            inputTrain, outputTrain, selectionAlgorithm
        )
        adaptiveStrategy = ot.FixedStrategy(multivariateBasis, basisSize)
        chaosAlgorithm = ot.FunctionalChaosAlgorithm(
            inputTrain, outputTrain, distribution, adaptiveStrategy, projectionStrategy
        )
        chaosAlgorithm.run()
        polynomialChaosResult = chaosAlgorithm.getResult()
        metamodel = polynomialChaosResult.getMetaModel()
        
        # Validation
        inputTest = distribution.getSample(validation_sample_size)
        outputTest = model_g(inputTest)
        predicted_output = metamodel(inputTest)
        
        # Convert to numpy for metrics
        true_output_np = np.array(outputTest)
        predicted_output_np = np.array(predicted_output)
        
        # Compute metrics
        mse = np.mean((true_output_np - predicted_output_np) ** 2)
        r2 = 1 - mse / np.var(true_output_np)
        
        # Plot validation results
        fig_validation = go.Figure()
        fig_validation.add_trace(go.Scatter(
            x=true_output_np.flatten(),
            y=predicted_output_np.flatten(),
            mode='markers',
            name='Validation Points'
        ))
        
        # Add diagonal line
        min_val = min(true_output_np.min(), predicted_output_np.min())
        max_val = max(true_output_np.max(), predicted_output_np.max())
        fig_validation.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_validation.update_layout(
            title=f'PCE Validation (R² = {r2:.3f})',
            xaxis_title='True Values',
            yaxis_title='Predicted Values',
            showlegend=True
        )
        
        # Display validation plot
        st.plotly_chart(fig_validation)
        
        # Print parameters and validation results
        numberOfCoefficients = polynomialChaosResult.getIndices().getSize()
        pce_result_markdown = f"""
## Settings

- Training sample size = {train_sample_size}
- Validation sample size = {validation_sample_size}
- Maximum basis size factor = {basis_size_factor}
- Basis size = {basisSize}
- Use model selection? {use_model_selection}
- Number of coefficients = {numberOfCoefficients}

## Validation

- R2 score = {r2:.4f}
"""
        st.markdown(pce_result_markdown)
        
        # Compute and plot Sobol indices
        st.subheader("Sensitivity Analysis")
        
        # Get variable names
        variable_names = get_variable_names(distribution)
        first_order = []
        total_order = []
        
        # Create functional chaos result for Sobol indices
        sobol = ot.FunctionalChaosSobolIndices(polynomialChaosResult)
        
        for i in range(dimension):
            first_order.append(sobol.getSobolIndex(i))
            total_order.append(sobol.getSobolTotalIndex(i))
        
        fig_sobol = go.Figure()
        fig_sobol.add_trace(go.Bar(
            name='First Order',
            x=variable_names,
            y=first_order,
            marker_color='blue'
        ))
        fig_sobol.add_trace(go.Bar(
            name='Total Order',
            x=variable_names,
            y=total_order,
            marker_color='red'
        ))
        
        fig_sobol.update_layout(
            title='Sobol Sensitivity Indices',
            xaxis_title='Variables',
            yaxis_title='Sensitivity Index',
            barmode='group',
            showlegend=True
        )
        
        st.plotly_chart(fig_sobol)
        
        # Store results in session state
        if 'pce_results' not in st.session_state:
            st.session_state.pce_results = {}
            
        st.session_state.pce_results.update({
            'metamodel': metamodel,
            'r2': r2,
            'sobol_indices': {
                'first_order': dict(zip(variable_names, first_order)),
                'total_order': dict(zip(variable_names, total_order))
            }
        })
        
        # Generate surrogate model code
        superPCE = SuperChaosResult(polynomialChaosResult)
        pce_code = superPCE.toPython()
        problem_code = problem_to_python_code(problem)
        
        # Generate the complete surrogate code
        pce_least_squares_code = f"""{pce_code}

# Problem definition
{problem_code}
model = function_of_interest
"""
        
        # Save data
        pce_least_squares_save_Python(pce_least_squares_code)
        pce_least_squares_save_Pickle(distribution, metamodel)
        
        # Set the flag
        st.session_state.pce_least_squares_generated = True
        st.success("PCE Surrogate Model generated successfully.")
        
    except Exception as e:
        st.error(f"Error in PCE analysis: {str(e)}")
        raise

@st.cache_data
def pce_least_squares_save_Python(pce_least_squares_code):
    """Save the Python surrogate model code."""
    st.download_button(
        "Download Python surrogate",
        data=pce_least_squares_code,
        file_name="pce_surrogate.py"
    )

@st.cache_data
def pce_least_squares_save_Pickle(distribution, metamodel):
    """Save the Pickle data for the surrogate model."""
    surrogate_model_data = [distribution, metamodel]
    st.download_button(
        "Download Pickle surrogate",
        data=pickle.dumps(surrogate_model_data),
        file_name="pce_surrogate.pkl"
    )

def reset_pce_least_squares_results():
    """Reset PCE results in session state."""
    keys_to_reset = [
        'pce_results',
        'pce_least_squares_generated'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
