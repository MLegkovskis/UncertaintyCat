import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import openturns as ot
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION

def sobol_sensitivity_analysis(N, model, problem, model_code_str, language_model='groq'):
    """Perform Sobol sensitivity analysis.
    
    Parameters
    ----------
    N : int
        Number of samples for Sobol analysis
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    model_code_str : str
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    """
    try:
        # Verify input types
        if not isinstance(model, ot.Function):
            raise TypeError("Model must be an OpenTURNS Function")
        if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
            raise TypeError("Problem must be an OpenTURNS Distribution or JointDistribution")
            
        # Get dimension from the model's input dimension
        dimension = model.getInputDimension()
        
        # Create independent copy of the distribution for Sobol analysis
        marginals = [problem.getMarginal(i) for i in range(dimension)]
        independent_dist = ot.JointDistribution(marginals)
        
        # Get variable names
        variable_names = [problem.getMarginal(i).getDescription()[0] for i in range(dimension)]
        
        # Create Sobol algorithm
        compute_second_order = True
        sie = ot.SobolIndicesExperiment(independent_dist, N, compute_second_order)
        input_design = sie.generate()
        
        # Evaluate model
        output_design = model(input_design)
        
        # Calculate Sobol indices
        sensitivity_analysis = ot.SaltelliSensitivityAlgorithm(input_design, output_design, N)
        
        # Get first and total order indices
        S1 = sensitivity_analysis.getFirstOrderIndices()
        ST = sensitivity_analysis.getTotalOrderIndices()
        S2 = sensitivity_analysis.getSecondOrderIndices()
        
        # Get confidence intervals
        S1_interval = sensitivity_analysis.getFirstOrderIndicesInterval()
        ST_interval = sensitivity_analysis.getTotalOrderIndicesInterval()
        
        # Create DataFrame for indices
        indices_data = []
        for i, name in enumerate(variable_names):
            # Get confidence intervals for this index
            S1_lower = S1_interval.getLowerBound()[i]
            S1_upper = S1_interval.getUpperBound()[i]
            ST_lower = ST_interval.getLowerBound()[i]
            ST_upper = ST_interval.getUpperBound()[i]
            
            indices_data.append({
                'Variable': name,
                'First Order': float(S1[i]),
                'First Order CI': f"[{S1_lower:.4f}, {S1_upper:.4f}]",
                'Total Order': float(ST[i]),
                'Total Order CI': f"[{ST_lower:.4f}, {ST_upper:.4f}]"
            })
        
        # Plot results
        st.write("### Sobol Sensitivity Indices")
        st.write("This analysis shows the contribution of each input variable to the output variance.")
        
        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(variable_names))
        width = 0.35
        
        ax.bar(x - width/2, [d['First Order'] for d in indices_data], width, 
               label='First Order', color='skyblue')
        ax.bar(x + width/2, [d['Total Order'] for d in indices_data], width, 
               label='Total Order', color='lightcoral')
        
        ax.set_ylabel('Sensitivity Index')
        ax.set_title('Sobol Sensitivity Indices')
        ax.set_xticks(x)
        ax.set_xticklabels(variable_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Display numerical results
        st.write("\n### Numerical Results")
        for data in indices_data:
            st.write(f"**{data['Variable']}**")
            st.write(f"- First Order Index: {data['First Order']:.4f} ({data['First Order CI']})")
            st.write(f"- Total Order Index: {data['Total Order']:.4f} ({data['Total Order CI']})")
        
        # Calculate total sensitivity
        total_sensitivity = sum(d['First Order'] for d in indices_data)
        st.write(f"\nTotal First Order Sensitivity: {total_sensitivity:.4f}")
        
        if abs(1 - total_sensitivity) > 0.1:
            st.write("\n**Note:** The sum of first-order indices is significantly different from 1, "
                     "indicating important interaction effects between variables.")
        
        # Generate prompt for GPT
        indices_table = "\n".join(
            f"- {data['Variable']}:\n"
            f"  First Order: {data['First Order']:.4f} {data['First Order CI']}\n"
            f"  Total Order: {data['Total Order']:.4f} {data['Total Order CI']}"
            for data in indices_data
        )
        
        # Add distribution information
        dist_info = "\n".join(
            f"- {name}: {problem.getMarginal(i).__class__.__name__}, "
            f"parameters {problem.getMarginal(i).getParameter()}"
            for i, name in enumerate(variable_names)
        )
        
        prompt = f"""
Analyze these Sobol sensitivity analysis results:

```python
{model_code_str}
```

Input Distributions:
{dist_info}

Sobol Indices:
{indices_table}

Total First Order Sensitivity: {total_sensitivity:.4f}

Please provide:
1. A technical interpretation of these sensitivity indices
2. Explanation of which parameters are most influential and why
3. Discussion of any significant interaction effects
4. Recommendations for model simplification or improvement

{RETURN_INSTRUCTION}
"""
        
        # Call GPT API and display response
        response = call_groq_api(prompt, language_model)
        st.markdown(response)
        
    except Exception as e:
        st.error(f"Error in Sobol sensitivity analysis: {str(e)}")
        raise
