"""
Distribution Fitting module for UncertaintyCat.

This module provides functionality to fit probability distributions to data
and evaluate the goodness of fit using various statistical tests.
"""

import streamlit as st
import numpy as np
import pandas as pd
import openturns as ot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import matplotlib.pyplot as plt
import os
import plotly.express as px

def get_distribution_factories():
    """
    Get OpenTURNS distribution factories grouped by type.
    
    Returns
    -------
    dict
        Dictionary with distribution types as keys and lists of factories as values
    """
    factories = {
        "Continuous Univariate": ot.DistributionFactory.GetContinuousUniVariateFactories(),
        "Discrete Univariate": ot.DistributionFactory.GetDiscreteUniVariateFactories(),
        "Continuous Multivariate": ot.DistributionFactory.GetContinuousMultiVariateFactories(),
        "Discrete Multivariate": ot.DistributionFactory.GetDiscreteMultiVariateFactories()
    }
    
    # Remove problematic factories for common use cases
    problematic_factories = [
        "BetaFactory",     # Requires data in [0, 1]
        "DirichletFactory", # Requires data in simplex
        "LogisticFactory", # Often causes convergence issues
        "TrapezoidalFactory", # Often causes convergence issues
        "TriangularFactory" # Often causes convergence issues
    ]
    
    for category in factories:
        factories[category] = [f for f in factories[category] 
                              if not any(p in f.getClassName() for p in problematic_factories)]
    
    return factories

def fit_distribution(data, factory):
    """
    Fit a distribution to data using the provided factory.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data to fit the distribution to
    factory : ot.DistributionFactory
        Factory to use for fitting
        
    Returns
    -------
    ot.Distribution
        Fitted distribution
    dict
        Fitting results including statistics
    """
    try:
        # Convert data to OpenTURNS sample
        ot_sample = ot.Sample(data.reshape(-1, 1))
        
        # Build the distribution
        result = factory.buildEstimator(ot_sample)
        distribution = result.getDistribution()
        
        # Store the factory type for later code generation
        factory_name = factory.getClassName().replace("Factory", "")
        
        # Calculate fitting statistics
        stats = {}
        stats["factory_type"] = factory_name  # Store the factory type for code generation
        
        try:
            # Kolmogorov-Smirnov test
            ks_test = ot.FittingTest.Kolmogorov(ot_sample, distribution)
            stats["KS p-value"] = ks_test.getPValue()
        except Exception:
            stats["KS p-value"] = 0.0
            
        # BIC (approximate)
        n = len(data)
        k = len(distribution.getParameter())
        
        try:
            # Compute log-likelihood
            log_likelihood = distribution.computeLogPDF(ot_sample).computeMean()[0] * n
            stats["BIC"] = -2 * log_likelihood + k * np.log(n)
            
            # AIC (approximate)
            stats["AIC"] = -2 * log_likelihood + 2 * k
        except Exception:
            stats["BIC"] = float('inf')
            stats["AIC"] = float('inf')
        
        return distribution, stats
    
    except Exception as e:
        return None, {}

def plot_distribution_fit(data, distributions, titles, stats=None, selected_idx=None):
    """
    Create plots comparing data histogram with fitted distributions.
    
    Parameters
    ----------
    data : numpy.ndarray
        Original data
    distributions : list
        List of fitted OpenTURNS distributions
    titles : list
        List of distribution names
    stats : list, optional
        List of statistics dictionaries for each distribution
    selected_idx : int, optional
        Index of the selected distribution to highlight
        
    Returns
    -------
    fig
        Plotly figure object
    """
    # Create a figure with subplots
    n_plots = len(distributions)
    
    if n_plots == 0:
        return None
    
    # Choose an appropriate layout based on number of distributions
    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots <= 2:
        rows, cols = 1, 2
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    elif n_plots <= 12:
        rows, cols = 3, 4
    elif n_plots <= 16:
        rows, cols = 4, 4
    elif n_plots <= 20:
        rows, cols = 4, 5
    else:
        # For large number of plots, calculate rows and columns
        cols = min(5, n_plots)  # Maximum 5 columns
        rows = (n_plots + cols - 1) // cols  # Ceiling division
    
    # Create subplot titles with BIC values if available
    plot_titles = []
    for i, title in enumerate(titles):
        if stats and i < len(stats):
            bic_value = stats[i].get("BIC", float('inf'))
            plot_titles.append(f"{title} (BIC: {round(bic_value, 3)})")
        else:
            plot_titles.append(title)
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=plot_titles)
    
    # Calculate histogram bins using Freedman-Diaconis rule
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    if bin_width > 0:
        n_bins = int((np.max(data) - np.min(data)) / bin_width)
        n_bins = min(max(n_bins, 10), 50)  # Between 10 and 50 bins
    else:
        n_bins = 10
    
    # Generate histogram
    hist, bins = np.histogram(data, bins=n_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot each distribution
    for i, distribution in enumerate(distributions):
        row = i // cols + 1
        col = i % cols + 1
        
        # Determine if this plot should be highlighted
        is_selected = (i == selected_idx) if selected_idx is not None else False
        
        # Set visual attributes based on selection status
        opacity = 0.7
        histogram_color = "lightblue"
        curve_color = "red"
        line_width = 2
        
        if is_selected:
            # Highlight selected plot
            histogram_color = "rgba(144, 238, 144, 0.7)"  # light green for histogram
            curve_color = "green"
            line_width = 3
            opacity = 0.9
        
        # Add histogram
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                name="Data",
                opacity=opacity,
                marker_color=histogram_color,
                showlegend=i==0
            ),
            row=row, col=col
        )
        
        # Generate distribution PDF values
        x = np.linspace(min(data), max(data), 200)
        try:
            y = [distribution.computePDF(xi) for xi in x]
            
            # Add PDF curve
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=titles[i],
                    line=dict(color=curve_color, width=line_width),
                    showlegend=i==0
                ),
                row=row, col=col
            )
        except Exception as e:
            st.warning(f"Error plotting {titles[i]}: {str(e)}")
    
    # Update layout for a more compact view
    fig.update_layout(
        height=max(400, 250 * rows),  # Minimum height of 400px
        width=min(1000, 250 * cols),  # Maximum width of 1000px
        title_text="Distribution Fitting Results",
        barmode='overlay',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Make subplot spacing tighter
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        font=dict(size=10),  # Smaller font size
    )
    
    fig.update_annotations(font_size=10)
    
    # Make the axes tighter
    fig.update_xaxes(showticklabels=True, title=None)
    fig.update_yaxes(showticklabels=True, title=None)
    
    return fig

def create_multivariate_distribution(marginals, variable_names):
    """
    Create a multivariate distribution from marginals assuming independence.
    
    Parameters
    ----------
    marginals : list
        List of OpenTURNS distributions
    variable_names : list
        List of variable names
        
    Returns
    -------
    openturns.JointDistribution
        The multivariate distribution
    """
    # Set descriptions for all marginals
    for i, marginal in enumerate(marginals):
        marginal.setDescription([variable_names[i]])
    
    # Create independent joint distribution
    joint_dist = ot.JointDistribution(marginals)
    
    return joint_dist

def fit_normal_copula(data, marginals):
    """
    Fit a Normal (Gaussian) copula to the data, using the provided marginal distributions.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data used for fitting the copula
    marginals : list
        List of fitted OpenTURNS marginal distributions
        
    Returns
    -------
    openturns.NormalCopula
        The fitted Normal copula
    """
    # Transform data to uniform using the marginals' CDFs
    uniform_samples = []
    
    for i, (col, marginal) in enumerate(zip(data.columns, marginals)):
        # Transform the data: compute the CDF for each data point
        uniform_data = np.array([marginal.computeCDF(x) for x in data[col].values])
        uniform_samples.append(uniform_data)
    
    # Combine the uniform data into a single sample
    uniform_sample = np.column_stack(uniform_samples)
    
    # Convert to OpenTURNS Sample
    ot_uniform_sample = ot.Sample(uniform_sample)
    
    # Fit a Normal copula
    copula_factory = ot.NormalCopulaFactory()
    fitted_copula = copula_factory.build(ot_uniform_sample)
    
    return fitted_copula

def create_dependent_multivariate_distribution(marginals, variable_names, copula):
    """
    Create a multivariate distribution from marginals and a copula.
    
    Parameters
    ----------
    marginals : list
        List of OpenTURNS distributions
    variable_names : list
        List of variable names
    copula : openturns.Copula
        The copula modeling dependence structure
        
    Returns
    -------
    openturns.ComposedDistribution
        The multivariate distribution with dependencies
    """
    # Set descriptions for all marginals
    for i, marginal in enumerate(marginals):
        marginal.setDescription([variable_names[i]])
    
    # Create joint distribution with the provided copula
    joint_dist = ot.ComposedDistribution(marginals, copula)
    
    return joint_dist

def get_distribution_code_string(dist):
    """
    Generate a string representing the distribution code.
    
    Parameters
    ----------
    dist : ot.Distribution
        OpenTURNS distribution
        
    Returns
    -------
    str
        Distribution code string
    """
    # Try to get the actual OpenTURNS class name
    try:
        ot_class_name = dist.getName()
    except Exception:
        try:
            ot_class_name = dist.getClassName()
            if ot_class_name.endswith("Implementation"):
                ot_class_name = ot_class_name[:-14]  # Remove "Implementation"
        except Exception:
            ot_class_name = dist.getClassName().replace("Factory", "")
    
    # Format with full precision
    params = dist.getParameter()
    param_str = ", ".join([str(p) for p in params])
    formula_str = f"ot.{ot_class_name}({param_str})"
    
    return formula_str

def distribution_fitting_page():
    """
    Main function for the distribution fitting page.
    """
    st.markdown("<h2 style='color: #1E88E5;'>Distribution Fitting</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #1E88E5;'>
    This tool allows you to fit probability distributions to your data and use them for uncertainty quantification analysis.
    The fitted distributions can be used to replace manually specified distributions in your model.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables for this page
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'fitted_distributions' not in st.session_state:
        st.session_state.fitted_distributions = {}
    if 'selected_distributions' not in st.session_state:
        st.session_state.selected_distributions = {}
    if 'problem_distribution' not in st.session_state:
        st.session_state.problem_distribution = None
    
    # Data source selection
    st.markdown("<h3 style='color: #424242;'>Step 1: Import Data</h3>", unsafe_allow_html=True)
    
    # Load sample data once for use in both upload and manual entry sections
    sample_data = ""
    sample_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_inputs.csv")
    try:
        with open(sample_file_path, 'r') as f:
            sample_data = f.read()
    except Exception as e:
        st.error(f"Error reading sample file: {e}")
        # Fallback sample data if file can't be read
        sample_data = """E,F,L,I
34359951.21,21373.82,253.31,397.88
28057355.74,30617.79,253.61,403.44
35431982.12,34967.84,253.19,371.93
28041948.69,21315.02,250.95,366.93"""
    
    data_source = st.radio(
        "Select Data Source",
        ["Upload CSV/Excel File", "Enter Data Manually"],
        horizontal=True
    )
    
    # Data upload or manual entry
    data_df = None
    
    if data_source == "Upload CSV/Excel File":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your data file", 
                type=["csv", "xlsx", "xls"],
                help="Upload a CSV or Excel file with your data. Each column should represent a variable with a header row for variable names."
            )
            
            st.markdown("""
            <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em;'>
            <b>Expected Format:</b> CSV or Excel file with variable names in the first row, and values in subsequent rows.<br>
            Each column represents a variable for which you want to fit a distribution.
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("<h4>Example CSV Format:</h4>", unsafe_allow_html=True)
            st.code("""E,F,L,I
34359951.21,21373.82,253.31,397.88
28057355.74,30617.79,253.61,403.44
35431982.12,34967.84,253.19,371.93
28041948.69,21315.02,250.95,366.93
36444562.19,26905.25,252.88,377.62
...""", language="text")
            
            # Add download button for sample data
            b64 = base64.b64encode(sample_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sample_inputs.csv" style="display: inline-block; padding: 0.375rem 0.75rem; font-size: 1rem; text-align: center; text-decoration: none; vertical-align: middle; cursor: pointer; border: 1px solid #0d6efd; border-radius: 0.25rem; color: #fff; background-color: #0d6efd; margin-top: 10px;">Download Sample Data</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data_df = pd.read_csv(uploaded_file)
                else:
                    data_df = pd.read_excel(uploaded_file)
                
                st.session_state.uploaded_data = data_df
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    else:  # Manual data entry
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;'>
            <b>Instructions:</b><br>
            • First line should contain variable names separated by commas<br>
            • Each subsequent line contains values for each variable<br>
            • Values should be separated by commas or spaces<br>
            • Each variable (column) should have the same number of values
            </div>
            """, unsafe_allow_html=True)
            
            manual_data = st.text_area(
                "Enter your data (first line: variable names, subsequent lines: values)",
                height=200,
                value=sample_data
            )
            
        with col2:
            st.markdown("<h4>Expected Format:</h4>", unsafe_allow_html=True)
            st.code("""E,F,L,I
34359951.21,21373.82,253.31,397.88
28057355.74,30617.79,253.61,403.44
35431982.12,34967.84,253.19,371.93
...""", language="text")
        
        if manual_data:
            try:
                # Parse the manual data
                rows = [row.strip() for row in manual_data.strip().split('\n') if row.strip()]
                
                if len(rows) < 2:
                    st.error("Please enter at least a header row and one data row")
                else:
                    # First row is headers
                    headers = []
                    if ',' in rows[0]:
                        headers = [h.strip() for h in rows[0].split(',') if h.strip()]
                    else:
                        headers = [h.strip() for h in rows[0].split() if h.strip()]
                    
                    # Parse data rows
                    data_rows = []
                    for row in rows[1:]:
                        if ',' in row:
                            values = [float(x.strip()) for x in row.split(',') if x.strip()]
                        else:
                            values = [float(x.strip()) for x in row.split() if x.strip()]
                        data_rows.append(values)
                    
                    # Check consistency
                    if not all(len(row) == len(headers) for row in data_rows):
                        st.error("All rows must have the same number of values as there are headers")
                    else:
                        # Create DataFrame
                        data_df = pd.DataFrame(data_rows, columns=headers)
                        st.session_state.uploaded_data = data_df
            
            except Exception as e:
                st.error(f"Error parsing data: {str(e)}")
    
    # Display the data if available
    if data_df is not None:
        st.markdown("<h3 style='color: #424242;'>Step 2: Review and Edit Data</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Data Preview")
            st.dataframe(data_df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("#### Data Statistics")
            stats_df = data_df.describe().T
            stats_df = stats_df[['count', 'mean', 'min', 'max']]
            stats_df = stats_df.round(2)
            st.dataframe(stats_df, use_container_width=True)
        
        # Add option to edit data
        st.markdown("#### Edit Data (Optional)")
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;'>
        You can add, remove, or modify data points directly in the table below. Changes will be automatically applied to the analysis.
        </div>
        """, unsafe_allow_html=True)
        
        edited_data = st.data_editor(
            data_df,
            use_container_width=True,
            num_rows="dynamic",
            height=300
        )
        
        # Replace the original data with edited data
        data_df = edited_data
        st.session_state.uploaded_data = data_df
        
        # Distribution fitting
        st.markdown("<h3 style='color: #424242;'>Step 3: Fit Distributions</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;'>
        For each selected variable, multiple probability distributions will be fitted and ranked according to goodness-of-fit criteria.
        The Bayesian Information Criterion (BIC) is used as the primary metric for ranking (lower values indicate better fit).
        Kolmogorov-Smirnov test results are also provided to assess the quality of fit (higher p-values indicate better fit).
        </div>
        """, unsafe_allow_html=True)
        
        # Select variables to fit
        variables = data_df.columns.tolist()
        selected_variables = st.multiselect(
            "Select Variables to Fit",
            variables,
            default=variables,
            help="Select the variables you want to fit distributions to."
        )
        
        if selected_variables:
            col1, col2 = st.columns(2)
            
            with col1:
                # Select distribution types to fit
                st.markdown("#### Distribution Types")
                
                # Add explanation for distribution types
                st.markdown("""
                <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;'>
                <b>Distribution Types:</b>
                <ul>
                    <li><b>Continuous Distributions:</b> Suitable for data that can take any value within a range (e.g., measurements, lengths, weights).</li>
                    <li><b>Discrete Distributions:</b> Suitable for data that can only take specific values (e.g., counts, integers).</li>
                </ul>
                <b>Selection Impact:</b>
                <ul>
                    <li>Selecting both types will test more distributions but may take longer to compute.</li>
                    <li>If your data is clearly continuous (e.g., physical measurements), you may only need continuous distributions.</li>
                    <li>For count data or integers, include discrete distributions.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                distribution_types = {
                    "Continuous": st.checkbox("Continuous Distributions", value=True),
                    "Discrete": st.checkbox("Discrete Distributions")
                }
            
            with col2:
                # Limit the number of distribution types
                st.markdown("#### Analysis Settings")
                max_dists = st.slider(
                    "Maximum distributions to evaluate per variable", 
                    min_value=3, 
                    max_value=20, 
                    value=10
                )
            
            factories = get_distribution_factories()
            selected_factories = []
            
            # Add continuous univariate factories
            if distribution_types["Continuous"]:
                selected_factories.extend(factories["Continuous Univariate"])
            
            # Add discrete univariate factories
            if distribution_types["Discrete"]:
                selected_factories.extend(factories["Discrete Univariate"])
            
            # Only show top N distributions based on fit quality
            selected_factories = selected_factories[:max_dists]
            
            # Add progress information
            st.info("Note: Some distributions may not be suitable for your data and will be automatically excluded.")
            
            # Fit button with improved styling
            fit_col, _ = st.columns([1, 3])
            with fit_col:
                fit_button = st.button(
                    "Fit Distributions", 
                    key="fit_distributions",
                    type="primary"
                )
            
            if fit_button:
                with st.spinner("Fitting distributions..."):
                    # Dictionary to store fitted distributions for each variable
                    all_fitted_dists = {}
                    
                    # Create a progress bar
                    total_fits = len(selected_variables) * len(selected_factories)
                    progress_bar = st.progress(0)
                    fit_counter = 0
                    
                    for var in selected_variables:
                        data = data_df[var].values
                        
                        # Skip if data has negative values and factory is for positive-only distributions
                        data_min = np.min(data)
                        
                        # Fit distributions using each factory
                        var_dists = []
                        var_stats = []
                        var_names = []
                        
                        for factory in selected_factories:
                            # Update progress
                            fit_counter += 1
                            progress_bar.progress(fit_counter / total_fits)
                            
                            factory_name = factory.getClassName().replace("Factory", "")
                            
                            # Skip incompatible factories based on data range
                            if data_min < 0:
                                positive_only = ["Gamma", "LogNormal", "Weibull", "Beta", "Exponential", 
                                                "ChiSquare", "LogUniform", "Rayleigh", "Rice", "Pareto"]
                                if any(pos in factory_name for pos in positive_only):
                                    continue
                                
                            dist, stats = fit_distribution(data, factory)
                            
                            if dist is not None and stats:
                                var_dists.append(dist)
                                var_stats.append(stats)
                                var_names.append(factory_name)
                        
                        # Sort by BIC (lower is better)
                        if var_dists:
                            sorted_indices = np.argsort([stats.get("BIC", float('inf')) for stats in var_stats])
                            var_dists = [var_dists[i] for i in sorted_indices]
                            var_stats = [var_stats[i] for i in sorted_indices]
                            var_names = [var_names[i] for i in sorted_indices]
                            
                            all_fitted_dists[var] = {
                                "distributions": var_dists,
                                "statistics": var_stats,
                                "names": var_names
                            }
                    
                    # Store in session state
                    st.session_state.fitted_distributions = all_fitted_dists
                    
                    # Display success message
                    st.success(f"Successfully fitted distributions for {len(all_fitted_dists)} variables!")
            
            # Display fitted distributions
            if st.session_state.fitted_distributions:
                st.markdown("<h3 style='color: #424242;'>Step 4: Review Fitting Results</h3>", unsafe_allow_html=True)
                
                # Create session state for explanations to show them only once
                if 'shown_table_explanation' not in st.session_state:
                    st.session_state.shown_table_explanation = False
                if 'shown_plot_explanation' not in st.session_state:
                    st.session_state.shown_plot_explanation = False
                
                # For each variable
                for var_index, (var, fit_results) in enumerate(st.session_state.fitted_distributions.items()):
                    if var in selected_variables:  # Only show selected variables
                        st.markdown(f"<h4 style='color: #1E88E5;'>{var}</h4>", unsafe_allow_html=True)
                        
                        # Display table of fitting statistics
                        st.markdown("#### Distribution Ranking")
                        
                        # Show explanation only for the first variable
                        if not st.session_state.shown_table_explanation:
                            st.markdown("""
                            <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;'>
                            <b>Understanding this table:</b>
                            <ul>
                                <li>Each row represents a different probability distribution that was tested against your data.</li>
                                <li>The table shows all distributions that could be successfully fitted to your data.</li>
                                <li><b>BIC (Bayesian Information Criterion)</b>: Lower values indicate a better fit. This is the primary criterion for ranking.</li>
                                <li><b>AIC (Akaike Information Criterion)</b>: Lower values indicate a better fit. Similar to BIC but with different penalties.</li>
                                <li><b>KS p-value</b>: Higher values (closer to 1.0) indicate a better fit. Values above 0.05 usually suggest a good fit.</li>
                            </ul>
                            The distributions are ranked from best to worst based on BIC value.
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.shown_table_explanation = True

                        stat_df = pd.DataFrame([
                            {
                                "Distribution": get_distribution_code_string(fit_results["distributions"][i]),
                                "BIC": round(stats.get("BIC", float('inf')), 3),
                                "AIC": round(stats.get("AIC", float('inf')), 3),
                                "KS p-value": round(stats.get("KS p-value", 0), 3)
                            }
                            for i, stats in enumerate(fit_results["statistics"])
                        ])


                        # Color code the table
                        def highlight_best(s):
                            if s.name == 'BIC' or s.name == 'AIC':
                                return ['background-color: rgba(76, 175, 80, 0.2)' if v == s.min() 
                                        else '' for v in s]
                            elif s.name == 'KS p-value':
                                return ['background-color: rgba(76, 175, 80, 0.2)' if v == s.max() 
                                        else '' for v in s]
                            else:
                                return ['' for v in s]
                            
                        st.dataframe(stat_df.style.apply(highlight_best), use_container_width=True)
                        
                        # Plot all distributions
                        st.markdown("#### Visual Comparison")
                        
                        # Show explanation only for the first variable
                        if not st.session_state.shown_plot_explanation:
                            st.markdown("""
                            <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;'>
                            <b>Understanding these plots:</b>
                            <ul>
                                <li>Each plot shows a <b>single distribution</b> (from the table above) compared to your actual data.</li>
                                <li>All fitted distributions are shown, ranked by BIC value (best fit first).</li>
                                <li>In each plot:
                                    <ul>
                                        <li>The <b>blue histogram</b> represents your actual data.</li>
                                        <li>The <b>colored curve</b> shows how well the specified distribution fits your data.</li>
                                    </ul>
                                </li>
                                <li>The plot title indicates the distribution type and its BIC value.</li>
                                <li>Your currently selected distribution is highlighted in <span style='color: green; font-weight: bold;'>green</span>.</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.shown_plot_explanation = True
                        
                        # Pre-select the first (best) distribution by default
                        default_selection = 0
                        selected_idx = st.selectbox(
                            f"Select distribution for {var}",
                            range(len(fit_results["distributions"])),
                            format_func=lambda i: get_distribution_code_string(fit_results["distributions"][i]),
                            index=default_selection
                        )
                        
                        # Store selected distribution
                        st.session_state.selected_distributions[var] = {
                            "distribution": fit_results["distributions"][selected_idx],
                            "name": fit_results["names"][selected_idx]
                        }
                        
                        # Display the selected distribution formula
                        dist = fit_results["distributions"][selected_idx]
                        formula_str = get_distribution_code_string(dist)
                        
                        st.info(f"**Selected distribution code:** {formula_str}", icon="ℹ️")
                        
                        # Plot all distributions with the selected one highlighted
                        fig = plot_distribution_fit(
                            data_df[var].values,
                            fit_results["distributions"],
                            fit_results["names"],
                            fit_results["statistics"],
                            selected_idx
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                
                # Create joint distribution from selected distributions
                if st.session_state.selected_distributions:
                    st.markdown("<h3 style='color: #424242;'>Step 5: Create Joint Distribution</h3>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;'>
                    The selected distributions for each variable will be combined into a joint distribution assuming independence between variables.
                    This joint distribution can be used for uncertainty quantification analysis in your model.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display table of selected distributions
                    st.markdown("#### Selected Distributions Summary")
                    selected_df = pd.DataFrame([
                        {
                            "Variable": var,
                            "Distribution Formula": get_distribution_code_string(info["distribution"])
                        }
                        for var, info in st.session_state.selected_distributions.items()
                    ])
                    
                    st.dataframe(selected_df, use_container_width=True)
                    
                    # Create button to generate joint distribution
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        # Add option to fit a copula
                        fit_copula = st.checkbox("Fit a Normal copula to model dependencies", value=False,
                                                help="When checked, a Normal (Gaussian) copula will be fitted to model dependencies between variables. Otherwise, variables will be assumed independent.")
                        
                        create_button = st.button(
                            "Create Joint Distribution", 
                            key="create_joint_dist",
                            type="primary"
                        )
                    
                    if create_button:
                        try:
                            # Get selected distributions in the same order as selected variables
                            selected_dists = []
                            selected_vars = []
                            for var in selected_variables:
                                if var in st.session_state.selected_distributions:
                                    selected_dists.append(st.session_state.selected_distributions[var]["distribution"])
                                    selected_vars.append(var)
                            
                            if fit_copula and len(selected_vars) > 1:
                                # Create subset of data with only selected variables
                                selected_data = data_df[selected_vars]
                                
                                # Fit a Normal copula to model dependencies
                                with st.spinner("Fitting copula to model dependencies..."):
                                    fitted_copula = fit_normal_copula(selected_data, selected_dists)
                                    
                                    # Get the correlation matrix from the copula
                                    correlation_matrix = fitted_copula.getCorrelation()
                                    
                                    # Create joint distribution with the fitted copula
                                    joint_dist = create_dependent_multivariate_distribution(selected_dists, selected_vars, fitted_copula)
                                    
                                    # Store in session state
                                    st.session_state.problem_distribution = joint_dist
                                    st.session_state.fitted_copula = fitted_copula
                                    st.session_state.correlation_matrix = correlation_matrix
                                    
                                    st.success("Joint distribution created successfully with a fitted Normal copula to model dependencies!")
                                    
                                    # Display the correlation matrix
                                    st.markdown("#### Fitted Correlation Matrix")
                                    st.markdown("""
                                    <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;'>
                                    This correlation matrix represents the dependencies between variables captured by the Normal copula.
                                    Values close to 1 indicate strong positive correlation, values close to -1 indicate strong negative correlation,
                                    and values close to 0 indicate little to no correlation.
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Create a DataFrame for the correlation matrix
                                    n = correlation_matrix.getDimension()
                                    corr_data = np.zeros((n, n))
                                    for i in range(n):
                                        for j in range(n):
                                            corr_data[i, j] = correlation_matrix[i, j]
                                    
                                    corr_df = pd.DataFrame(corr_data, index=selected_vars, columns=selected_vars)
                                    
                                    # Display as a heatmap
                                    fig = px.imshow(corr_df, 
                                                    text_auto=True, 
                                                    color_continuous_scale='RdBu_r',
                                                    range_color=[-1, 1],
                                                    labels=dict(x="Variable", y="Variable", color="Correlation"))
                                    fig.update_layout(width=600, height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Use independent copula (the original approach)
                                joint_dist = create_multivariate_distribution(selected_dists, selected_vars)
                                
                                # Store in session state
                                st.session_state.problem_distribution = joint_dist
                                
                                if fit_copula and len(selected_vars) <= 1:
                                    st.warning("Copula fitting requires at least 2 variables. Using independent model instead.")
                                else:
                                    st.success("Joint distribution created successfully with independent copula!")
                            
                            # Generate code for the joint distribution
                            st.markdown("#### Generated Code")
                            st.markdown("""
                            <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;'>
                            Copy this code into your model file to use the fitted distributions in your uncertainty quantification analysis.
                            This will replace your manually defined distributions with ones fitted to your data.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            dist_code = f"""
# Distribution fitted from data
import openturns as ot
import numpy as np

def function_of_interest(X):
    {', '.join([var.replace(' ', '_') for var in selected_vars])} = X
    Y = {' + '.join([var.replace(' ', '_') for var in selected_vars])}
    return [Y]

model = ot.PythonFunction({len(selected_vars)}, 1, function_of_interest)
# Define marginal distributions
"""
                            for i, var in enumerate(selected_vars):
                                dist_info = st.session_state.selected_distributions[var]
                                dist = dist_info["distribution"]
                                params = dist.getParameter()
                                param_str = ", ".join([f"{p}" for p in params])
                                
                                # Preserve the original variable name casing
                                var_code_name = var.replace(' ', '_')
                                
                                # Try to get the distribution name using getName() method first
                                try:
                                    dist_class = dist.getName()
                                except Exception:
                                    try:
                                        dist_class = dist.getClassName()
                                        if dist_class.endswith("Implementation"):
                                            dist_class = dist_class[:-14]  # Remove "Implementation"
                                    except Exception:
                                        dist_class = dist_info["name"]
                                
                                # If we end up with "Distribution" as the class, default to "Normal"
                                if dist_class == "Distribution":
                                    dist_class = "Normal"
                                
                                # Debug information to help diagnose issues
                                debug_info = f"# {var}: Using distribution class {dist_class}\n"
                                dist_code += debug_info
                                
                                # Generate the distribution code with the detected class
                                dist_code += f"{var_code_name} = ot.{dist_class}({param_str})\n"
                                dist_code += f"{var_code_name}.setDescription([\"{var}\"])\n\n"
                            
                            # Add code for joint distribution (with or without copula)
                            if fit_copula and len(selected_vars) > 1 and 'fitted_copula' in st.session_state:
                                # Get the correlation matrix from the fitted copula
                                correlation_matrix = st.session_state.correlation_matrix
                                n = correlation_matrix.getDimension()
                                
                                # Generate code for the correlation matrix
                                dist_code += "# Create correlation matrix for the Normal copula\n"
                                dist_code += f"R = ot.CorrelationMatrix({n})\n"
                                
                                # Add non-diagonal correlation entries
                                for i in range(n):
                                    for j in range(i+1, n):
                                        if abs(correlation_matrix[i, j]) > 0.01:  # Only include non-zero correlations
                                            dist_code += f"R[{i}, {j}] = {correlation_matrix[i, j]:.4f}  # Correlation between {selected_vars[i]} and {selected_vars[j]}\n"
                                
                                # Generate code for creating the copula and joint distribution
                                dist_code += "\n# Create Normal copula from correlation matrix\n"
                                dist_code += "copula = ot.NormalCopula(R)\n\n"
                                dist_code += "# Define joint distribution with dependencies modeled by the copula\n"
                                dist_code += "problem = ot.ComposedDistribution([\n    "
                                dist_code += ",\n    ".join([f"{var.replace(' ', '_')}" for var in selected_vars])
                                dist_code += "\n], copula)\n\n"
                            else:
                                # Generate code for independent joint distribution
                                dist_code += "# Define joint distribution (independent)\n"
                                dist_code += "problem = ot.JointDistribution([\n    "
                                dist_code += ",\n    ".join([f"{var.replace(' ', '_')}" for var in selected_vars])
                                dist_code += "\n])\n\n"
                            
                            # Add comments about using the model
                            dist_code += "# Example of how to use this model in UncertaintyCat:\n"
                            dist_code += "# 1. Copy this entire code into the 'Model Definition' section\n"
                            dist_code += "# 2. Modify the function_of_interest to represent your actual model\n"
                            dist_code += "# 3. Run the analysis using the fitted distributions\n"
                            
                            st.code(dist_code, language="python")
                            
                            # Explain how to use
                            st.markdown("#### How to Use")
                            st.markdown("""
                            1. Copy the code above into your model file.
                            2. Replace the existing distribution definitions with this code.
                            3. Run your uncertainty quantification analysis with the fitted distributions.
                            
                            This approach allows you to perform uncertainty quantification based on real data 
                            rather than assumed distributions, leading to more accurate results.
                            """)
                            
                        except Exception as e:
                            st.error(f"Error creating joint distribution: {str(e)}")

def run_distribution_fitting_analysis(*args, **kwargs):
    """
    Wrapper function for the distribution fitting page.
    This is to match the signature of other analysis functions.
    """
    distribution_fitting_page()
