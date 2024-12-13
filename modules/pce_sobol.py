import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import streamlit as st  # Added import statement
import squarify
import openturns as ot
import openturns.viewer as otv
import openturns.experimental as otexp
from modules.openturns_utils import get_ot_distribution, get_ot_model
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION


def compute_pce_treemap_report(
    sensitivity_threshold, significant_indices, significant_labels
):
    # Prepare the data required for interpretation of the treemap
    # Create a Markdown report of the treemap data
    treemap_table = f"""
| Group | Sobol interaction |
|---|---|
"""
    for i in range(len(significant_indices)):
        treemap_table += f"| {significant_labels[i]} | {significant_indices[i]:.4f} |\n"

    # Description of the radial plot with numerical data
    treemap_plot_description = f"""
    ## Sobol' interaction indices with a Treemap plot

    The Sobol Indices Treemap Plot is a unit square where each rectangle
    represents the interaction Sobol' index of a group of variables.
    By definition, the sum of interaction Sobol' indices is equal to 1.
    This is why all the interaction Sobol' indices are represented
    within a unit square.

    The elements of the plot are:

    - **Variables**: Each group of variables is represented
      within square brackets.

    - **Rectangles**: The area of each rectangle is equal to the 
      interaction indice of the corresponding group of variables.

    - **Threshold**: Not all groups of variables are represented.
      Only the interaction Sobol' indices greater than {sensitivity_threshold}
      are presented in the plot.

    Numerical data for the plot:

    {treemap_table}

    This plot has several advantages.

    - We can see interaction of variables, which is not easy to see
      with other plots.
    - Insignificant groups of variables are not represented, because
      of the threshold.
      More importantly, if these groups of variables were represented as rectangles
      their area would be so small that then could not be visible.
    - Significant groups of variables occupy a visual space which
      is larger, because it is proportional to the interaction Sobol' index.
    - Computing all the Sobol' interaction indices can be costly with
      sampling methods.
      It can be very efficient, however, if a polynomial chaos expansion (PCE) is 
      used because evaluating all Sobol' interaction indices of the required 
      groups of variables is instantaneous.
      This method requires that a satisfactory PCE metamodel is 
      obtained first.
      Hence, validating the metamodel must be done first.
      If the PCE metamodel does not have e.g. a R2 larger than, say, 0.9,
      then the analysis from the PCE should not be pushed further.

    This plot has several drawbacks.

    - When the model has a large number of significant interactions,
      then the plot is made of a large number of rectangles.
      This correctly reflects the complexity of the model, but can
      be inconvenient sometimes.
    - Plotting the rectangles so that they fit within the unit square
      while being not too thin is an integer combinatorial problem which
      is NP-hard to solve.
      Hence, solving the problem may require meta-heuristics to keep the
      CPU time to an acceptable level.
      The downside of meta-heuristics is that the rectangles can be relatively
      thin in some cases, which is not visually appealing.
    """
    return treemap_plot_description


def ComputeGroupLabelsFromLabelNames(variableNamesList, groupsList):
    """
    Compute the labels of groups from names

    Parameters
    ----------
    variableNamesList : list(str)
        The list of variables names.

    groupsList: list(list(int))
        Each item in the list is a list of indices of variables
        representing a group.
        The number of items in the list is the number of
        groups.

    Returns
    -------
    group_labels : list(str)
        The labels of the groups of variables.

    Example
    -------
    >>> variableNamesList = ["A", "B", "C"]
    >>> groupsList = [[0], [1], [2], [0, 1], [0, 2]]
    >>> group_labels = ComputeGroupLabelsFromLabelNames(variableNamesList, groupsList)
    >>> group_labels
    ["A", "B", "C", "AB", "AC"]
    """
    number_of_keys = len(variableNamesList)
    group_labels = []
    for i in range(len(groupsList)):
        group = groupsList[i]
        for j in group:
            if j < 0 or j >= number_of_keys:
                raise ValueError(
                    f"index = {j} inconsistent with the number of keys {number_of_keys}"
                )
        label_list = [variableNamesList[j] for j in group]
        label = "[" + ",".join(label_list) + "]"
        group_labels.append(label)
    return group_labels


# %%
def draw_treemap_values(
    polynomialChaosResult,
    sensitivity_threshold=0.05,
    print_indices=False,
    verbose=False,
):
    pcesa = ot.FunctionalChaosSobolIndices(polynomialChaosResult)

    # Compute all actived groups of variables involved in the PCE decomposition
    indices = polynomialChaosResult.getIndices()
    distribution = polynomialChaosResult.getDistribution()
    dimension = distribution.getDimension()
    basis = polynomialChaosResult.getOrthogonalBasis()
    enumerateFunction = basis.getEnumerateFunction()
    listOfActiveGroups = []
    for i in indices:
        multiindex = enumerateFunction(i)
        group = []
        for j in range(dimension):
            if multiindex[j] > 0:
                group.append(j)
        if verbose:
            print(f"i = {i}, multiindex = {multiindex}, group = {group}")
        if group not in listOfActiveGroups and i > 0:
            listOfActiveGroups.append(group)

    if verbose:
        print(f"List of active groups = {listOfActiveGroups}")

    # Compute the interaction Sobol' indices of active groups
    groups_sensitivity_values = []
    groups_list_threshold = []
    for group in listOfActiveGroups:
        value = pcesa.getSobolIndex(group)
        if verbose:
            print("group = ", group, " : ", value)
        groups_list_threshold.append(group)
        groups_sensitivity_values.append(value)

    # Compute labels to print in the Treemap
    if print_indices:
        # Compute the label from the index
        group_labels_treshold = []
        for group in listOfActiveGroups:
            group_labels_treshold.append(str(group))
    else:
        variableNamesList = distribution.getDescription()
        group_labels_treshold = ComputeGroupLabelsFromLabelNames(
            variableNamesList, groups_list_threshold
        )

    if verbose:
        print_indices_and_tuples(groups_sensitivity_values, groups_list_threshold)

    # Gather and sum indices below a threshold
    significant_indices = []
    significant_labels = []
    ignored_indices = 0.0
    for i in range(len(groups_sensitivity_values)):
        if groups_sensitivity_values[i] > sensitivity_threshold:
            significant_indices.append(groups_sensitivity_values[i])
            significant_labels.append(group_labels_treshold[i])
        else:
            ignored_indices += groups_sensitivity_values[i]

    # Add a star label to represent ignored indices
    ignored_label = "*"
    if ignored_indices > 0.0:
        significant_indices.append(ignored_indices)
        significant_labels.append("*")

    # Plot the tree map using the squarify library
    fig = plt.figure(figsize=(3.5, 3.5))
    colors = [cm.tab10(i) for i in range(len(significant_indices))]
    squarify.plot(
        sizes=significant_indices, label=significant_labels, color=colors, alpha=0.8
    )

    # Set axis labels and title
    plt.axis("off")
    plt.title("%s = %.1e" % (ignored_label, ignored_indices))
    ax = plt.gca()
    ax.set_aspect("equal")

    return fig, significant_indices, significant_labels


# %%
def ComputeSparseLeastSquaresChaos(
    inputTrain,
    outputTrain,
    multivariateBasis,
    totalDegree,
    distribution,
    sparse=True,
    verbose=False,
):
    """
    Compute a sparse polynomial chaos using least squares.

    * Uses the enumerate rule in basis.
    * Uses the LeastSquaresStrategy to compute the coefficients based on
      least squares.
    * Uses LeastSquaresMetaModelSelectionFactory to use the LARS selection method.
    * Uses FixedStrategy in order to keep all the coefficients that the
      LARS method selected.

    Parameters
    ----------
    inputTrain : ot.Sample(size, input_dimension)
        The input training sample.
    outputTrain : ot.Sample(size, output_dimension)
        The output training sample.
    multivariateBasis : ot.Basis
        The multivariate basis.
    totalDegree : int
        The total polynomial degree.
    distribution : ot.Distribution(input_dimension)
        The input distribution.

    Returns
    -------
    chaosResult : ot.FunctionalChaosResult
        The polynomial chaos result.

    """
    enumerateFunction = multivariateBasis.getEnumerateFunction()
    basisSize = enumerateFunction.getBasisSizeFromTotalDegree(totalDegree)
    if verbose:
        print("basisSize = ", basisSize)
    chaosResult = ComputeSparseLeastSquaresBasisSize(
        inputTrain, outputTrain, multivariateBasis, basisSize, distribution, sparse
    )
    return chaosResult


# %%
def ComputeSparseLeastSquaresBasisSize(
    inputTrain,
    outputTrain,
    multivariateBasis,
    basisSize,
    distribution,
    sparse=True,
):
    """
    Create a sparse polynomial chaos using least squares.

    * Uses the enumerate rule in basis.
    * Uses the LeastSquaresStrategy to compute the coefficients based on
      least squares.
    * Uses LeastSquaresMetaModelSelectionFactory to use the LARS selection method.
    * Uses FixedStrategy in order to keep all the coefficients that the
      LARS method selected.

    Parameters
    ----------
    inputTrain : ot.Sample(size, input_dimension)
        The input training sample.
    outputTrain : ot.Sample(size, output_dimension)
        The output training sample.
    multivariateBasis : ot.Basis
        The multivariate basis.
    basisSize : int
        The maximum number of coefficients in the basis.
    distribution : ot.Distribution(input_dimension)
        The input distribution.
    sparse : bool
        Set to True to create a sparse polynomial chaos expansion.
        Set to False to create a full polynomial chaos expansion.

    Returns
    -------
    chaosResult : ot.FunctionalChaosResult
        The polynomial chaos result.

    """
    if sparse:
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
    chaosResult = chaosAlgorithm.getResult()
    return chaosResult


# %%
def print_indices_and_tuples(groups_sensitivity_values, groups_list):
    """
    Print the sensitivity indices and the groups.

    Parameters
    ----------
    groups_sensitivity_values : list(float)
        The sensitivity indices.
    groups_list : list(list(int))
        The list of groups of variables.

    Returns
    -------
    None.

    """
    if len(groups_list) != len(groups_sensitivity_values):
        raise ValueError(
            "The indices values have length %d, but the tuples have length %d"
            % (len(groups_sensitivity_values), len(groups_list))
        )
    print("+ Tuples and values :")
    for i in range(len(groups_sensitivity_values)):
        print(groups_list[i], ":", groups_sensitivity_values[i])
    return


def pce_sobol(
    N,
    model,
    problem,
    model_code_str,
    language_model="groq",
    totalDegree=4,
    sparse=False,
    sensitivity_threshold=0.05,
    verbose=False,
):
    """Plot Sobol indices from a PCE.

    The list of updated figures and markdown reports is:

    'pce_validation_fig', 'pce_sobol_fig', 'pce_treemap_fig', 'pce_sobol_markdown',
    "pce_sobol_response_markdown".

    Update `reset_analysis_results()` accordingly if required.
    """

    # Get the input distribution and the model
    distribution = get_ot_distribution(problem)
    model_g = get_ot_model(model, problem)

    # Evaluate model for training
    inputTrain = distribution.getSample(N)
    outputTrain = model_g(inputTrain)

    # Create PCE
    dimension = distribution.getDimension()
    marginalList = [distribution.getMarginal(i) for i in range(dimension)]
    multivariateBasis = ot.OrthogonalProductPolynomialFactory(marginalList)
    polynomialChaosResult = ComputeSparseLeastSquaresChaos(
        inputTrain, outputTrain, multivariateBasis, totalDegree, distribution, sparse
    )

    # Evaluate model for testing
    inputTest = distribution.getSample(N)
    model_g = get_ot_model(model, problem)
    outputTest = model_g(inputTest)

    # Test the PCE
    metamodel = polynomialChaosResult.getMetaModel()
    # 1. Basic split
    metamodelPredictions = metamodel(inputTest)
    simpleValidation = ot.MetaModelValidation(outputTest, metamodelPredictions)
    r2ScoreSimple = simpleValidation.computeR2Score()
    # 2. Analytical LOO validation
    splitterLOO = ot.LeaveOneOutSplitter(N)
    ot.ResourceMap.SetAsBool("FunctionalChaosValidation-ModelSelection", True)
    validation = otexp.FunctionalChaosValidation(polynomialChaosResult, splitterLOO)
    r2ScoreLOO = validation.computeR2Score()

    # Print parameters and validation results
    numberOfCoefficients = polynomialChaosResult.getIndices().getSize()
    pce_result_markdown = f"""
## Settings

- N = {N}
- Total degree = {totalDegree}
- Sparse PCE? {sparse}
- Number of coefficients = {numberOfCoefficients}

## Validation

- R2 score (on independent sample): {r2ScoreSimple[0]:.4f}
- R2 LOO score (analytical): {r2ScoreLOO[0]:.4f}
"""

    # Plot PCE simple validation
    graph = simpleValidation.drawValidation().getGraph(0, 0)
    graph.setXTitle("Model")
    graph.setYTitle("PCE")
    graph.setTitle(r"$R^2$" f"={r2ScoreSimple[0] * 100:.2f}%")
    view = otv.View(graph, figure_kw={"figsize": (3.0, 2.0)})
    fig = view.getFigure()
    fig_key = "pce_validation_fig"
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

    # Print Sobol' indices from PCE
    sensitivityAnalysis = ot.FunctionalChaosSobolIndices(polynomialChaosResult)
    pce_sobol_markdown = sensitivityAnalysis.__repr_markdown__()

    # Create a technical report of this analysis
    pce_sobol_report_markdown = f"""
{pce_result_markdown}

## Sobol' indices (based on PCE)

{pce_sobol_markdown}
"""
    pce_sobol_report_key = "pce_sobol_markdown"
    if pce_sobol_report_key not in st.session_state:
        st.session_state[pce_sobol_report_key] = pce_sobol_report_markdown
    else:
        pce_sobol_report_markdown = st.session_state[pce_sobol_report_key]

    if verbose:
        print("pce_sobol_report_markdown")
        print(pce_sobol_report_markdown)

    st.markdown(pce_sobol_report_markdown)

    # Compute Sobol' indices from PCE
    first_order = [sensitivityAnalysis.getSobolIndex(i) for i in range(dimension)]
    total_order = [sensitivityAnalysis.getSobolTotalIndex(i) for i in range(dimension)]
    input_names = distribution.getDescription()
    graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(
        input_names, first_order, total_order
    )
    graph.setLegendPosition("upper left")
    graph.setLegendCorner([1.0, 1.0])
    view = otv.View(graph, figure_kw={"figsize": (5.0, 2.5)})
    fig = view.getFigure()
    fig_key = "pce_sobol_fig"
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

    # Draw Treemap
    fig, significant_indices, significant_labels = draw_treemap_values(
        polynomialChaosResult, sensitivity_threshold, verbose=verbose
    )
    fig_key = "pce_treemap_fig"
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

    # Compute the PCE Treemap report
    treemap_plot_description = compute_pce_treemap_report(
        sensitivity_threshold, significant_indices, significant_labels
    )

    # Format the model code for inclusion in the prompt
    model_code_formatted = "\n".join(
        ["    " + line for line in model_code_str.strip().split("\n")]
    )

    # Prepare the inputs description
    input_parameters = []
    for name, dist_info in zip(problem["names"], problem["distributions"]):
        input_parameters.append(
            f"- **{name}**: {dist_info['type']} distribution with parameters {dist_info['params']}"
        )

    inputs_description = "\n".join(input_parameters)

    # Prepare the prompt
    prompt = f"""
{RETURN_INSTRUCTION}

Given the following user-defined model defined in Python code:

```python
{model_code_formatted}
```

and the following uncertain input distributions:

{inputs_description}

Given the following PCE settings and validation results:

{pce_result_markdown}

Given the following Sobol' indices:

{pce_sobol_markdown}

Given the interaction Sobol Indices Treemap data:

{treemap_plot_description}

Please:
  - Use the title "Interpretation of the results".
  - Display all the index values as separate tables (if the tables are big - 
    feel free to show only top 10 ranked inputs).
  - First check if the metamodel is satisfactory: is the R2 score greater than 0.9?
    Is the R2 score computed from simple validation is consistent with the 
    Leave-One-Out R2 score?
  - Reference the Sobol indices tables in your discussion.
  - Briefly explain the Sobol method and the difference between first-order and
    total-order indices in terms of their mathematics and what they represent.
  - Explain the significance of high-impact Sobol' indices and the importance of
    the corresponding input variables from both mathematical and physical perspectives.
  - Explain the interaction indices based on the Treemap and emphasize the most
    significant interactions.
  - Which variables have a total Sobol' index close to 0 (e.g. smaller than 0.01)?
    These variables are insignificant: they could be set to zero
    without changing the variance of the output of the model.
  - Which variables have a first-order Sobol' index close to 1 (e.g. larger than 0.5)?
    These variables are very influential.
  - Which variables have a total Sobol' index significantly larger than the 
    first-order Sobol' index?
    These variables may have interaction with others.
  - Which groups of variables which Sobol' total interaction indices are close
    to zero (e.g. lower than 0.01)?
    These groups of variables could be set to a constant value without
    changing the variance of the output much.
"""
    if verbose:
        print("prompt")
        print(prompt)

    response_key = "pce_sobol_response_markdown"
    if response_key not in st.session_state:
        response_markdown = call_groq_api(prompt, model_name=language_model)
        st.session_state[response_key] = response_markdown
    else:
        response_markdown = st.session_state[response_key]

    if verbose:
        print("response_markdown")
        print(response_markdown)

    st.markdown(response_markdown)