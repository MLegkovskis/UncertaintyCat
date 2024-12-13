import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import streamlit as st  # Added import statement
import squarify
import openturns as ot
import openturns.viewer as otv
import openturns.experimental as otexp
from modules.openturns_utils import get_ot_distribution, get_ot_model

# %%
def draw_treemap_values(polynomialChaosResult, sensitivity_threshold = 0.05, verbose=False):
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
    group_labels_treshold = []
    for group in listOfActiveGroups:
        value = pcesa.getSobolIndex(group)
        if verbose:
            print("group = ", group, " : ", value)
        groups_list_threshold.append(group)
        groups_sensitivity_values.append(value)
        group_labels_treshold.append(str(group))

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
    fig = plt.figure()
    colors = [cm.tab10(i) for i in range(len(significant_indices))]
    squarify.plot(sizes=significant_indices, label=significant_labels, color=colors, alpha=0.8)

    # Set axis labels and title
    plt.axis("off")
    plt.title("%s = %.1e" % (ignored_label, ignored_indices))
    ax = plt.gca()
    ax.set_aspect("equal")
    return fig


# %%
def ComputeSparseLeastSquaresChaos(
    inputTrain, outputTrain, multivariateBasis, totalDegree, distribution, verbose=False
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
        inputTrain, outputTrain, multivariateBasis, basisSize, distribution
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


def pce_sobol(N, model, problem, model_code_str, language_model='groq', totalDegree = 4, sparse=True):
    """Plot Sobol indices from a PCE."""
    print("N")
    print(N)
    print("problem")
    print(problem)
    print("model_code_str")
    print(model_code_str)

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
        inputTrain, outputTrain, multivariateBasis, totalDegree, distribution
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
    pce_markdown = f"""
## Settings

- N = {N}
- Total degree = {totalDegree}
- Sparse PCE ? {sparse}
- Number of coefficients = {numberOfCoefficients}

## Validation

- R2 score (on independent sample): {r2ScoreSimple[0]:.4f}
- R2 LOO score (analytical): {r2ScoreLOO[0]:.4f}
"""
    response_key = 'pce_markdown'
    if response_key not in st.session_state:
        st.session_state[response_key] = pce_markdown
    else:
        pce_markdown = st.session_state[response_key]

    st.markdown(pce_markdown)

    # Plot PCE simple validation
    graph = simpleValidation.drawValidation().getGraph(0, 0)
    graph.setXTitle("Model")
    graph.setYTitle("PCE")
    graph.setTitle(f"R2={r2ScoreSimple[0] * 100:.2f}%%")
    view = otv.View(graph, figure_kw={"figsize": (4.0, 3.0)})
    fig = view.getFigure()
    fig_key = 'pce_validation_fig'
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

    # Print Sobol' indices from PCE
    sensitivityAnalysis = ot.FunctionalChaosSobolIndices(polynomialChaosResult)
    pce_sobol_markdown = sensitivityAnalysis.__repr_markdown__()
    response_key = 'pce_sobol_markdown'
    if response_key not in st.session_state:
        st.session_state[response_key] = pce_sobol_markdown
    else:
        pce_sobol_markdown = st.session_state[response_key]

    st.markdown(pce_sobol_markdown)

    # Compute Sobol' indices from PCE
    first_order = [sensitivityAnalysis.getSobolIndex(i) for i in range(dimension)]
    total_order = [sensitivityAnalysis.getSobolTotalIndex(i) for i in range(dimension)]
    input_names = distribution.getDescription()
    graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(input_names, first_order, total_order)
    view = otv.View(graph, figure_kw={"figsize": (4.0, 3.0)})
    fig = view.getFigure()
    fig_key = 'pce_sobol_fig'
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

    # Draw Treemap
    fig = draw_treemap_values(polynomialChaosResult)
    fig_key = 'pce_treemap_fig'
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

