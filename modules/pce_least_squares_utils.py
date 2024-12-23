import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
import streamlit as st  # Added import statement
import squarify
import openturns as ot
import openturns.viewer as otv
import openturns.experimental as otexp
from modules.openturns_utils import get_ot_distribution, get_ot_model
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION


def plot_pce_sobol_radial(
    polynomialChaosResult,
    sensitivity_threshold=0.01,
    max_marker_size=70.0,
    figsize=(3.5, 3.5),
    verbose=False,
):
    """
    Plot Sobol indices on a radial plot.

    The Sobol Indices Radial Plot is a polar plot where each input variable
    is placed at equal angular intervals around a circle.
    The elements of the plot are:

    - Variables: Each input variable is positioned at a specific angle on
      the circle, equally spaced from others.

    - Circles:
        - The outer circle (white fill) represents the total-order Sobol'
          index (ST) for each variable.
        - The inner circle (black fill) represents the first-order Sobol' index (S1).
        - The size of the circles is proportional to the magnitude of the respective
          Sobol' indices.

    - Lines:
        - Lines connecting variables represent second-order Sobol' indices (S2).
        - The thickness of the lines corresponds to the magnitude of the
          interaction between the two variables; thicker lines indicate
          stronger interactions.

    Authors
    -------
    - Mark Legkovskis
    - M. Baudin.

    Parameters
    ----------
    polynomialChaosResult : ot.FunctionalChaosResult
        The PCE.
    sensitivity_threshold : float, in [0, 1]
        The sensitivity_threshold on the sensitivity indices.
    max_marker_size : float, > 0
        The marker size in points.
    figsize : list(floats), >0
        The figure size.
    verbose : bool
        If True, print intermediate messages.

    Returns
    -------
    fig : Matplotlib.figure
        The plot

    References
    ----------
    - Jana Fruth, Olivier Roustant, and Sonja Kuhnt.
      Total interaction index: A variance-based sensitivity
      index for second-order interaction screening.
      Journal of Statistical Planning and Inference,
      147:212–223, 2014.
    """
    # Setup Sobol' indices
    distribution = polynomialChaosResult.getDistribution()
    dimension = distribution.getDimension()
    names = distribution.getDescription()
    pcesa = ot.FunctionalChaosSobolIndices(polynomialChaosResult)

    # Initialization
    n = len(names)
    locs = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Compute the first-order Sobol' indices
    firstSobolIndices = []
    for i in range(dimension):
        firstSobolIndices.append(pcesa.getSobolIndex(i))
    # Compute the second-order Sobol' indices
    secondSobolIndices = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i + 1, dimension):
            secondSobolIndices[i, j] = pcesa.getSobolIndex([i, j])
    # Compute the total Sobol' indices
    totalSobolIndices = []
    for i in range(dimension):
        totalSobolIndices.append(pcesa.getSobolTotalIndex(i))

    Si = {
        "S1": firstSobolIndices,
        "S2": secondSobolIndices,
        "ST": totalSobolIndices,
    }

    # Get indices
    ST = np.abs(Si["ST"])
    S1 = np.abs(Si["S1"])
    names = np.array(names)

    # Filter out insignificant indices
    significant = ST > sensitivity_threshold
    filtered_names = names[significant]
    filtered_locs = locs[significant]
    ST = ST[significant]
    S1 = S1[significant]

    # Prepare S2 matrix
    S2_matrix = np.zeros((len(filtered_names), len(filtered_names)))
    for i in range(len(filtered_names)):
        for j in range(i + 1, len(filtered_names)):
            idx_i = np.where(names == filtered_names[i])[0][0]
            idx_j = np.where(names == filtered_names[j])[0][0]
            S2_value = Si["S2"][idx_i, idx_j]
            if np.isnan(S2_value) or abs(S2_value) < sensitivity_threshold:
                S2_value = 0.0
            S2_matrix[i, j] = S2_value

    # Plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(filtered_locs)
    ax.set_xticklabels(filtered_names)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.5)

    # Plot ST and S1 using ax.scatter

    # First plot the ST circles (outer circles)
    for loc, st_val in zip(filtered_locs, ST):
        s = st_val * max_marker_size**2
        if verbose:
            print(f"loc = {loc}, st_val = {st_val}, s = {s}")
        ax.scatter(loc, 1, s=s, c="white", edgecolors="black", zorder=2)

    # Then plot the S1 circles (inner circles)
    for loc, s1_val in zip(filtered_locs, S1):
        s = s1_val * max_marker_size**2
        if verbose:
            print(f"loc = {loc}, s1_val = {s1_val}, s = {s}")
        ax.scatter(loc, 1, s=s, c="black", edgecolors="black", zorder=3)

    # Plot S2 interactions
    if verbose:
        print("S2")
        print(S2_matrix)
    for i in range(len(filtered_names)):
        for j in range(i + 1, len(filtered_names)):
            if S2_matrix[i, j] > 0:
                lw = S2_matrix[i, j] * max_marker_size
                if verbose:
                    print(f"S2_matrix[i, j] = {S2_matrix[i, j]}, lw = {lw}")
                ax.plot(
                    [filtered_locs[i], filtered_locs[j]],
                    [1, 1],
                    c="darkgray",
                    lw=lw,
                    zorder=1,
                )

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="ST",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="S1",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=15,
        ),
        Line2D([0], [0], color="darkgray", lw=3, label="S2"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Sobol Indices Radial Plot")
    plt.tight_layout()
    return fig


# %%
def gather_small_sum_values(values, labels, threshold=0.01, insignificant_label="*"):
    """
    Gather values which sum is lower than the threshold

    Sort the values in decreasing order.
    Gather the values which sum is lower than the threshold.
    The corresponding values are given the insignificant label.

    Parameters
    ----------
    values : list(float)
        A list of values in [0, 1] which sum to 1.
    labels : list(str)
        The labels.
    insignificant_label : str
        The label of the group of insignificant values.

    Returns
    -------
    significant_values : list(float)
        The list of significant values.
    gathered_labels : list(float)
        The labels.
    insignificant_sum : float, > 0
        The sum of insignificant values.
    """

    # Sort indices in decreasing order.
    values = np.array(values)
    sorting_indices = np.argsort(-values)
    sorting_indices = [int(v) for v in sorting_indices]
    sorted_sensitivity_values = values[sorting_indices]
    sorted_group_labels = [labels[i] for i in sorting_indices]

    # Get the index which cumulated sum is greater than complementary threshold
    cumulated_sum = np.cumsum(sorted_sensitivity_values)
    greater_than_threshold_indices = np.where(cumulated_sum > 1.0 - threshold)[0]
    number_of_values = len(values)
    if len(greater_than_threshold_indices) > 0:
        threshold_index = int(greater_than_threshold_indices[0])
    else:
        threshold_index = number_of_values

    # Gather and sum indices which sum below the threshold
    significant_values = []
    gathered_labels = []
    insignificant_sum = 0.0
    for i in range(number_of_values):
        if i > threshold_index:
            insignificant_sum += sorted_sensitivity_values[i]
        else:
            significant_values.append(sorted_sensitivity_values[i])
            gathered_labels.append(sorted_group_labels[i])

    # Add a star label to represent ignored indices
    if insignificant_sum > 0.0:
        significant_values.append(insignificant_sum)
        gathered_labels.append(insignificant_label)
    return significant_values, gathered_labels, insignificant_sum


def gather_small_values(values, labels, threshold=0.01, insignificant_label="*"):
    """
    Gather small values in a list

    Gather the values which are lower than the threshold.
    The corresponding values are given the insignificant label.

    Parameters
    ----------
    values : list(float)
        A list of values in [0, 1] which sum to 1.
    labels : list(str)
        The labels.
    threshold : float
        The treshold.
    insignificant_label : str
        The label of the group of insignificant values.

    Returns
    -------
    significant_values : list(float)
        The list of significant values.
    gathered_labels : list(float)
        The labels.
    insignificant_sum : float, > 0
        The sum of insignificant values.
    """
    # Gather and sum indices below the threshold
    significant_values = []
    gathered_labels = []
    insignificant_sum = 0.0
    for i in range(len(values)):
        if values[i] > threshold:
            significant_values.append(values[i])
            gathered_labels.append(labels[i])
        else:
            insignificant_sum += values[i]

    # Add a star label to represent insignificant indices
    if insignificant_sum > 0.0:
        significant_values.append(insignificant_sum)
        gathered_labels.append(insignificant_label)
    return significant_values, gathered_labels, insignificant_sum


# %%
def plot_pce_sobol(polynomialChaosResult, rotation=45, figsize=(3.5, 3.5)):
    """
    Plot Sobol' indices from a PCE

    Authors
    -------
    - Mark Legkovskis
    - M. Baudin.

    Parameters
    ----------
    polynomialChaosResult : ot.FunctionalChaosResult
        The PCE.
    rotation : float, in [0, 360]
        The rotation of the labels
    figsize : list(floats), >0
        The figure size.
    verbose : bool
        If True, print intermediate messages.

    Returns
    -------
    fig : Matplotlib.figure
        The plot
    """
    pcesa = ot.FunctionalChaosSobolIndices(polynomialChaosResult)
    distribution = polynomialChaosResult.getDistribution()
    dimension = distribution.getDimension()
    names = distribution.getDescription()
    inputSample = polynomialChaosResult.getInputSample()
    N = inputSample.getSize()
    # Compute the first-order Sobol' indices
    firstSobolIndices = []
    for i in range(dimension):
        firstSobolIndices.append(pcesa.getSobolIndex(i))
    # Compute the total Sobol' indices
    totalSobolIndices = []
    for i in range(dimension):
        totalSobolIndices.append(pcesa.getSobolTotalIndex(i))

    # Plot
    ind = np.arange(dimension)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(ind, firstSobolIndices, "o")
    plt.plot(ind, totalSobolIndices, "s")
    plt.xticks(ind, names, rotation=rotation)
    plt.title(f"Sobol Sensitivity Indices (N = {N})")
    plt.ylabel("Sensitivity Index")
    plt.xlabel("Input Variables")
    ax.legend(["First-order", "Total order"])
    return fig


def compute_pce_treemap_report(
    sensitivity_threshold, gathered_indices, gathered_labels
):
    # Prepare the data required for interpretation of the treemap
    # Create a Markdown report of the treemap data
    treemap_table = f"""
| Group | Sobol interaction |
|---|---|
"""
    for i in range(len(gathered_indices)):
        treemap_table += f"| {gathered_labels[i]} | {gathered_indices[i]:.4f} |\n"

    # Description of the treemap plot with numerical data
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
    gather_on_value=False,
    insignificant_label="*",
    figsize=(3.5, 3.5),
    verbose=False,
):
    """
    Plot the Sobol' interaction sensitivity indices of a PCE in a Treemap

    Parameters
    ----------
    polynomialChaosResult : ot.FunctionalChaosResult
        The PCE.
    sensitivity_threshold : float, in [0, 1]
        The threshold on the sensitivity indices.
    print_indices : bool
        If True, print the indices in the Treemap.
        Otherwise, print the variable names.
    gather_on_value : bool
        If True, consider insignificant each Sobol' index which value is below
        the threshold.
        Otherwise, gather the Sobol' indices into a set which sum is below the
        threshold.
    insignificant_label : str
        The label of insignificant Sobol' indices.
    figsize : list(floats), >0
        The figure size.
    verbose : bool
        If True, print intermediate messages.

    Returns
    -------
    fig : Matplotlib.figure
        The plot
    """
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
        group_labels = []
        for group in listOfActiveGroups:
            group_labels.append(str(group))
    else:
        variableNamesList = distribution.getDescription()
        group_labels = ComputeGroupLabelsFromLabelNames(
            variableNamesList, groups_list_threshold
        )

    if verbose:
        print_indices_and_tuples(groups_sensitivity_values, groups_list_threshold)

    # Gather and sum indices below a threshold
    if gather_on_value:
        (
            gathered_indices,
            gathered_labels,
            insignificant_sum,
        ) = gather_small_values(
            groups_sensitivity_values,
            group_labels,
            sensitivity_threshold,
            insignificant_label=insignificant_label,
        )
    else:
        (
            gathered_indices,
            gathered_labels,
            insignificant_sum,
        ) = gather_small_sum_values(
            groups_sensitivity_values,
            group_labels,
            sensitivity_threshold,
            insignificant_label=insignificant_label,
        )

    # Plot the tree map using the squarify library
    fig = plt.figure(figsize=figsize)
    number_of_significant_values = len(gathered_indices)
    colormap = mpl.colormaps["viridis"].resampled(number_of_significant_values)
    colors = [colormap(i) for i in range(len(gathered_indices))]
    squarify.plot(
        sizes=gathered_indices, label=gathered_labels, color=colors, alpha=0.8
    )

    # Set axis labels and title
    plt.axis("off")
    plt.title("%s = %.1e" % (insignificant_label, insignificant_sum))
    ax = plt.gca()
    ax.set_aspect("equal")

    return fig, gathered_indices, gathered_labels


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
    use_model_selection=True,
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
    use_model_selection : bool
        Set to True to create a sparse polynomial chaos expansion.
        Set to False to create a full polynomial chaos expansion.

    Returns
    -------
    chaosResult : ot.FunctionalChaosResult
        The polynomial chaos result.

    """
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
    train_sample_size,
    validation_sample_size,
    model,
    problem,
    model_code_str,
    language_model,
    basis_size_factor=0.5,
    use_model_selection=False,
    sensitivity_threshold=0.05,
    verbose=False,
):
    """Plot Sobol indices from a PCE.

    The list of updated figures and markdown reports is:

    'pce_validation_fig', 'pce_sobol_fig', 'pce_sobol_radial_fig', 'pce_treemap_fig', 'pce_sobol_markdown',
    "pce_sobol_response_markdown".

    Update `reset_pce_least_squares_results()` accordingly if required.

    Parameters
    ----------
    train_sample_size: int
        The size of the training sample.
    validation_sample_size: int
        The size of the validation sample.
    model:
        The physical model
    problem:
        The probabilistic model
    model_code_str: str
        The code of the physical model
    language_model: str
        The interpretation language model
    basis_size_factor: float, in [0, 1]
        The number of coefficients in the PCE, as a fraction of the training sample size.
    use_model_selection: bool
        If True, then uses model selection to select the best possible coefficients.
        If False, then uses model selection to select the best possible coefficients.
    sensitivity_threshold : float, in [0, 1]
        The sensitivity_threshold on the sensitivity indices.
        Sensitivity indices lower that this treshold are considered as insignificant.
    """

    # Get the input distribution and the model
    distribution = get_ot_distribution(problem)
    model_g = get_ot_model(model, problem)

    # Evaluate model for training
    inputTrain = distribution.getSample(train_sample_size)
    outputTrain = model_g(inputTrain)

    # Create PCE
    dimension = distribution.getDimension()
    marginalList = [distribution.getMarginal(i) for i in range(dimension)]
    multivariateBasis = ot.OrthogonalProductPolynomialFactory(marginalList)
    basisSize = int(basis_size_factor * train_sample_size)

    polynomialChaosResult = ComputeSparseLeastSquaresBasisSize(
        inputTrain,
        outputTrain,
        multivariateBasis,
        basisSize,
        distribution,
        use_model_selection,
    )

    # Evaluate model for testing
    inputTest = distribution.getSample(validation_sample_size)
    model_g = get_ot_model(model, problem)
    outputTest = model_g(inputTest)

    # Test the PCE
    metamodel = polynomialChaosResult.getMetaModel()
    # 1. Basic split
    metamodelPredictions = metamodel(inputTest)
    simpleValidation = ot.MetaModelValidation(outputTest, metamodelPredictions)
    r2ScoreSimple = simpleValidation.computeR2Score()
    # 2. Analytical LOO validation
    splitterLOO = ot.LeaveOneOutSplitter(train_sample_size)
    ot.ResourceMap.SetAsBool("FunctionalChaosValidation-ModelSelection", True)
    validation = otexp.FunctionalChaosValidation(polynomialChaosResult, splitterLOO)
    r2ScoreLOO = validation.computeR2Score()

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

    # Plot Sobol' indices from PCE using first-order and total order indices
    fig = plot_pce_sobol(polynomialChaosResult, figsize=(4.0, 3.0))
    fig_key = "pce_sobol_fig"
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

    # Plot Sobol' indices from PCE using radial plot
    fig = plot_pce_sobol_radial(polynomialChaosResult, figsize=(3.5, 3.5))
    fig_key = "pce_sobol_radial_fig"
    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.pyplot(fig, use_container_width=False)

    # Draw Treemap
    fig, gathered_indices, gathered_labels = draw_treemap_values(
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
        sensitivity_threshold, gathered_indices, gathered_labels
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


def reset_pce_least_squares_results():
    """Resets the analysis results in session state."""
    keys_to_reset = [
        "pce_validation_fig",
        "pce_sobol_fig",
        "pce_treemap_fig",
        "pce_sobol_markdown",
        "pce_sobol_response_markdown",
        "pce_sobol_radial_fig",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
