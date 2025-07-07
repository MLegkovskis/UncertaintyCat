# UncertaintyCat 🐱 | AI-Driven Uncertainty Quantification

[](https://uncertaintycat.streamlit.app/)
[](https://www.google.com/search?q=LICENSE)
[](https://www.python.org/downloads/)
[](https://www.google.com/search?q=https://openturns.github.io/openturns/latest/)
[](https://github.com/MLegkovskis/UncertaintyCat/actions/workflows/ci.yml)

**UncertaintyCat** is a web-based platform designed to make advanced **Uncertainty Quantification (UQ)** and **Sensitivity Analysis (SA)** accessible to engineers, researchers, and data scientists. Powered by **OpenTURNS** and augmented with **Generative AI**, it streamlines the entire UQ workflow from model definition to actionable insight, all within your browser.

[**🌐 Try the Live Application\!**](https://uncertaintycat.streamlit.app/)

-----

## 🎯 Table of Contents

  - [The Motivation](https://www.google.com/search?q=%23-the-motivation)
  - [Key Features](https://www.google.com/search?q=%23-key-features)
  - [How It Works](https://www.google.com/search?q=%23-how-it-works)
  - [Defining Your Model for UncertaintyCat](https://www.google.com/search?q=%23-defining-your-model-for-uncertaintycat)
  - [Available Analysis Modules](https://www.google.com/search?q=%23-available-analysis-modules)
  - [Local Installation](https://www.google.com/search?q=%23-local-installation)
  - [Contributing & Roadmap](https://www.google.com/search?q=%23-contributing--roadmap)
  - [Acknowledgments](https://www.google.com/search?q=%23-acknowledgments)
  - [License](https://www.google.com/search?q=%23-license)

-----

## 💡 The Motivation

UncertaintyCat was born from a real industrial need encountered during a PhD with **Tata Steel**. While building high-dimensional models of reheating furnaces, the value of UQ became undeniable. A single Sobol analysis could pinpoint which process inputs truly dominated output variability, leading to immediate "aha\!" moments for engineers.

The pain point was clear: every new analysis required fresh **OpenTURNS** code and a time-consuming manual interpretation report.

**The solution is UncertaintyCat**: A single, powerful web application that runs a comprehensive suite of UQ analyses with a few clicks and delivers AI-generated, engineer-ready commentary. It transforms the UQ cycle from days of manual coding and report-writing into minutes of automated computation and insight.

-----

## 🌟 Key Features

  - **🚀 Zero-Install & Cloud-Native**: Runs entirely in your browser. All computations are handled in-memory on a managed cloud workspace, and your model code is never persisted.
  - **🔧 Complete UQ & SA Toolkit**: A comprehensive suite of 11 analysis engines, covering the modern UQ canon:
      - Monte Carlo Simulation & Expectation Convergence
      - Variance-Based SA: **Sobol**, **FAST**, **ANCOVA**
      - Screening Methods: **Morris**
      - Dependence-Based SA: **HSIC** (Hilbert-Schmidt Independence Criterion)
      - Local SA: **Taylor Expansion**
      - Global Interpretation: **Shapley Values** (via machine learning surrogates)
      - Correlation Analysis & Exploratory Data Analysis
  - **🤖 Engine-by-Engine AI Insight**: Each analysis module bundles its numeric results with relevant theory and makes a targeted call to a Large Language Model (e.g., Llama-4, Gemma-2 via Groq Cloud) to generate concise, context-aware interpretations.
  - **📊 Dynamic & Interactive Visualizations**: Dashboards powered by Plotly adapt from low- to high-dimensional models, with hover tool-tips, filtering, and image export as standard features.
  - **⚙️ Data-Driven Input Modeling**:
      - **Distribution Fitting Wizard**: Upload CSV/Excel data to fit probability distributions and copulas, generating a ready-to-use `problem` definition.
      - **Dimensionality Reduction**: Use Morris screening to identify and fix non-influential variables, simplifying your model for further analysis.
  - **📚 Built-in Benchmarks**: Get started instantly with classic UQ benchmark functions like Ishigami, Borehole, and Beam Deflection, perfect for demos, teaching, and testing.

-----

## ⚙️ How It Works

UncertaintyCat simplifies the UQ workflow into a few straightforward steps:

1.  **Select a Workspace**: Choose between the main `UQ Dashboard`, `Dimensionality Reduction` for Morris screening, or `Distribution Fitting` to define inputs from data.
2.  **Provide Your Model**: In the code editor, define your model as an `openturns.Function` and your input uncertainties as an `openturns.Distribution`.
3.  **Run Analyses**: Click **"Run Full UQ Suite"** for a comprehensive report, or run any of the 11 analysis modules individually for faster, targeted insights.
4.  **Review & Interact**: Explore the interactive plots and tables. Use the sidebar **AI Chatbot**, which has full context of all your analysis results, to ask follow-up questions like *"Compare the Sobol and FAST results and explain any discrepancies."*

-----

## 🐍 Defining Your Model for UncertaintyCat

To use your own model, you must define two Python variables within the code editor: `model` and `problem`.

1.  **`model`**: An `openturns.Function` object. This can be created from a Python function or a symbolic string.
2.  **`problem`**: An `openturns.Distribution` object, typically a `JointDistribution` defining the uncertain input variables.

**Complete Example (Ishigami Function):**

You can paste this code directly into the app's editor to get started.

```python
import openturns as ot
import numpy as np

# 1. Define the model as a standard Python function
# It must take a list or array of inputs and return a list or array of outputs.
def ishigami_function(X):
    x1, x2, x3 = X[0], X[1], X[2]
    
    a = 7.0
    b = 0.1
    
    term1 = np.sin(x1)
    term2 = a * (np.sin(x2)**2)
    term3 = b * (x3**4) * np.sin(x1)
    
    return [term1 + term2 + term3]

# 2. Wrap the Python function into an OpenTURNS Function object
# The first argument is the number of inputs (3), the second is the number of outputs (1).
model = ot.PythonFunction(3, 1, ishigami_function_for_ot)

# 3. Define the input probability distributions for each variable
# The list of marginals must match the order of inputs in your function.
marginals = [
    ot.Uniform(-np.pi, np.pi), # X1
    ot.Uniform(-np.pi, np.pi), # X2
    ot.Uniform(-np.pi, np.pi)  # X3
]

# Set descriptions for clear labeling in plots and tables
marginals[0].setDescription(["X1_Uniform"])
marginals[1].setDescription(["X2_Uniform"])
marginals[2].setDescription(["X3_Uniform"])

# 4. Create the joint distribution object for the problem definition
# For Sobol analysis, inputs are treated as independent, so an IndependentCopula is used.
problem = ot.ComposedDistribution(marginals)

```

-----

## 🔬 Available Analysis Modules

| Module                             | Description                                                                                              |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Model Understanding** | AI-driven summary of the model's mathematical structure and input distributions.                           |
| **Exploratory Data Analysis** | Statistical summary and visualizations of Monte Carlo simulation results.                               |
| **Expectation Convergence** | Analyzes the convergence of the model's expected value with increasing sample size.                        |
| **Sobol Analysis** | Decomposes output variance into contributions from each input (first, total, and second-order indices).      |
| **FAST Analysis** | A variance-based sensitivity analysis method efficient for identifying first-order sensitivities.        |
| **ANCOVA Analysis** | A variance-based method that is effective for models with **correlated** input variables.                 |
| **Taylor Analysis** | A local, derivative-based sensitivity analysis around the mean of the inputs.                               |
| **Correlation Analysis** | Computes Pearson and Spearman correlation coefficients between inputs and the output.                    |
| **HSIC Analysis** | A kernel-based method to detect both linear and non-linear dependencies between inputs and output.     |
| **Shapley Analysis** | A game-theoretic approach (using a machine learning surrogate) to fairly attribute importance to inputs. |
| **Morris Screening** | An efficient method for screening and ranking the importance of a large number of input factors.            |

-----

## 🖥️ Local Installation

While the public app requires zero installation, you can run UncertaintyCat locally for development or offline use.

### Prerequisites

  - Python 3.12+
  - Git

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/MLegkovskis/UncertaintyCat.git
    cd UncertaintyCat
    ```

2.  **Create a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Key** (Optional, for AI features)
    Create a `.streamlit/secrets.toml` file and add your Groq API key:

    ```toml
    GROQ_API_KEY="your-api-key-here"
    ```

5.  **Run the App**

    ```bash
    streamlit run UncertaintyCat.py
    ```

-----

## 🗺️ Contributing & Roadmap

**UncertaintyCat is an open-source project, and contributions are highly welcome\!** Our goal is to create the most intuitive and powerful UQ tool for the engineering and scientific communities.

### Current Roadmap

We are actively looking for collaborators to help with:

  - **Equation to Model**: Paste a LaTeX equation to auto-generate the Python model code.
  - **Scalable Backends**: Integrate a Ray or GPU-based backend to push Monte Carlo and Sobol analyses to $\>10^6$ samples.
  - **One-Click Docker Bundle**: Create a simple Docker Compose file for easy deployment in private cloud environments, with optional on-device LLM weights.
  - **New Analysis Modules**: Integrate additional UQ methods and surrogate modeling techniques.

### How to Contribute

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

Please feel free to open an issue to discuss a bug or a new feature.

-----

## 🙏 Acknowledgments

  - This project is built on the robust and powerful **[OpenTURNS](https://www.google.com/search?q=https://openturns.github.io/openturns/latest/)** library. Its comprehensive, well-documented API makes this application possible.
  - The interactive user interface is created with **[Streamlit](https://streamlit.io/)**.
  - AI-powered analysis is provided by Large Language Models running on **[Groq Cloud](https://groq.com/)**.
  - You can view the **[OpenTURNS User Day 2025 Presentation](https://github.com/openturns/presentation/blob/master/userday2025/JU_OT_2025_UncertaintyCat.pdf)** for more context on the project's motivation and architecture.

-----

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.