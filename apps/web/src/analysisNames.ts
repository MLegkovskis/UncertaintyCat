// Display labels only. Applicability and numerical metadata remain catalog-owned.
const analysisNames: Record<string, string> = {
  monte_carlo: "Uncertainty Propagation",
  eda: "Exploratory Data Analysis",
  convergence: "Expectation Convergence",
  correlation: "Correlation Analysis",
  sobol: "Sobol Sensitivity Analysis",
  fast: "FAST Sensitivity Analysis",
  hsic: "HSIC Dependence Analysis",
  target_hsic: "Target-Domain HSIC Sensitivity",
  taylor: "Taylor Expansion Moments",
  reliability: "Reliability Analysis",
  ancova: "ANCOVA Dependent-Input Sensitivity",
  morris: "Morris Screening",
  pce: "Polynomial Chaos Surrogate",
  gpr: "Gaussian Process Surrogate",
  calibration_nlls: "Nonlinear Least-Squares Calibration",
};

export function analysisDisplayName(key: string) {
  return analysisNames[key] ?? key.replaceAll("_", " ");
}
