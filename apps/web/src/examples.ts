export const ISHIGAMI_SOURCE = `import openturns as ot
import numpy as np

def ishigami(X):
    x1, x2, x3 = X
    return [np.sin(x1) + 7.0 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)]

model = ot.PythonFunction(3, 1, ishigami)
model.setOutputDescription(["response"])

problem = ot.JointDistribution([
    ot.Uniform(-np.pi, np.pi),
    ot.Uniform(-np.pi, np.pi),
    ot.Uniform(-np.pi, np.pi),
])
problem.setDescription(["x1", "x2", "x3"])
`;

export interface BuilderVariable {
  name: string;
  distribution: "Normal" | "Uniform";
  first: number;
  second: number;
}

export function buildSymbolicModel(
  variables: BuilderVariable[],
  formula: string,
): string {
  const names = variables
    .map((variable) => JSON.stringify(variable.name))
    .join(", ");
  const marginals = variables
    .map(
      (variable) =>
        `    ot.${variable.distribution}(${variable.first}, ${variable.second})`,
    )
    .join(",\n");
  return `import openturns as ot

model = ot.SymbolicFunction([${names}], [${JSON.stringify(formula)}])
model.setOutputDescription(["response"])

problem = ot.JointDistribution([
${marginals}
])
problem.setDescription([${names}])
`;
}
