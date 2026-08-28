export type DistributionName =
  | "Normal"
  | "Uniform"
  | "LogNormal"
  | "Exponential"
  | "Gamma"
  | "Beta"
  | "Triangular";

interface DistributionParameter {
  key: string;
  label: string;
  defaultValue: number;
  finite?: boolean;
}

export interface DistributionDefinition {
  key: DistributionName;
  label: string;
  parameters: readonly DistributionParameter[];
  validate: (values: readonly number[]) => string | null;
  constructor: (values: readonly number[]) => string;
}

const finite = (values: readonly number[]) =>
  values.every(Number.isFinite) ? null : "All distribution parameters must be finite.";
const positive = (value: number, label: string) =>
  value > 0 ? null : `${label} must be greater than zero.`;
const ordered = (lower: number, upper: number) =>
  upper > lower ? null : "Upper bound must be greater than lower bound.";

export const DISTRIBUTION_REGISTRY: readonly DistributionDefinition[] = [
  {
    key: "Normal",
    label: "Normal",
    parameters: [
      { key: "mean", label: "Mean", defaultValue: 0 },
      { key: "standardDeviation", label: "Std dev", defaultValue: 1 },
    ],
    validate: (values) => finite(values) ?? positive(values[1] ?? 0, "Standard deviation"),
    constructor: ([mean, standardDeviation]) => `ot.Normal(${mean}, ${standardDeviation})`,
  },
  {
    key: "Uniform",
    label: "Uniform",
    parameters: [
      { key: "lower", label: "Lower", defaultValue: -1 },
      { key: "upper", label: "Upper", defaultValue: 1 },
    ],
    validate: (values) => finite(values) ?? ordered(values[0] ?? 0, values[1] ?? 0),
    constructor: ([lower, upper]) => `ot.Uniform(${lower}, ${upper})`,
  },
  {
    key: "LogNormal",
    label: "Log-normal",
    parameters: [
      { key: "logMean", label: "Log mean", defaultValue: 0 },
      { key: "logStandardDeviation", label: "Log std dev", defaultValue: 1 },
      { key: "shift", label: "Shift", defaultValue: 0 },
    ],
    validate: (values) => finite(values) ?? positive(values[1] ?? 0, "Log standard deviation"),
    constructor: ([logMean, logStandardDeviation, shift]) =>
      `ot.LogNormal(${logMean}, ${logStandardDeviation}, ${shift})`,
  },
  {
    key: "Exponential",
    label: "Exponential",
    parameters: [
      { key: "rate", label: "Rate", defaultValue: 1 },
      { key: "shift", label: "Shift", defaultValue: 0 },
    ],
    validate: (values) => finite(values) ?? positive(values[0] ?? 0, "Rate"),
    constructor: ([rate, shift]) => `ot.Exponential(${rate}, ${shift})`,
  },
  {
    key: "Gamma",
    label: "Gamma",
    parameters: [
      { key: "shape", label: "Shape", defaultValue: 2 },
      { key: "rate", label: "Rate", defaultValue: 1 },
      { key: "shift", label: "Shift", defaultValue: 0 },
    ],
    validate: (values) =>
      finite(values) ?? positive(values[0] ?? 0, "Shape") ?? positive(values[1] ?? 0, "Rate"),
    constructor: ([shape, rate, shift]) => `ot.Gamma(${shape}, ${rate}, ${shift})`,
  },
  {
    key: "Beta",
    label: "Beta",
    parameters: [
      { key: "alpha", label: "Alpha", defaultValue: 2 },
      { key: "beta", label: "Beta", defaultValue: 2 },
      { key: "lower", label: "Lower", defaultValue: 0 },
      { key: "upper", label: "Upper", defaultValue: 1 },
    ],
    validate: (values) =>
      finite(values) ??
      positive(values[0] ?? 0, "Alpha") ??
      positive(values[1] ?? 0, "Beta") ??
      ordered(values[2] ?? 0, values[3] ?? 0),
    constructor: ([alpha, beta, lower, upper]) =>
      `ot.Beta(${alpha}, ${beta}, ${lower}, ${upper})`,
  },
  {
    key: "Triangular",
    label: "Triangular",
    parameters: [
      { key: "lower", label: "Lower", defaultValue: -1 },
      { key: "mode", label: "Mode", defaultValue: 0 },
      { key: "upper", label: "Upper", defaultValue: 1 },
    ],
    validate: (values) => {
      const invalid = finite(values) ?? ordered(values[0] ?? 0, values[2] ?? 0);
      if (invalid) return invalid;
      return (values[1] ?? 0) >= (values[0] ?? 0) &&
        (values[1] ?? 0) <= (values[2] ?? 0)
        ? null
        : "Mode must be within the lower and upper bounds.";
    },
    constructor: ([lower, mode, upper]) => `ot.Triangular(${lower}, ${mode}, ${upper})`,
  },
] as const;

export interface BuilderVariable {
  id: string;
  name: string;
  distribution: DistributionName;
  parameters: number[];
}

interface BuilderOutput {
  id: string;
  name: string;
  formula: string;
}

export interface BuilderSpec {
  variables: BuilderVariable[];
  outputs: BuilderOutput[];
  copula: {
    kind: "independent" | "normal";
    correlation: number[][];
  };
}

export function distributionDefinition(key: DistributionName) {
  return DISTRIBUTION_REGISTRY.find((item) => item.key === key)!;
}

export function createBuilderVariable(index: number): BuilderVariable {
  const definition = distributionDefinition("Normal");
  return {
    id: crypto.randomUUID(),
    name: `x${index + 1}`,
    distribution: "Normal",
    parameters: definition.parameters.map((parameter) => parameter.defaultValue),
  };
}

export function identityCorrelation(size: number): number[][] {
  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_unused, column) => (row === column ? 1 : 0)),
  );
}

export function validateCorrelation(matrix: number[][]): string | null {
  const size = matrix.length;
  if (!size || matrix.some((row) => row.length !== size)) return "Correlation matrix must be square.";
  const lower = Array.from({ length: size }, () => Array<number>(size).fill(0));
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      const value = matrix[row]?.[column];
      const mirror = matrix[column]?.[row];
      if (!Number.isFinite(value) || value! < -1 || value! > 1) return "Correlations must be finite values from -1 to 1.";
      if (Math.abs(value! - mirror!) > 1e-10) return "Correlation matrix must be symmetric.";
      if (row === column && Math.abs(value! - 1) > 1e-10) return "Correlation diagonal must equal 1.";
      let subtotal = value!;
      for (let inner = 0; inner < column; inner += 1) subtotal -= lower[row]![inner]! * lower[column]![inner]!;
      if (row === column) {
        if (subtotal <= 1e-12) return "Correlation matrix must be positive definite.";
        lower[row]![column] = Math.sqrt(subtotal);
      } else {
        lower[row]![column] = subtotal / lower[column]![column]!;
      }
    }
  }
  return null;
}

export function validateBuilder(spec: BuilderSpec): string[] {
  const errors: string[] = [];
  if (!spec.variables.length) errors.push("Add at least one input.");
  if (!spec.outputs.length) errors.push("Add at least one output.");
  const names = spec.variables.map((variable) => variable.name.trim());
  if (names.some((name) => !/^[A-Za-z_]\w*$/.test(name))) errors.push("Input names must be valid identifiers.");
  if (new Set(names).size !== names.length) errors.push("Input names must be unique.");
  for (const variable of spec.variables) {
    const issue = distributionDefinition(variable.distribution).validate(variable.parameters);
    if (issue) errors.push(`${variable.name || "Input"}: ${issue}`);
  }
  if (spec.outputs.some((output) => !output.name.trim() || !output.formula.trim())) errors.push("Every output needs a name and formula.");
  if (spec.copula.kind === "normal") {
    const issue = validateCorrelation(spec.copula.correlation);
    if (issue) errors.push(issue);
  }
  return errors;
}

export function buildSymbolicModel(spec: BuilderSpec): string {
  const errors = validateBuilder(spec);
  if (errors.length) throw new Error(errors[0]);
  const names = spec.variables.map((variable) => JSON.stringify(variable.name)).join(", ");
  const outputs = spec.outputs.map((output) => JSON.stringify(output.formula)).join(", ");
  const outputNames = spec.outputs.map((output) => JSON.stringify(output.name)).join(", ");
  const marginals = spec.variables
    .map((variable) => `    ${distributionDefinition(variable.distribution).constructor(variable.parameters)}`)
    .join(",\n");
  const copula =
    spec.copula.kind === "normal"
      ? `\ncorrelation = ot.CorrelationMatrix(${spec.variables.length})\n${spec.copula.correlation
          .flatMap((row, rowIndex) =>
            row.slice(0, rowIndex).map((value, columnIndex) => `correlation[${rowIndex}, ${columnIndex}] = ${value}`),
          )
          .join("\n")}\ncopula = ot.NormalCopula(correlation)\n`
      : "";
  return `import openturns as ot\n\nmodel = ot.SymbolicFunction([${names}], [${outputs}])\nmodel.setOutputDescription([${outputNames}])\n\nmarginals = [\n${marginals}\n]\n${copula}\nproblem = ot.JointDistribution(marginals${spec.copula.kind === "normal" ? ", copula" : ""})\nproblem.setDescription([${names}])\n`;
}
