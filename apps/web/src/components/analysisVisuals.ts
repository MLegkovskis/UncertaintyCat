import type { TableData } from "@uncertaintycat/contracts";

export type TableBarChartDefinition = {
  kind: "bar";
  table: string;
  title: string;
  category: string;
  values: string[];
  stacked?: boolean;
};

export type TableScatterChartDefinition = {
  kind: "scatter";
  table: string;
  title: string;
  label: string;
  x: string;
  y: string;
};

export type TableChartDefinition =
  | TableBarChartDefinition
  | TableScatterChartDefinition;

export const TABLE_CHART_DEFINITIONS: Readonly<
  Record<string, readonly TableChartDefinition[]>
> = {
  ancova: [
    {
      kind: "bar",
      table: "indices",
      title: "ANCOVA contribution decomposition",
      category: "Input",
      values: [
        "ANCOVA Contribution",
        "Physical Contribution",
        "Correlation Contribution",
      ],
    },
  ],
  calibration_nlls: [
    {
      kind: "bar",
      table: "calibrated_parameters",
      title: "Starting and calibrated parameter values",
      category: "Parameter",
      values: ["Starting Value", "Calibrated Value"],
    },
  ],
  fast: [
    {
      kind: "bar",
      table: "indices",
      title: "FAST first-order and total-order indices",
      category: "Variable",
      values: ["First Order", "Total Order"],
    },
    {
      kind: "bar",
      table: "indices",
      title: "FAST total-order decomposition",
      category: "Variable",
      values: ["First Order", "Interaction"],
      stacked: true,
    },
  ],
  hsic: [
    {
      kind: "bar",
      table: "indices",
      title: "Global HSIC dependence ranking",
      category: "Variable",
      values: ["Normalized HSIC"],
    },
  ],
  morris: [
    {
      kind: "scatter",
      table: "effects",
      title: "Morris effect magnitude and dispersion",
      label: "Variable",
      x: "Mean Absolute Effect",
      y: "Effect Dispersion",
    },
  ],
  pce: [
    {
      kind: "bar",
      table: "pce_sobol_indices",
      title: "PCE-derived Sobol indices",
      category: "Input",
      values: ["First Order", "Total Order"],
    },
  ],
  reliability: [
    {
      kind: "bar",
      table: "design_point",
      title: "FORM/SORM design-point importance factors",
      category: "Variable",
      values: ["Importance Factor"],
    },
  ],
  sobol: [
    {
      kind: "bar",
      table: "indices",
      title: "Sobol first-order and total-order indices",
      category: "Variable",
      values: ["First Order", "Total Order"],
    },
  ],
  target_hsic: [
    {
      kind: "bar",
      table: "target_indices",
      title: "Target-domain HSIC association ranking",
      category: "Input",
      values: ["Target R2-HSIC"],
    },
  ],
  taylor: [
    {
      kind: "bar",
      table: "indices",
      title: "Taylor importance factors",
      category: "Variable",
      values: ["Taylor Importance Factor"],
    },
  ],
};

export const ANALYSIS_VISUALIZATION_STRATEGY = {
  ancova: "table chart and validation scatter",
  calibration_nlls: "parameter chart, validation scatter, and correlation heatmap",
  convergence: "convergence series",
  correlation: "coefficient heatmaps",
  eda: "correlation heatmaps",
  fast: "sensitivity bar charts",
  gpr: "validation scatter",
  hsic: "dependence-ranking bar chart",
  monte_carlo: "output histogram",
  morris: "effect scatter",
  pce: "validation scatter and sensitivity bar chart",
  reliability: "importance bar chart or probability convergence series",
  sobol: "sensitivity bar chart and interaction heatmap",
  target_hsic: "target-association bar chart",
  taylor: "importance bar chart",
} as const;

export function tableColumnIndex(table: TableData, column: string) {
  return table.columns.indexOf(column);
}

export function finiteTableNumber(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return value;
}
