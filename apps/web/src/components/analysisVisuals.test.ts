import { describe, expect, it } from "vitest";

import {
  ANALYSIS_VISUALIZATION_STRATEGY,
  TABLE_CHART_DEFINITIONS,
} from "./analysisVisuals";

const CATALOG_ANALYSES = [
  "ancova",
  "calibration_nlls",
  "convergence",
  "correlation",
  "eda",
  "fast",
  "gpr",
  "hsic",
  "monte_carlo",
  "morris",
  "pce",
  "reliability",
  "sobol",
  "target_hsic",
  "taylor",
].sort();

describe("analysis visualization strategy", () => {
  it("declares meaningful visual evidence for every registered analysis", () => {
    expect(Object.keys(ANALYSIS_VISUALIZATION_STRATEGY).sort()).toEqual(
      CATALOG_ANALYSES,
    );
    expect(ANALYSIS_VISUALIZATION_STRATEGY.fast).toContain("bar chart");
    expect(ANALYSIS_VISUALIZATION_STRATEGY.monte_carlo).toContain("histogram");
  });

  it("keeps every table-driven chart bounded and fully specified", () => {
    for (const [analysis, definitions] of Object.entries(
      TABLE_CHART_DEFINITIONS,
    )) {
      expect(CATALOG_ANALYSES).toContain(analysis);
      expect(definitions.length).toBeGreaterThan(0);
      for (const definition of definitions) {
        expect(definition.table).not.toBe("");
        expect(definition.title).not.toBe("");
        if (definition.kind === "bar") {
          expect(definition.values.length).toBeGreaterThan(0);
          expect(new Set(definition.values).size).toBe(
            definition.values.length,
          );
        } else {
          expect(new Set([definition.label, definition.x, definition.y]).size).toBe(
            3,
          );
        }
      }
    }
  });
});
