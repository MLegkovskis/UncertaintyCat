import { describe, expect, it } from "vitest";

import {
  buildSymbolicModel,
  identityCorrelation,
  validateCorrelation,
} from "./examples";

describe("guided model builder", () => {
  it("produces the canonical model and problem contract", () => {
    const source = buildSymbolicModel({
      variables: [{ id: "input-1", name: "load", distribution: "Normal", parameters: [10, 2] }],
      outputs: [
        { id: "output-1", name: "stress", formula: "load^2" },
        { id: "output-2", name: "deflection", formula: "load / 2" },
      ],
      copula: { kind: "independent", correlation: identityCorrelation(1) },
    });
    expect(source).toContain("model = ot.SymbolicFunction");
    expect(source).toContain("problem = ot.JointDistribution");
    expect(source).toContain("ot.Normal(10, 2)");
    expect(source).toContain('["stress", "deflection"]');
  });

  it("rejects a non-positive-definite Normal copula matrix", () => {
    expect(validateCorrelation([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])).toMatch(
      /positive definite/,
    );
  });
});
