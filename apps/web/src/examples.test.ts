import { describe, expect, it } from "vitest";

import { buildSymbolicModel } from "./examples";

describe("guided model builder", () => {
  it("produces the canonical model and problem contract", () => {
    const source = buildSymbolicModel(
      [{ name: "load", distribution: "Normal", first: 10, second: 2 }],
      "load^2",
    );
    expect(source).toContain("model = ot.SymbolicFunction");
    expect(source).toContain("problem = ot.JointDistribution");
    expect(source).toContain("ot.Normal(10, 2)");
  });
});
