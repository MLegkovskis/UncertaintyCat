import { describe, expect, it } from "vitest";

import { parseProgressLine } from "./compute-progress";

describe("isolated compute progress protocol", () => {
  it("accepts a bounded structured progress line", () => {
    expect(
      parseProgressLine(
        'UNCERTAINTYCAT_PROGRESS {"phase":"permutation_inference","percent":58.4,"message":"OpenTURNS is evaluating 100 permutation replicates.","indeterminate":true}',
      ),
    ).toEqual({
      phase: "permutation_inference",
      percent: 58,
      message: "OpenTURNS is evaluating 100 permutation replicates.",
      indeterminate: true,
    });
  });

  it("ignores ordinary stderr and malformed progress instead of persisting it", () => {
    expect(
      parseProgressLine("model = ot.PythonFunction(8, 1, evaluate)"),
    ).toBeNull();
    expect(parseProgressLine("UNCERTAINTYCAT_PROGRESS not-json")).toBeNull();
    expect(
      parseProgressLine(
        'UNCERTAINTYCAT_PROGRESS {"phase":"sampling","percent":"half","message":"active","indeterminate":true}',
      ),
    ).toBeNull();
  });

  it("clamps percentages and truncates persisted display fields", () => {
    const progress = parseProgressLine(
      `UNCERTAINTYCAT_PROGRESS ${JSON.stringify({
        phase: "p".repeat(100),
        percent: 200,
        message: "m".repeat(300),
        indeterminate: false,
      })}`,
    );

    expect(progress?.percent).toBe(100);
    expect(progress?.phase).toHaveLength(80);
    expect(progress?.message).toHaveLength(240);
  });
});
