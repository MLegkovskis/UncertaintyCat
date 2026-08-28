import { describe, expect, it } from "vitest";

import { formatEvidenceCitations, humanizeSchemaKeys } from "./Markdown";

describe("report evidence citations", () => {
  it("replaces internal evidence paths with human-readable source labels", () => {
    const formatted = formatEvidenceCitations(
      "Input a leads [analysis.fact:sobol.most_influential_input] with 0.61 【analysis.metric:taylor.first_order_variance】; dependence is [analysis.metric:ancova.sum_correlation_contributions].",
    );
    expect(formatted).toContain("Source: Sobol · Most Influential Input");
    expect(formatted).toContain("Source: Taylor · First Order Variance");
    expect(formatted).toContain("Source: ANCOVA · Sum Correlation Contributions");
    expect(formatted).not.toContain("[analysis.fact:sobol.most_influential_input]");
  });

  it("humanizes stored schema keys without altering ordinary prose", () => {
    expect(
      humanizeSchemaKeys(
        "base_sample_size = 1000; largest_total_order_index = 0.61",
      ),
    ).toBe("Base Sample Size = 1000; Largest Total Order Index = 0.61");
  });
});
