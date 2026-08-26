import { describe, expect, it } from "vitest";

import { formatEvidenceCitations } from "./Markdown";

describe("report evidence citations", () => {
  it("replaces internal evidence paths with human-readable source labels", () => {
    const formatted = formatEvidenceCitations(
      "Input a leads [analysis.fact:sobol.most_influential_input] with 0.61 【analysis.metric:taylor.first_order_variance】.",
    );
    expect(formatted).toContain("Source: Sobol · Most Influential Input");
    expect(formatted).toContain("Source: Taylor · First Order Variance");
    expect(formatted).not.toContain("[analysis.fact:sobol.most_influential_input]");
  });
});
