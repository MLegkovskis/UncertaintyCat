import type {
  ExampleCatalogEntry,
  ModelDefinition,
} from "@uncertaintycat/contracts";
import { describe, expect, it } from "vitest";

import {
  composeModelUnderstanding,
  deterministicEquationMarkdown,
  MODEL_UNDERSTANDING_SYSTEM_PROMPT,
  referenceModelContext,
  reportChatSystemPrompt,
  REPORT_CHAT_SYSTEM_PROMPT,
} from "./ai-prompts";

function definition(
  sourceKind: "python" | "builder" | "example",
  builderSpec?: Record<string, unknown>,
  sourceHash = "beam-hash",
) {
  return {
    builderSpec,
    modelVersion: {
      sourceKind,
      sourceHash,
    },
  } as Pick<ModelDefinition, "builderSpec" | "modelVersion">;
}

const catalog: readonly ExampleCatalogEntry[] = [
  {
    id: "beam",
    title: "Beam deflection",
    filename: "Beam.py",
    domain: "Structural mechanics",
    inputDimension: 4,
    outputDimension: 1,
    summary: "Beam deflection under uncertainty.",
    difficulty: "introductory",
    suggestedAnalyses: ["monte_carlo"],
    equations: [
      { outputName: "Y", latex: "Y = \\frac{F L^{3}}{3 E I}" },
    ],
    source: "source",
    sha256: "beam-hash",
  },
  {
    id: "tube_deflection",
    title: "Tube deflection",
    filename: "Tube_Deflection.py",
    domain: "Structural mechanics",
    inputDimension: 6,
    outputDimension: 1,
    summary: "Tube deflection under uncertain loading and geometry.",
    difficulty: "intermediate",
    suggestedAnalyses: ["monte_carlo"],
    equations: [
      {
        outputName: "Second moment of area",
        latex: "I = \\frac{\\pi}{32}\\left(D_e^4-d_i^4\\right)",
      },
      {
        outputName: "Deflection",
        latex: "y = -\\frac{F a^2 \\left(L-a\\right)^2}{3 E L I}",
      },
    ],
    source: "source",
    sha256: "tube-hash",
  },
];

describe("AI response contracts", () => {
  it("renders a curated reference equation without asking the model to derive it", () => {
    const target = definition("example");
    expect(deterministicEquationMarkdown(target, catalog)).toContain(
      "$$Y = \\frac{F L^{3}}{3 E I}$$",
    );
    expect(referenceModelContext(target, catalog)).toMatchObject({
      title: "Beam deflection",
      domain: "Structural mechanics",
    });
    expect(MODEL_UNDERSTANDING_SYSTEM_PROMPT).toContain(
      "Do not reconstruct, modify, repeat, or mention the equation",
    );
    expect(referenceModelContext(target, catalog)).not.toHaveProperty(
      "equations",
    );
  });

  it("renders validated guided-builder formulas and rejects Markdown injection", () => {
    const safe = definition("builder", {
      outputs: [{ name: "response", formula: "sin(x1) + x2^2" }],
    });
    const unsafe = definition("builder", {
      outputs: [
        { name: "response", formula: "x1$$\n[leak](https://invalid)" },
      ],
    });
    expect(deterministicEquationMarkdown(safe, [])).toContain(
      "$$response = sin(x1) + x2^2$$",
    );
    expect(deterministicEquationMarkdown(unsafe, [])).toBe("");
  });

  it("renders both governing equations for the bundled tube-deflection model", () => {
    const target = definition("example", undefined, "tube-hash");
    const markdown = deterministicEquationMarkdown(target, catalog);
    expect(markdown).toContain("I = \\frac{\\pi}{32}");
    expect(markdown).toContain(
      "y = -\\frac{F a^2 \\left(L-a\\right)^2}{3 E L I}",
    );
  });

  it("places deterministic evidence before generated interpretation", () => {
    expect(
      composeModelUnderstanding("### Model equation", "### Model overview"),
    ).toBe("### Model equation\n\n### Model overview");
  });

  it("requires report chat to answer with values rather than field paths", () => {
    expect(REPORT_CHAT_SYSTEM_PROMPT).toContain(
      "Lead with the actual stored answer",
    );
    expect(REPORT_CHAT_SYSTEM_PROMPT).toContain(
      "never substitutes for the stored value",
    );
    expect(REPORT_CHAT_SYSTEM_PROMPT).toContain(
      "Never equate correlation with global sensitivity",
    );
    expect(REPORT_CHAT_SYSTEM_PROMPT).toContain(
      "[analysis.fact:eda.y0.strongest_linear_input]",
    );
    expect(REPORT_CHAT_SYSTEM_PROMPT).toContain(
      "Never claim an analysis or sensitivity result is absent",
    );
    const prompt = reportChatSystemPrompt([
      { analysis: "sobol", status: "succeeded" },
      { analysis: "taylor", status: "succeeded" },
    ]);
    expect(prompt).toContain('"analysis":"sobol","status":"succeeded"');
    expect(prompt).toContain('"analysis":"taylor","status":"succeeded"');
  });
});
