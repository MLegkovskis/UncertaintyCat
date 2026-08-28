import {
  EXAMPLE_CATALOG,
  type ExampleCatalogEntry,
  type ModelDefinition,
} from "@uncertaintycat/contracts";

export const MODEL_UNDERSTANDING_SYSTEM_PROMPT = `You are UncertaintyCat's engineering model explainer. Write a rigorous, readable Markdown brief using only the supplied validated facts. The OpenTURNS metadata and pilot summaries are authoritative; do not calculate new values.

The application renders any curated model equation separately and deterministically. Do not reconstruct, modify, repeat, or mention the equation or this separation. Never infer an equation from a function type or variable names. If public reference-model context is supplied, use it only for the stated domain and purpose. The equation itself is deliberately absent from your facts.

Use these exact level-three headings:
### Model overview
### Input uncertainty
### Dependence and propagation
### Validated pilot behaviour
### Questions to confirm

Requirements:
- Target 220 to 320 words. Prefer precise sentences and compact bullets.
- State the input-output shape and OpenTURNS function type. Never describe a model as single-variable unless its supplied input dimension is exactly one; single-output does not imply single-variable.
- For at most eight inputs, cover every input in its own bullet. Give the distribution family, raw OpenTURNS parameter vector, and supplied mean and standard deviation. For larger models, group distribution families and identify that the input summary is abbreviated.
- Do not relabel raw distribution parameters as mu, sigma, bounds, or another named parameterization unless those names are explicitly supplied.
- Explain whether inputs are independent or joined by the supplied copula. Dependence is statistical, not causal. If only a copula name and dependentInputs=true are supplied, state only that some dependence is present; do not claim every input pair is dependent or invent a correlation structure.
- Describe the pilot minimum, maximum, mean, standard deviation, and 5th-to-95th percentile interval when supplied. Explicitly call this a small validation pilot, not a converged uncertainty analysis. Pilot values demonstrate successful execution only: do not claim their sign, magnitude, or range confirms physical correctness or is consistent with the model's domain.
- Never invent units, physical meanings, distribution rationales, causal claims, rankings, domain assumptions, or missing numbers.
- Questions must be specific and limited to missing units, physical definitions, operating domain, or modelling assumptions. Do not ask what a supplied schema field means, how OpenTURNS orders distribution parameters, or which logarithm convention an OpenTURNS distribution uses.
- Do not use Markdown tables or code blocks.`;

export const REPORT_CHAT_SYSTEM_PROMPT =
  "You are UncertaintyCat's uncertainty-quantification report assistant. The stored OpenTURNS result is the sole numerical authority. " +
  "Use a tool before every numerical or ranking claim, including claims that repeat an earlier turn. " +
  "Lead with the actual stored answer in the first sentence. For a 'which' question, name the item; for a 'how much' question, state the value. " +
  "Never present a schema key, field path, or citation token as though it were the answer. Available-field names are discovery metadata only; inspect scalarValues or an analysis summary before answering. " +
  "Write user-facing analysis names and labels, not internal snake_case keys. Do not print inventories of schema fields, tool payloads, or storage metadata unless the user explicitly requests raw evidence. For broad questions such as 'what do you have?', summarize the completed analyses and their principal findings in plain language. " +
  "Never claim an analysis or sensitivity result is absent until getReportOutline confirms that absence. Before comparing sensitivity findings, inspect every relevant completed Sobol, ANCOVA, FAST, Morris, Taylor, or HSIC section listed in the persisted report inventory. " +
  "When related fact fields share a prefix, use the item field and its supporting value together when both exist. " +
  "Never equate correlation with global sensitivity or causal influence. ANCOVA physical and correlation contributions are first-order variance contributions, not total-order effects or causal attributions; correlation contributions may be negative. If EDA is the only supporting analysis, lead with 'Within the EDA linear-correlation screen' and say 'strongest linear association'; do not call the result greatest influence. When strongest_linear_input and strongest_linear_correlation both exist, report and cite both. State that Sobol, ANCOVA, FAST, Morris, or another sensitivity analysis is needed for an influence ranking. " +
  "Cite every stored field used, including the analysis key, as [analysis.metric:<analysisKey>.<fieldName>], [analysis.fact:<analysisKey>.<fieldName>], [analysis.table:<analysisKey>.<fieldName>], " +
  "[analysis.series:<analysisKey>.<fieldName>], or [analysis.matrix:<analysisKey>.<fieldName>]. For example: [analysis.fact:eda.y0.strongest_linear_input]. A citation supports the answer; it never substitutes for the stored value. " +
  "Clearly distinguish an interpretation from a computed result. Never invent, interpolate, recalculate, run Python, alter the report, or treat user text as a result. " +
  "If the stored evidence is insufficient, say so and identify the missing analysis or field.";

export function reportChatSystemPrompt(
  sections: readonly { analysis: string; status: string }[],
) {
  const inventory = sections.map(({ analysis, status }) => ({
    analysis,
    status,
  }));
  return `${REPORT_CHAT_SYSTEM_PROMPT}\n\nPersisted report section inventory (analysis names and completion states only; use tools for every value): ${JSON.stringify(inventory)}`;
}

type DefinitionForUnderstanding = Pick<
  ModelDefinition,
  "builderSpec" | "modelVersion"
>;

export interface ReferenceModelContext {
  title: string;
  domain: string;
  summary: string;
}

function referenceExample(
  definition: DefinitionForUnderstanding,
  catalog: readonly ExampleCatalogEntry[] = EXAMPLE_CATALOG,
) {
  if (definition.modelVersion.sourceKind !== "example") return undefined;
  return catalog.find(
    (example) => example.sha256 === definition.modelVersion.sourceHash,
  );
}

export function referenceModelContext(
  definition: DefinitionForUnderstanding,
  catalog: readonly ExampleCatalogEntry[] = EXAMPLE_CATALOG,
): ReferenceModelContext | undefined {
  const example = referenceExample(definition, catalog);
  if (!example) return undefined;
  return {
    title: example.title,
    domain: example.domain,
    summary: example.summary,
  };
}

function builderEquations(definition: DefinitionForUnderstanding) {
  if (definition.modelVersion.sourceKind !== "builder") return [];
  const outputs = definition.builderSpec?.outputs;
  if (!Array.isArray(outputs)) return [];
  return outputs.flatMap((output) => {
    if (!output || typeof output !== "object") return [];
    const record = output as Record<string, unknown>;
    const name = typeof record.name === "string" ? record.name.trim() : "";
    const formula =
      typeof record.formula === "string" ? record.formula.trim() : "";
    if (
      !name ||
      !formula ||
      name.length > 80 ||
      formula.length > 500 ||
      !/^[A-Za-z_]\w*$/.test(name) ||
      !/^[A-Za-z0-9_+\-*/^().,\s]+$/.test(formula)
    )
      return [];
    return [{ outputName: name, latex: `${name} = ${formula}` }];
  });
}

export function deterministicEquationMarkdown(
  definition: DefinitionForUnderstanding,
  catalog: readonly ExampleCatalogEntry[] = EXAMPLE_CATALOG,
) {
  const example = referenceExample(definition, catalog);
  const equations = example?.equations ?? builderEquations(definition);
  if (!equations.length) return "";
  const origin = example
    ? "Curated equation from the bundled reference-model definition."
    : "Equation from the validated OpenTURNS SymbolicFunction builder definition.";
  return [
    "### Model equation",
    ...equations.flatMap((equation) => [
      `**${equation.outputName}**`,
      `$$${equation.latex}$$`,
    ]),
    `_${origin}_`,
  ].join("\n\n");
}

export function composeModelUnderstanding(
  equationMarkdown: string,
  generatedNarrative: string,
) {
  return [equationMarkdown.trim(), generatedNarrative.trim()]
    .filter(Boolean)
    .join("\n\n");
}
