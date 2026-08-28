export interface ParsedComputeProgress {
  phase: string;
  percent: number;
  message: string;
  indeterminate: boolean;
}

const PROGRESS_PREFIX = "UNCERTAINTYCAT_PROGRESS ";

export function parseProgressLine(line: string): ParsedComputeProgress | null {
  if (!line.startsWith(PROGRESS_PREFIX)) return null;
  try {
    const value = JSON.parse(
      line.slice(PROGRESS_PREFIX.length),
    ) as Partial<ParsedComputeProgress>;
    if (
      typeof value.phase !== "string" ||
      typeof value.percent !== "number" ||
      !Number.isFinite(value.percent) ||
      typeof value.message !== "string" ||
      typeof value.indeterminate !== "boolean"
    ) {
      return null;
    }
    return {
      phase: value.phase.slice(0, 80),
      percent: Math.max(0, Math.min(100, Math.round(value.percent))),
      message: value.message.slice(0, 240),
      indeterminate: value.indeterminate,
    };
  } catch {
    return null;
  }
}
