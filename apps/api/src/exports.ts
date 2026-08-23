import type { ModelMetadata, Run } from "@uncertaintycat/contracts";
import { strToU8, zipSync } from "fflate";

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text =
    typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csv(columns: string[], rows: unknown[][]): Uint8Array {
  return strToU8(
    [columns, ...rows].map((row) => row.map(csvCell).join(",")).join("\r\n") +
      "\r\n",
  );
}

function slug(value: string): string {
  return (
    value
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-|-$/g, "") || "result"
  );
}

export function createReportBundle(
  run: Run,
  metadata: ModelMetadata | null,
  generatedAt: string,
): Uint8Array {
  const files: Record<string, Uint8Array> = {};
  const manifest = {
    manifestVersion: "1.0.0",
    generatedAt,
    runId: run.id,
    modelVersionId: run.modelVersionId,
    surrogateModelId: run.surrogateModelId ?? null,
    evidenceSource: run.evidenceSource ?? "direct",
    modelHash: metadata?.source_hash ?? null,
    openturnsVersion: metadata?.openturns_version ?? null,
    seed: run.seed,
    accuracyProfile: run.accuracyProfile,
    status: run.status,
    analyses: run.tasks.map((task) => ({
      key: task.analysisKey,
      status: task.status,
      pluginVersion: task.result?.plugin_version ?? null,
      schemaVersion: task.result?.result_schema_version ?? null,
    })),
  };
  files["manifest.json"] = strToU8(JSON.stringify(manifest, null, 2));
  files["report.json"] = strToU8(JSON.stringify({ metadata, run }, null, 2));
  run.tasks.forEach((task, taskIndex) => {
    const prefix = `${String(taskIndex + 1).padStart(2, "0")}-${slug(task.analysisKey)}`;
    files[`results/${prefix}.json`] = strToU8(JSON.stringify(task, null, 2));
    if (!task.result) return;
    Object.entries(task.result.payload.tables).forEach(([name, table]) => {
      files[`tables/${prefix}--${slug(name)}.csv`] = csv(
        table.columns,
        table.rows,
      );
    });
    Object.entries(task.result.payload.matrices).forEach(([name, matrix]) => {
      files[`matrices/${prefix}--${slug(name)}.csv`] = csv(
        ["row", ...matrix.column_labels],
        matrix.values.map((row, index) => [
          matrix.row_labels[index] ?? index,
          ...row,
        ]),
      );
    });
    Object.entries(task.result.payload.series).forEach(([name, series]) => {
      files[`series/${prefix}--${slug(name)}.csv`] = csv(
        [series.x_label ?? "x", series.y_label ?? "y"],
        series.x.map((x, index) => [x, series.y[index] ?? null]),
      );
    });
  });
  return zipSync(files, { level: 6 });
}
