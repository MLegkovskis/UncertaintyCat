import { unzipSync, strFromU8 } from "fflate";
import { describe, expect, it } from "vitest";

import { createReportBundle } from "./exports";

describe("createReportBundle", () => {
  it("includes provenance JSON and lossless CSV tables", () => {
    const archive = createReportBundle(
      {
        id: "run-1",
        projectId: "project-1",
        modelVersionId: "model-1",
        status: "succeeded",
        seed: 42,
        accuracyProfile: "standard",
        createdAt: "2026-01-01T00:00:00Z",
        tasks: [
          {
            id: "task-1",
            analysisKey: "demo",
            config: {},
            outputTargets: [],
            status: "succeeded",
            result: {
              analysis_key: "demo",
              plugin_version: "1.0.0",
              result_schema_version: "1.0.0",
              model_hash: "hash",
              seed: 42,
              uq_core_version: "0.2.0",
              openturns_version: "1.25",
              status: "succeeded",
              started_at: "2026-01-01T00:00:00Z",
              completed_at: "2026-01-01T00:00:01Z",
              runtime: { duration_ms: 1, model_evaluations: 1 },
              warnings: [],
              assumptions: [],
              payload: {
                metrics: {},
                facts: {},
                matrices: {},
                series: {},
                tables: {
                  values: {
                    columns: ["name", "value"],
                    rows: [["quoted, value", 2]],
                    row_count: 1,
                    truncated: false,
                  },
                },
              },
            },
          },
        ],
      },
      null,
      "2026-01-01T00:00:02Z",
    );
    const files = unzipSync(archive);
    expect(Object.keys(files)).toContain("manifest.json");
    expect(strFromU8(files["tables/01-demo--values.csv"]!)).toContain(
      '"quoted, value",2',
    );
  });
});
