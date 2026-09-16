import { getSandbox } from "@cloudflare/sandbox";
import { describe, expect, it, vi } from "vitest";

import { computeFetch } from "./compute-client";
import { parseProgressLine } from "./compute-progress";
import type { Env } from "./env";

vi.mock("@cloudflare/sandbox", () => ({ getSandbox: vi.fn() }));

describe("production Sandbox endpoint dispatch", () => {
  it.each([
    ["/v1/catalog", "catalog"],
    ["/v1/validate", "validate"],
    ["/v1/execute", "execute"],
    ["/v1/data/inspect", "inspect-data"],
    ["/v1/data/fit", "fit-data"],
    ["/v1/data/surrogate", "fit-data-surrogate"],
    ["/v1/surrogates/serialize", "serialize-surrogate"],
    ["/v1/surrogates/execute", "execute-surrogate"],
  ])("dispatches %s to its exact CLI operation %s", async (path, operation) => {
    const exec = vi.fn().mockResolvedValue({
      success: true,
      stdout: JSON.stringify({ status: 200, body: { result: { status: "succeeded" } } }),
      stderr: "",
    });
    const writeFile = vi.fn().mockResolvedValue(undefined);
    const destroy = vi.fn().mockResolvedValue(undefined);
    vi.mocked(getSandbox).mockReturnValue({ exec, writeFile, destroy } as unknown as ReturnType<typeof getSandbox>);
    const body = JSON.stringify({ xml_base64: "private-artifact-marker", run_id: "12345678-1234-1234-1234-123456789012" });
    const response = await computeFetch({ SANDBOX: {} } as Env, path, { method: "POST", body });
    expect(response.status).toBe(200);
    const [command] = exec.mock.calls[0]!;
    expect(command).toMatch(new RegExp(`services\\.compute\\.cli ${operation} /workspace/request-[0-9a-f-]+\\.json$`));
    expect(command).not.toContain("private-artifact-marker");
    expect(writeFile).toHaveBeenCalledWith(expect.stringMatching(/^\/workspace\/request-/), body);
    if (operation === "execute" || operation === "execute-surrogate") {
      expect(getSandbox).toHaveBeenLastCalledWith(expect.anything(), "uq-run-12345678-1234-1234-1234-123456789012", { enableDefaultSession: false });
      expect(destroy).not.toHaveBeenCalled();
    } else {
      expect(destroy).toHaveBeenCalledOnce();
    }
  });

  it("rejects suffix lookalikes before creating a Sandbox", async () => {
    vi.mocked(getSandbox).mockClear();
    await expect(computeFetch({ SANDBOX: {} } as Env, "/unrecognized/private-marker/execute")).rejects.toThrow("Unsupported compute path.");
    expect(getSandbox).not.toHaveBeenCalled();
  });
});

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
