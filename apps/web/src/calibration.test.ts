import { describe, expect, it } from "vitest";

import {
  MAX_CALIBRATION_ROWS,
  OFFICIAL_CALIBRATION_CSV,
  parseCalibrationCsv,
} from "./calibration";

describe("calibration CSV", () => {
  it("parses and reorders the named official observations", () => {
    const parsed = parseCalibrationCsv(OFFICIAL_CALIBRATION_CSV, ["x"], "y");
    expect(parsed.inputNames).toEqual(["x"]);
    expect(parsed.inputs).toHaveLength(10);
    expect(parsed.inputs[0]).toEqual([0.5]);
    expect(parsed.outputs.at(-1)).toBe(141.54079923319819);

    const reordered = parseCalibrationCsv("y,x\n4.5,1.25", ["x"], "y");
    expect(reordered.inputs).toEqual([[1.25]]);
    expect(reordered.outputs).toEqual([4.5]);
  });

  it("rejects malformed names and non-finite values", () => {
    expect(() => parseCalibrationCsv("x,x,y\n1,2,3", ["x"], "y")).toThrow("unique");
    expect(() => parseCalibrationCsv("wrong,y\n1,2", ["x"], "y")).toThrow("exactly match");
    expect(() => parseCalibrationCsv("x,y\n1,Infinity", ["x"], "y")).toThrow("not finite");
    expect(() => parseCalibrationCsv("x,y\n1", ["x"], "y")).toThrow("expected 2");
  });

  it("enforces the stored-observation bound", () => {
    const rows = Array.from({ length: MAX_CALIBRATION_ROWS + 1 }, (_, index) => `${index},${index}`);
    expect(() => parseCalibrationCsv(["x,y", ...rows].join("\n"), ["x"], "y")).toThrow("at most 250");
  });
});
