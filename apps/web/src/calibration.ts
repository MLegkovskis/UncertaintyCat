export const MAX_CALIBRATION_ROWS = 250;
export const MAX_CALIBRATION_PARAMETERS = 8;

export const OFFICIAL_CALIBRATION_CSV = `x,y
0.5,4.3712405825862275
1.5,5.2770913648243774
2.5,6.9664982679561884
3.5,9.7657971212483066
4.5,14.076213741899407
5.5,21.588660365352318
6.5,33.730657548172388
7.5,53.897160865582379
8.5,86.9670282151489
9.5,141.54079923319819`;

export interface ParsedCalibrationData {
  inputNames: string[];
  inputs: number[][];
  outputs: number[];
}

export function parseCalibrationCsv(
  value: string,
  expectedInputNames: string[],
  expectedOutputName: string,
): ParsedCalibrationData {
  const lines = value
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) throw new Error("Include a named header and at least one observation row.");

  const header = lines[0]!.split(",").map((cell) => cell.trim());
  if (header.some((cell) => !cell)) throw new Error("CSV column names cannot be empty.");
  if (new Set(header).size !== header.length) throw new Error("CSV column names must be unique.");
  const required = [...expectedInputNames, expectedOutputName];
  const missing = required.filter((name) => !header.includes(name));
  const extra = header.filter((name) => !required.includes(name));
  if (missing.length || extra.length || header.length !== required.length) {
    const details = [
      missing.length ? `missing ${missing.join(", ")}` : "",
      extra.length ? `unexpected ${extra.join(", ")}` : "",
    ].filter(Boolean).join("; ");
    throw new Error(`CSV columns must exactly match ${required.join(", ")}${details ? ` (${details})` : ""}.`);
  }

  const observationLines = lines.slice(1);
  if (observationLines.length > MAX_CALIBRATION_ROWS) {
    throw new Error(`Calibration accepts at most ${MAX_CALIBRATION_ROWS} observation rows.`);
  }
  const rows = observationLines.map((line, rowIndex) => {
    const cells = line.split(",").map((cell) => cell.trim());
    if (cells.length !== header.length) {
      throw new Error(`Observation row ${rowIndex + 1} has ${cells.length} values; expected ${header.length}.`);
    }
    return cells.map((cell, columnIndex) => {
      if (!cell) throw new Error(`Observation row ${rowIndex + 1} contains an empty value.`);
      const number = Number(cell);
      if (!Number.isFinite(number)) {
        throw new Error(`Observation row ${rowIndex + 1}, column ${header[columnIndex]} is not finite.`);
      }
      return number;
    });
  });
  const indexByName = new Map(header.map((name, index) => [name, index]));
  return {
    inputNames: expectedInputNames,
    inputs: rows.map((row) => expectedInputNames.map((name) => row[indexByName.get(name)!]!)),
    outputs: rows.map((row) => row[indexByName.get(expectedOutputName)!]!),
  };
}

export function isOfficialCalibrationModel(
  inputNames: string[],
  outputNames: string[],
) {
  return inputNames.join("|") === "a|b|c|x" && outputNames.join("|") === "y";
}
