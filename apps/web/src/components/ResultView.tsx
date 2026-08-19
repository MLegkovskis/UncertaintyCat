import type {
  AnalysisResult,
  MatrixData,
  SeriesData,
  TableData,
} from "@uncertaintycat/contracts";

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (value !== 0 && (Math.abs(value) < 0.001 || Math.abs(value) >= 100_000))
      return value.toExponential(4);
    return new Intl.NumberFormat("en-GB", {
      maximumSignificantDigits: 6,
    }).format(value);
  }
  return String(value ?? "—");
}

function DataTable({ table }: { table: TableData }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {table.columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((value, columnIndex) => (
                <td key={`${rowIndex}-${columnIndex}`}>{formatValue(value)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {table.truncated && (
        <p className="table-note">
          Showing {table.rows.length.toLocaleString()} of{" "}
          {table.row_count.toLocaleString()} rows. The complete data remains
          available in the export.
        </p>
      )}
    </div>
  );
}

function matrixColour(value: number | null, scale: number): string {
  if (value === null) return "rgba(148, 163, 184, 0.12)";
  const intensity = Math.min(
    1,
    Math.abs(value) / Math.max(scale, Number.EPSILON),
  );
  return value < 0
    ? `rgba(244, 101, 117, ${0.15 + intensity * 0.75})`
    : `rgba(65, 145, 245, ${0.15 + intensity * 0.75})`;
}

function Heatmap({ name, matrix }: { name: string; matrix: MatrixData }) {
  const scale = Math.max(
    ...matrix.values.flatMap((row) => row.map((value) => Math.abs(value ?? 0))),
    1e-12,
  );
  const columns = `minmax(8rem, auto) repeat(${matrix.column_labels.length}, minmax(4.5rem, 1fr))`;
  return (
    <section className="plot-panel">
      <h4>{name.replaceAll("_", " ")}</h4>
      <div className="matrix-scroll">
        <div className="matrix-plot" style={{ gridTemplateColumns: columns }}>
          <span />
          {matrix.column_labels.map((label) => (
            <strong className="matrix-column" key={label}>
              {label}
            </strong>
          ))}
          {matrix.values.flatMap((row, rowIndex) => [
            <strong className="matrix-row" key={`row-${rowIndex}`}>
              {matrix.row_labels[rowIndex]}
            </strong>,
            ...row.map((value, columnIndex) => (
              <span
                className="matrix-cell"
                key={`${rowIndex}-${columnIndex}`}
                style={{ backgroundColor: matrixColour(value, scale) }}
                title={`${matrix.row_labels[rowIndex]} × ${matrix.column_labels[columnIndex]}: ${formatValue(value)}`}
              >
                {formatValue(value)}
              </span>
            )),
          ])}
        </div>
      </div>
      <div className="matrix-legend">
        <span>negative</span>
        <i />
        <span>positive</span>
      </div>
    </section>
  );
}

const CHART_COLOURS = ["#62a8ff", "#75d6a3", "#f6b86b", "#ef7f8e", "#b79aff"];

function SeriesChart({ series }: { series: Record<string, SeriesData> }) {
  const entries = Object.entries(series)
    .map(([key, value]) => ({
      key,
      value,
      points: value.x
        .map((x, index) => [Number(x), Number(value.y[index])] as const)
        .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)),
    }))
    .filter((entry) => entry.points.length > 0);
  if (!entries.length) return null;
  const all = entries.flatMap((entry) => entry.points);
  let xMin = Math.min(...all.map(([x]) => x));
  let xMax = Math.max(...all.map(([x]) => x));
  let yMin = Math.min(...all.map(([, y]) => y));
  let yMax = Math.max(...all.map(([, y]) => y));
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const width = 800;
  const height = 330;
  const left = 74;
  const right = 24;
  const top = 22;
  const bottom = 58;
  const xScale = (value: number) =>
    left + ((value - xMin) / (xMax - xMin)) * (width - left - right);
  const yScale = (value: number) =>
    top + (1 - (value - yMin) / (yMax - yMin)) * (height - top - bottom);
  const ticks = Array.from({ length: 5 }, (_, index) => index / 4);
  const scatter =
    entries.length === 1 && /validation|scatter/i.test(entries[0]!.key);
  return (
    <section className="plot-panel">
      <h4>{scatter ? "Observed versus predicted" : "Series"}</h4>
      <svg
        className="series-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Analysis result series"
      >
        <g className="chart-grid">
          {ticks.map((tick) => (
            <g key={`y-${tick}`}>
              <line
                x1={left}
                x2={width - right}
                y1={yScale(yMin + tick * (yMax - yMin))}
                y2={yScale(yMin + tick * (yMax - yMin))}
              />
              <text
                x={left - 10}
                y={yScale(yMin + tick * (yMax - yMin)) + 4}
                textAnchor="end"
              >
                {formatValue(yMin + tick * (yMax - yMin))}
              </text>
            </g>
          ))}
          {ticks.map((tick) => (
            <g key={`x-${tick}`}>
              <line
                x1={xScale(xMin + tick * (xMax - xMin))}
                x2={xScale(xMin + tick * (xMax - xMin))}
                y1={top}
                y2={height - bottom}
              />
              <text
                x={xScale(xMin + tick * (xMax - xMin))}
                y={height - bottom + 24}
                textAnchor="middle"
              >
                {formatValue(xMin + tick * (xMax - xMin))}
              </text>
            </g>
          ))}
        </g>
        {entries.map((entry, entryIndex) => (
          <g key={entry.key}>
            {!scatter && (
              <polyline
                fill="none"
                stroke={CHART_COLOURS[entryIndex % CHART_COLOURS.length]}
                strokeWidth="2.5"
                points={entry.points
                  .map(([x, y]) => `${xScale(x)},${yScale(y)}`)
                  .join(" ")}
              />
            )}
            {entry.points.map(([x, y], pointIndex) => (
              <circle
                key={pointIndex}
                cx={xScale(x)}
                cy={yScale(y)}
                r={scatter ? 3.2 : 2.1}
                fill={CHART_COLOURS[entryIndex % CHART_COLOURS.length]}
                opacity={scatter ? 0.7 : 1}
              >
                <title>
                  {entry.value.name}: {formatValue(x)}, {formatValue(y)}
                </title>
              </circle>
            ))}
          </g>
        ))}
        <text
          className="axis-label"
          x={(left + width - right) / 2}
          y={height - 8}
          textAnchor="middle"
        >
          {entries[0]!.value.x_label ?? "x"}
        </text>
        <text
          className="axis-label"
          transform={`translate(18 ${(top + height - bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {entries[0]!.value.y_label ?? "y"}
        </text>
      </svg>
      <div className="chart-legend">
        {entries.map((entry, index) => (
          <span key={entry.key}>
            <i
              style={{
                background: CHART_COLOURS[index % CHART_COLOURS.length],
              }}
            />
            {entry.value.name}
          </span>
        ))}
      </div>
    </section>
  );
}

export function ResultView({ result }: { result: AnalysisResult }) {
  const metricEntries = Object.entries(result.payload.metrics);
  return (
    <div className="result-view">
      {result.warnings.length > 0 && (
        <div className="warning-box">
          <strong>Review before interpretation</strong>
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      )}
      {metricEntries.length > 0 && (
        <div className="metrics-grid">
          {metricEntries.map(([name, value]) => (
            <div className="metric" key={name}>
              <span>{name.replaceAll("_", " ")}</span>
              <strong>{formatValue(value)}</strong>
            </div>
          ))}
        </div>
      )}
      {Object.entries(result.payload.tables).map(([name, table]) => (
        <section className="result-block" key={name}>
          <div className="block-heading">
            <h4>{name.replaceAll("_", " ")}</h4>
            <span>{table.row_count.toLocaleString()} rows</span>
          </div>
          <DataTable table={table} />
        </section>
      ))}
      {Object.keys(result.payload.series).length > 0 && (
        <SeriesChart series={result.payload.series} />
      )}
      {Object.entries(result.payload.matrices).map(([name, matrix]) => (
        <Heatmap key={name} name={name} matrix={matrix} />
      ))}
      {Object.keys(result.payload.facts).length > 0 && (
        <section className="facts">
          <h4>Grounded facts</h4>
          <dl>
            {Object.entries(result.payload.facts).map(([name, value]) => (
              <div key={name}>
                <dt>{name.replaceAll(".", " · ").replaceAll("_", " ")}</dt>
                <dd>{formatValue(value)}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
      {result.assumptions.length > 0 && (
        <details className="assumptions">
          <summary>Method assumptions and provenance</summary>
          <ul>
            {result.assumptions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <p>
            OpenTURNS {result.openturns_version} · Core {result.uq_core_version}{" "}
            · Seed {result.seed} · {formatValue(result.runtime.duration_ms)} ms
          </p>
        </details>
      )}
    </div>
  );
}
