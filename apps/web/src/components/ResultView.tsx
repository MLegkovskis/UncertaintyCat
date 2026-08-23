import type { EChartsOption } from "echarts";
import { lazy, Suspense } from "react";

import type {
  AnalysisResult,
  MatrixData,
  SeriesData,
  TableData,
} from "@uncertaintycat/contracts";

const EChart = lazy(() =>
  import("./EChart").then((module) => ({ default: module.EChart })),
);

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (value !== 0 && (Math.abs(value) < 0.001 || Math.abs(value) >= 100_000))
      return value.toExponential(4);
    return new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 6 }).format(value);
  }
  return String(value ?? "—");
}

function tableValue(value: unknown): string | number | boolean | null {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (value === undefined) return null;
  return JSON.stringify(value);
}

function DataTable({ table }: { table: TableData }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{table.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>
          {table.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>{row.map((value, columnIndex) => <td key={`${rowIndex}-${columnIndex}`}>{formatValue(value)}</td>)}</tr>
          ))}
        </tbody>
      </table>
      {table.truncated && <p className="table-note">Showing {table.rows.length.toLocaleString()} of {table.row_count.toLocaleString()} rows. The complete data remains available in the export.</p>}
    </div>
  );
}

function ChartLoading() {
  return <div className="chart-loading" aria-live="polite">Loading interactive chart…</div>;
}

function Heatmap({ name, matrix }: { name: string; matrix: MatrixData }) {
  const values = matrix.values.flatMap((row, rowIndex) =>
    row.map((value, columnIndex) => [columnIndex, rowIndex, value] as [number, number, number | null]),
  );
  const finite = values.map((item) => item[2]).filter((value): value is number => value !== null && Number.isFinite(value));
  const extent = Math.max(...finite.map(Math.abs), 1e-12);
  const option: EChartsOption = {
    tooltip: { position: "top" },
    toolbox: { feature: { saveAsImage: { title: "Export image" }, restore: {} } },
    grid: { top: 32, right: 28, bottom: 70, left: 100 },
    xAxis: { type: "category", data: matrix.column_labels, splitArea: { show: true }, axisLabel: { rotate: matrix.column_labels.length > 8 ? 35 : 0 } },
    yAxis: { type: "category", data: matrix.row_labels, splitArea: { show: true } },
    visualMap: { min: -extent, max: extent, calculable: true, orient: "horizontal", left: "center", bottom: 5, inRange: { color: ["#c94f61", "#f7f8fa", "#08717b"] } },
    series: [{ name, type: "heatmap", data: values, label: { show: matrix.row_labels.length <= 12 && matrix.column_labels.length <= 12, formatter: (params) => formatValue((params.value as unknown[])[2]) }, emphasis: { itemStyle: { shadowBlur: 8, shadowColor: "rgba(0,0,0,0.25)" } } }],
  };
  return (
    <section className="plot-panel">
      <h4>{name.replaceAll("_", " ")}</h4>
      <Suspense fallback={<ChartLoading />}><EChart option={option} ariaLabel={`${name} heatmap with ${matrix.row_labels.length} rows and ${matrix.column_labels.length} columns`} /></Suspense>
      <details className="chart-data-fallback"><summary>Exact heatmap values</summary><DataTable table={{ columns: ["Row", ...matrix.column_labels], rows: matrix.values.map((row, index) => [matrix.row_labels[index] ?? index, ...row]), row_count: matrix.values.length, truncated: false }} /></details>
    </section>
  );
}

function SeriesChart({ series }: { series: Record<string, SeriesData> }) {
  const entries = Object.entries(series).map(([key, value]) => ({
    key,
    value,
    points: value.x.map((x, index) => [Number(x), Number(value.y[index])] as [number, number]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)),
  })).filter((entry) => entry.points.length > 0);
  if (!entries.length) return null;
  const scatter = entries.length === 1 && /validation|scatter|qq/i.test(entries[0]!.key);
  const pointCount = Math.max(...entries.map((entry) => entry.points.length));
  const option: EChartsOption = {
    tooltip: { trigger: scatter ? "item" : "axis" },
    legend: { top: 0, data: entries.map((entry) => entry.value.name) },
    toolbox: { right: 4, feature: { dataZoom: {}, restore: {}, saveAsImage: { title: "Export image" } } },
    grid: { top: 52, right: 30, bottom: pointCount > 100 ? 72 : 50, left: 68 },
    dataZoom: pointCount > 100 ? [{ type: "inside" }, { type: "slider", bottom: 8 }] : [{ type: "inside" }],
    xAxis: { type: "value", name: entries[0]!.value.x_label ?? "x", nameLocation: "middle", nameGap: 32, scale: true },
    yAxis: { type: "value", name: entries[0]!.value.y_label ?? "y", nameLocation: "middle", nameGap: 48, scale: true },
    series: entries.map((entry) => ({
      name: entry.value.name,
      type: scatter ? "scatter" : "line",
      data: entry.points,
      symbolSize: scatter ? 7 : 4,
      showSymbol: scatter || entry.points.length < 200,
      sampling: "lttb",
      ...(scatter ? { markLine: { symbol: ["none", "none"], data: [[{ coord: ["min", "min"] }, { coord: ["max", "max"] }]], lineStyle: { type: "dashed", color: "#64748b" } } } : {}),
    })),
  };
  const maxRows = Math.max(...entries.map((entry) => entry.value.x.length));
  const rows = Array.from({ length: maxRows }, (_, index) => entries.flatMap((entry) => [tableValue(entry.value.x[index]), tableValue(entry.value.y[index])]));
  return (
    <section className="plot-panel">
      <h4>{scatter ? "Observed versus predicted" : "Convergence and series"}</h4>
      <Suspense fallback={<ChartLoading />}><EChart option={option} ariaLabel={scatter ? "Observed versus predicted validation scatter plot" : `Interactive result chart with ${entries.length} series`} /></Suspense>
      <details className="chart-data-fallback"><summary>Exact chart data</summary><DataTable table={{ columns: entries.flatMap((entry) => [`${entry.value.name} · ${entry.value.x_label ?? "x"}`, `${entry.value.name} · ${entry.value.y_label ?? "y"}`]), rows, row_count: maxRows, truncated: false }} /></details>
    </section>
  );
}

export function ResultView({ result }: { result: AnalysisResult }) {
  const metricEntries = Object.entries(result.payload.metrics);
  return (
    <div className="result-view">
      {result.warnings.length > 0 && <div className="warning-box"><strong>Review before interpretation</strong>{result.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}
      {metricEntries.length > 0 && <div className="metrics-grid">{metricEntries.map(([name, value]) => <div className="metric" key={name}><span>{name.replaceAll("_", " ")}</span><strong>{formatValue(value)}</strong></div>)}</div>}
      {Object.entries(result.payload.tables).map(([name, table]) => <section className="result-block" key={name}><div className="block-heading"><h4>{name.replaceAll("_", " ")}</h4><span>{table.row_count.toLocaleString()} rows</span></div><DataTable table={table} /></section>)}
      {Object.keys(result.payload.series).length > 0 && <SeriesChart series={result.payload.series} />}
      {Object.entries(result.payload.matrices).map(([name, matrix]) => <Heatmap key={name} name={name} matrix={matrix} />)}
      {Object.keys(result.payload.facts).length > 0 && <section className="facts"><h4>Grounded facts</h4><dl>{Object.entries(result.payload.facts).map(([name, value]) => <div key={name}><dt>{name.replaceAll(".", " · ").replaceAll("_", " ")}</dt><dd>{formatValue(value)}</dd></div>)}</dl></section>}
      {result.assumptions.length > 0 && <details className="assumptions"><summary>Method assumptions and provenance</summary><ul>{result.assumptions.map((item) => <li key={item}>{item}</li>)}</ul><p>OpenTURNS {result.openturns_version} · Core {result.uq_core_version} · Seed {result.seed} · {formatValue(result.runtime.duration_ms)} ms</p></details>}
    </div>
  );
}
