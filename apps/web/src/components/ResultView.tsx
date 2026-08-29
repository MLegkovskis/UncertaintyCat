import type { EChartsOption } from "echarts";
import { lazy, Suspense } from "react";

import type {
  AnalysisResult,
  MatrixData,
  SeriesData,
  TableData,
} from "@uncertaintycat/contracts";

import {
  finiteTableNumber,
  TABLE_CHART_DEFINITIONS,
  tableColumnIndex,
  type TableChartDefinition,
} from "./analysisVisuals";

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

function TableChart({
  definition,
  table,
}: {
  definition: TableChartDefinition;
  table: TableData;
}) {
  if (definition.kind === "scatter") {
    const labelIndex = tableColumnIndex(table, definition.label);
    const xIndex = tableColumnIndex(table, definition.x);
    const yIndex = tableColumnIndex(table, definition.y);
    if ([labelIndex, xIndex, yIndex].some((index) => index < 0)) return null;
    const points = table.rows.flatMap((row) => {
      const x = finiteTableNumber(row[xIndex]);
      const y = finiteTableNumber(row[yIndex]);
      if (x === undefined || y === undefined) return [];
      return [{ name: String(row[labelIndex] ?? "Input"), value: [x, y] }];
    });
    if (!points.length) return null;
    const option: EChartsOption = {
      tooltip: { trigger: "item" },
      toolbox: {
        right: 4,
        feature: { dataZoom: {}, restore: {}, saveAsImage: { title: "Export image" } },
      },
      grid: { top: 34, right: 30, bottom: 58, left: 72 },
      xAxis: {
        type: "value",
        name: definition.x,
        nameLocation: "middle",
        nameGap: 36,
        scale: true,
      },
      yAxis: {
        type: "value",
        name: definition.y,
        nameLocation: "middle",
        nameGap: 52,
        scale: true,
      },
      series: [
        {
          name: "Inputs",
          type: "scatter",
          data: points,
          symbolSize: 12,
          label: {
            show: points.length <= 20,
            position: "top",
            formatter: "{b}",
          },
        },
      ],
    };
    return (
      <section className="plot-panel evidence-visual">
        <h4>{definition.title}</h4>
        <Suspense fallback={<ChartLoading />}>
          <EChart
            option={option}
            ariaLabel={`${definition.title}; ${points.length} labelled inputs`}
          />
        </Suspense>
      </section>
    );
  }

  const categoryIndex = tableColumnIndex(table, definition.category);
  const valueIndices = definition.values.map((column) =>
    tableColumnIndex(table, column),
  );
  if (categoryIndex < 0 || valueIndices.some((index) => index < 0)) return null;
  const rows = table.rows.filter((row) =>
    valueIndices.some((index) => finiteTableNumber(row[index]) !== undefined),
  );
  if (!rows.length) return null;
  const categories = rows.map((row) => String(row[categoryIndex] ?? "Input"));
  const horizontal = categories.length > 8 || categories.some((item) => item.length > 18);
  const series = definition.values.map((column, index) => ({
    name: column,
    type: "bar" as const,
    data: rows.map((row) => finiteTableNumber(row[valueIndices[index]!]) ?? null),
    ...(definition.stacked ? { stack: "total" } : {}),
    emphasis: { focus: "series" as const },
  }));
  const categoryAxis = {
    type: "category" as const,
    data: categories,
    axisLabel: { interval: 0, rotate: !horizontal && categories.length > 6 ? 25 : 0 },
  };
  const valueAxis = { type: "value" as const, scale: true };
  const option: EChartsOption = {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { top: 0, data: definition.values },
    toolbox: { right: 4, feature: { restore: {}, saveAsImage: { title: "Export image" } } },
    grid: horizontal
      ? { top: 52, right: 36, bottom: 42, left: 118, containLabel: true }
      : { top: 52, right: 30, bottom: 74, left: 66, containLabel: true },
    xAxis: horizontal ? valueAxis : categoryAxis,
    yAxis: horizontal ? categoryAxis : valueAxis,
    series,
  };
  return (
    <section className="plot-panel evidence-visual">
      <h4>{definition.title}</h4>
      <Suspense fallback={<ChartLoading />}>
        <EChart
          option={option}
          ariaLabel={`${definition.title}; ${categories.length} inputs and ${definition.values.length} numerical series`}
        />
      </Suspense>
    </section>
  );
}

function TableCharts({ result }: { result: AnalysisResult }) {
  const definitions = TABLE_CHART_DEFINITIONS[result.analysis_key] ?? [];
  return definitions.map((definition, index) => {
    const table = result.payload.tables[definition.table];
    return table ? (
      <TableChart
        definition={definition}
        key={`${definition.table}-${definition.title}-${index}`}
        table={table}
      />
    ) : null;
  });
}

function histogram(values: number[]) {
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (minimum === maximum) {
    return { labels: [formatValue(minimum)], counts: [values.length] };
  }
  const binCount = Math.max(8, Math.min(40, Math.ceil(Math.log2(values.length) + 1)));
  const width = (maximum - minimum) / binCount;
  const counts = Array.from({ length: binCount }, () => 0);
  for (const value of values) {
    const index = Math.min(binCount - 1, Math.floor((value - minimum) / width));
    counts[index] = (counts[index] ?? 0) + 1;
  }
  const labels = counts.map((_, index) => {
    const lower = minimum + index * width;
    const upper = lower + width;
    return `${formatValue(lower)}–${formatValue(upper)}`;
  });
  return { labels, counts };
}

function OutputHistograms({ series }: { series: Record<string, SeriesData> }) {
  return Object.entries(series).flatMap(([key, value]) => {
    const values = value.y.flatMap((item) => {
      const numerical = finiteTableNumber(item);
      return numerical === undefined ? [] : [numerical];
    });
    if (!values.length) return [];
    const { labels, counts } = histogram(values);
    const option: EChartsOption = {
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      toolbox: { right: 4, feature: { dataZoom: {}, restore: {}, saveAsImage: { title: "Export image" } } },
      grid: { top: 34, right: 30, bottom: 92, left: 68 },
      dataZoom: labels.length > 20 ? [{ type: "inside" }, { type: "slider", bottom: 12 }] : [{ type: "inside" }],
      xAxis: { type: "category", data: labels, name: value.y_label ?? value.name, nameLocation: "middle", nameGap: 66, axisLabel: { rotate: 35 } },
      yAxis: { type: "value", name: "Frequency", minInterval: 1 },
      series: [{ name: value.name, type: "bar", data: counts, barMaxWidth: 42 }],
    };
    return [
      <section className="plot-panel evidence-visual" key={key}>
        <h4>Output distribution · {value.name}</h4>
        <Suspense fallback={<ChartLoading />}>
          <EChart
            option={option}
            ariaLabel={`Histogram of ${value.name} with ${values.length} retained samples`}
          />
        </Suspense>
        <details className="chart-data-fallback">
          <summary>Exact chart data</summary>
          <DataTable
            table={{
              columns: ["Output interval", "Frequency"],
              rows: labels.map((label, index) => [
                label,
                counts[index] ?? 0,
              ]),
              row_count: labels.length,
              truncated: false,
            }}
          />
        </details>
      </section>,
    ];
  });
}

function SeriesChart({ series }: { series: Record<string, SeriesData> }) {
  const entries = Object.entries(series).map(([key, value]) => ({
    key,
    value,
    points: value.x.map((x, index) => [Number(x), Number(value.y[index])] as [number, number]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)),
  })).filter((entry) => entry.points.length > 0);
  if (!entries.length) return null;
  const scatter = entries.every((entry) => /observed_vs_predicted/i.test(entry.key))
    || (entries.length === 1 && /validation|scatter|qq/i.test(entries[0]!.key));
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
  const hasSeries = Object.keys(result.payload.series).length > 0;
  return (
    <div className="result-view">
      {result.warnings.length > 0 && <div className="warning-box"><strong>Review before interpretation</strong>{result.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}
      {metricEntries.length > 0 && <div className="metrics-grid">{metricEntries.map(([name, value]) => <div className="metric" key={name}><span>{name.replaceAll("_", " ")}</span><strong>{formatValue(value)}</strong></div>)}</div>}
      <TableCharts result={result} />
      {hasSeries && result.analysis_key === "monte_carlo" && <OutputHistograms series={result.payload.series} />}
      {hasSeries && result.analysis_key !== "monte_carlo" && <SeriesChart series={result.payload.series} />}
      {Object.entries(result.payload.matrices).map(([name, matrix]) => <Heatmap key={name} name={name} matrix={matrix} />)}
      {Object.entries(result.payload.tables).map(([name, table]) => <section className="result-block" key={name}><div className="block-heading"><h4>{name.replaceAll("_", " ")}</h4><span>{table.row_count.toLocaleString()} rows</span></div><DataTable table={table} /></section>)}
      {Object.keys(result.payload.facts).length > 0 && <section className="facts"><h4>Grounded facts</h4><dl>{Object.entries(result.payload.facts).map(([name, value]) => <div key={name}><dt>{name.replaceAll(".", " · ").replaceAll("_", " ")}</dt><dd>{formatValue(value)}</dd></div>)}</dl></section>}
      {result.assumptions.length > 0 && <details className="assumptions"><summary>Method assumptions and provenance</summary><ul>{result.assumptions.map((item) => <li key={item}>{item}</li>)}</ul><p>OpenTURNS {result.openturns_version} · Core {result.uq_core_version} · Seed {result.seed} · {formatValue(result.runtime.duration_ms)} ms</p></details>}
    </div>
  );
}
