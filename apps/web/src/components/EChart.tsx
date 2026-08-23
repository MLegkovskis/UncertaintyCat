import type { ECharts, EChartsOption } from "echarts";
import { useEffect, useRef } from "react";

import { useTheme } from "./Theme";

export function EChart({
  option,
  ariaLabel,
  height = 340,
}: {
  option: EChartsOption;
  ariaLabel: string;
  height?: number;
}) {
  const { theme } = useTheme();
  const element = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let chart: ECharts | undefined;
    let observer: ResizeObserver | undefined;
    let cancelled = false;
    void import("echarts").then((echarts) => {
      if (cancelled || !element.current) return;
      chart = echarts.init(element.current, theme === "dark" ? "dark" : undefined, {
        renderer: "canvas",
      });
      chart.setOption({
        backgroundColor: "transparent",
        animationDuration: 350,
        aria: { enabled: true, description: ariaLabel },
        textStyle: {
          fontFamily: "system-ui, sans-serif",
          color: theme === "dark" ? "#dce7f3" : "#26364a",
        },
        ...option,
      });
      observer = new ResizeObserver(() => chart?.resize());
      observer.observe(element.current);
    });
    return () => {
      cancelled = true;
      observer?.disconnect();
      chart?.dispose();
    };
  }, [ariaLabel, option, theme]);
  return (
    <div
      ref={element}
      className="echart"
      style={{ height }}
      role="img"
      aria-label={ariaLabel}
    />
  );
}
