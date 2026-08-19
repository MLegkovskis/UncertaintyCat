import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";

const Activity = lazy(() =>
  import("./pages/Activity").then((module) => ({ default: module.Activity })),
);
const Home = lazy(() =>
  import("./pages/Home").then((module) => ({ default: module.Home })),
);
const ReportPage = lazy(() =>
  import("./pages/ReportPage").then((module) => ({
    default: module.ReportPage,
  })),
);
const RunPage = lazy(() =>
  import("./pages/RunPage").then((module) => ({ default: module.RunPage })),
);
const Workspace = lazy(() =>
  import("./pages/Workspace").then((module) => ({ default: module.Workspace })),
);

export function App() {
  return (
    <Shell>
      <Suspense
        fallback={<div className="route-loading">Loading workspace…</div>}
      >
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/workspace" element={<Workspace />} />
          <Route path="/activity" element={<Activity />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route path="/reports/:reportId" element={<ReportPage />} />
          <Route path="/shared/:token" element={<ReportPage shared />} />
        </Routes>
      </Suspense>
    </Shell>
  );
}
