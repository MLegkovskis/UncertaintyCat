import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";

const Studies = lazy(() =>
  import("./pages/Studies").then((module) => ({ default: module.Studies })),
);
const StudyDetail = lazy(() =>
  import("./pages/StudyDetail").then((module) => ({ default: module.StudyDetail })),
);
const DataLab = lazy(() =>
  import("./pages/DataLab").then((module) => ({ default: module.DataLab })),
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
          <Route path="/new-analysis" element={<Workspace />} />
          <Route path="/workspace" element={<Navigate to="/new-analysis" replace />} />
          <Route path="/studies" element={<Studies />} />
          <Route path="/studies/:projectId" element={<StudyDetail />} />
          <Route path="/studies/:projectId/workspace" element={<Workspace />} />
          <Route path="/activity" element={<Navigate to="/studies" replace />} />
          <Route path="/data-lab" element={<DataLab />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route path="/reports/:reportId" element={<ReportPage />} />
          <Route path="/shared/:token" element={<ReportPage shared />} />
        </Routes>
      </Suspense>
    </Shell>
  );
}
