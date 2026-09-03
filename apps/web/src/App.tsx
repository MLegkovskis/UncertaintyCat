import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AuthenticatedRoute } from "./components/AuthenticatedRoute";
import { OperatorRoute } from "./components/OperatorRoute";
import { Shell } from "./components/Shell";

const Studies = lazy(() =>
  import("./pages/Studies").then((module) => ({ default: module.Studies })),
);
const StudyDetail = lazy(() =>
  import("./pages/StudyDetail").then((module) => ({
    default: module.StudyDetail,
  })),
);
const DataLab = lazy(() =>
  import("./pages/DataLab").then((module) => ({ default: module.DataLab })),
);
const DimensionalityReduction = lazy(() =>
  import("./pages/DimensionalityReduction").then((module) => ({
    default: module.DimensionalityReduction,
  })),
);
const CalibrationStudio = lazy(() =>
  import("./pages/CalibrationStudio").then((module) => ({
    default: module.CalibrationStudio,
  })),
);
const SurrogateStudio = lazy(() =>
  import("./pages/SurrogateStudio").then((module) => ({
    default: module.SurrogateStudio,
  })),
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
const OperatorDashboard = lazy(() =>
  import("./pages/OperatorDashboard").then((module) => ({
    default: module.OperatorDashboard,
  })),
);

export function App() {
  const privatePage = (page: ReactNode) => (
    <AuthenticatedRoute>{page}</AuthenticatedRoute>
  );
  return (
    <Shell>
      <Suspense
        fallback={<div className="route-loading">Loading workspace…</div>}
      >
        <Routes>
          <Route path="/" element={<Home />} />
          <Route
            path="/new-analysis"
            element={privatePage(<Navigate to="/studies" replace />)}
          />
          <Route
            path="/workspace"
            element={privatePage(<Navigate to="/studies" replace />)}
          />
          <Route path="/studies" element={privatePage(<Studies />)} />
          <Route
            path="/operator"
            element={privatePage(
              <OperatorRoute>
                <OperatorDashboard />
              </OperatorRoute>,
            )}
          />
          <Route
            path="/studies/:projectId"
            element={privatePage(<StudyDetail />)}
          />
          <Route
            path="/studies/:projectId/workspace"
            element={privatePage(<Workspace />)}
          />
          <Route
            path="/studies/:projectId/dimension-reduction"
            element={privatePage(<DimensionalityReduction />)}
          />
          <Route
            path="/studies/:projectId/calibration"
            element={privatePage(<CalibrationStudio />)}
          />
          <Route
            path="/studies/:projectId/surrogates"
            element={privatePage(<SurrogateStudio />)}
          />
          <Route
            path="/studies/:projectId/data-lab"
            element={privatePage(<DataLab />)}
          />
          <Route
            path="/activity"
            element={privatePage(<Navigate to="/studies" replace />)}
          />
          <Route
            path="/dimension-reduction"
            element={privatePage(<Navigate to="/studies" replace />)}
          />
          <Route
            path="/surrogates"
            element={privatePage(<Navigate to="/studies" replace />)}
          />
          <Route
            path="/data-lab"
            element={privatePage(<Navigate to="/studies" replace />)}
          />
          <Route path="/runs/:runId" element={privatePage(<RunPage />)} />
          <Route
            path="/reports/:reportId"
            element={privatePage(<ReportPage />)}
          />
          <Route
            path="/shared/:token"
            element={privatePage(<ReportPage shared />)}
          />
        </Routes>
      </Suspense>
    </Shell>
  );
}
