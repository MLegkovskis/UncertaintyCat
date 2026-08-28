import {
  BarChart3,
  Database,
  FolderKanban,
  ScanSearch,
  Target,
  Waves,
} from "lucide-react";
import { NavLink } from "react-router-dom";

export function ProjectNav({
  projectId,
  projectName,
}: {
  projectId: string;
  projectName?: string | undefined;
}) {
  const root = `/studies/${projectId}`;
  return (
    <section className="project-context" aria-label="Project workspace navigation">
      <div className="project-context-title">
        <FolderKanban aria-hidden="true" />
        <div>
          <span>Current project</span>
          <strong>{projectName ?? "Project workspace"}</strong>
        </div>
      </div>
      <nav>
        <NavLink to={root} end>
          Overview
        </NavLink>
        <NavLink to={`${root}/workspace`}>
          <BarChart3 /> Model &amp; analyses
        </NavLink>
        <NavLink to={`${root}/dimension-reduction`}>
          <ScanSearch /> Dimensionality reduction
        </NavLink>
        <NavLink to={`${root}/calibration`}>
          <Target /> Calibration Studio
        </NavLink>
        <NavLink to={`${root}/surrogates`}>
          <Waves /> Surrogate Studio
        </NavLink>
        <NavLink to={`${root}/data-lab`}>
          <Database /> Distribution fitting
        </NavLink>
      </nav>
    </section>
  );
}
