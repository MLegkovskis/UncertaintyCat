CREATE INDEX IF NOT EXISTS runs_created_status_idx ON runs(created_at DESC, status);
CREATE INDEX IF NOT EXISTS tasks_status_created_idx ON analysis_tasks(status, created_at DESC);
CREATE INDEX IF NOT EXISTS data_analysis_status_created_idx ON data_analysis_runs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS model_understandings_status_updated_idx ON model_understandings(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS projects_updated_idx ON projects(updated_at DESC);
