-- Project-scoped keyset history must not scan unrelated projects' executions.
CREATE INDEX IF NOT EXISTS runs_owner_project_history_idx
ON runs(owner_id, project_id, created_at DESC, id DESC);
