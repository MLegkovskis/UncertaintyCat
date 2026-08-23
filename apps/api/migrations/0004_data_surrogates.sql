CREATE TABLE data_surrogate_models (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  owner_id TEXT NOT NULL,
  method TEXT NOT NULL CHECK(method = 'gpr'),
  plugin_version TEXT NOT NULL,
  openturns_version TEXT NOT NULL,
  input_columns_json TEXT NOT NULL,
  output_column TEXT NOT NULL,
  config_json TEXT NOT NULL,
  validation_json TEXT NOT NULL,
  object_key TEXT NOT NULL,
  artifact_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX data_surrogate_project_idx
  ON data_surrogate_models(project_id, created_at DESC);
