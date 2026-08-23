ALTER TABLE model_versions ADD COLUMN display_name TEXT NOT NULL DEFAULT 'Untitled model';
ALTER TABLE model_versions ADD COLUMN assessment_json TEXT;
ALTER TABLE model_versions ADD COLUMN parent_version_id TEXT REFERENCES model_versions(id);
ALTER TABLE model_versions ADD COLUMN derivation_json TEXT;

ALTER TABLE report_share_links ADD COLUMN include_model_definition INTEGER NOT NULL DEFAULT 0;

CREATE TABLE model_understandings (
  id TEXT PRIMARY KEY,
  model_version_id TEXT NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
  model_hash TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  ai_model_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending', 'generating', 'succeeded', 'failed')),
  content TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(model_hash, prompt_version)
);

CREATE TABLE datasets (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  owner_id TEXT NOT NULL,
  name TEXT NOT NULL,
  source_kind TEXT NOT NULL CHECK(source_kind IN ('csv', 'xlsx', 'paste')),
  object_key TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  column_metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX datasets_project_idx ON datasets(project_id, created_at DESC);

CREATE TABLE data_analysis_runs (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  owner_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'succeeded', 'failed')),
  config_json TEXT NOT NULL,
  result_json TEXT,
  generated_source TEXT,
  error_json TEXT,
  openturns_version TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE INDEX data_analysis_dataset_idx ON data_analysis_runs(dataset_id, created_at DESC);

CREATE TABLE surrogate_models (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  owner_id TEXT NOT NULL,
  source_model_version_id TEXT NOT NULL REFERENCES model_versions(id),
  source_model_hash TEXT NOT NULL,
  method TEXT NOT NULL CHECK(method IN ('pce', 'gpr')),
  plugin_version TEXT NOT NULL,
  openturns_version TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('draft', 'validated', 'promoted', 'rejected')),
  validation_json TEXT NOT NULL,
  acknowledgement_json TEXT,
  object_key TEXT,
  created_at TEXT NOT NULL,
  promoted_at TEXT
);
CREATE INDEX surrogate_project_idx ON surrogate_models(project_id, created_at DESC);

ALTER TABLE runs ADD COLUMN surrogate_model_id TEXT REFERENCES surrogate_models(id);
CREATE INDEX runs_surrogate_idx ON runs(surrogate_model_id);
