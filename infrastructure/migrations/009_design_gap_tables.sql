create table if not exists workflow_candidates (
  id text primary key,
  candidate_type text not null,
  title text not null,
  payload jsonb not null default '{}'::jsonb,
  evidence jsonb not null default '[]'::jsonb,
  confidence text not null default 'low',
  status text not null default 'proposed',
  created_by text not null default 'agent',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists workflow_runs (
  id text primary key,
  workflow_id text not null,
  trigger text not null default 'manual',
  actor_user_id text not null default '',
  guild_id text not null default '',
  input jsonb not null default '{}'::jsonb,
  candidates jsonb not null default '[]'::jsonb,
  drafts jsonb not null default '[]'::jsonb,
  validation_result jsonb not null default '{}'::jsonb,
  approval_required boolean not null default false,
  status text not null default 'running',
  error text not null default '',
  audit_log_id text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists action_specs (
  id text primary key,
  action_type text not null,
  description text not null default '',
  risk_level text not null default 'low',
  approval_policy text not null default 'approval_required',
  schema jsonb not null default '{}'::jsonb,
  enabled boolean not null default true,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists action_runs (
  id text primary key,
  action_type text not null,
  target text not null default '',
  actor_user_id text not null default '',
  status text not null default 'planned',
  risk_level text not null default 'low',
  idempotency_key text not null default '',
  request_payload jsonb not null default '{}'::jsonb,
  result_payload jsonb not null default '{}'::jsonb,
  error text not null default '',
  trace_id text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists action_approvals (
  id text primary key,
  action_run_id text not null,
  actor_user_id text not null default '',
  decision text not null,
  comment text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists llm_calls (
  id text primary key,
  provider text not null default '',
  model text not null default '',
  purpose text not null default '',
  input_metadata jsonb not null default '{}'::jsonb,
  output_metadata jsonb not null default '{}'::jsonb,
  status text not null default 'succeeded',
  cost_usd numeric not null default 0,
  trace_id text not null default '',
  created_at timestamptz not null default now()
);

create table if not exists tool_calls (
  id text primary key,
  tool_name text not null,
  input jsonb not null default '{}'::jsonb,
  output jsonb not null default '{}'::jsonb,
  status text not null default 'succeeded',
  error text not null default '',
  trace_id text not null default '',
  created_at timestamptz not null default now()
);

create table if not exists indexing_runs (
  id text primary key,
  source_kind text not null default '',
  status text not null default 'running',
  seen integer not null default 0,
  changed integer not null default 0,
  skipped integer not null default 0,
  deleted integer not null default 0,
  error text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists assets (
  id text primary key,
  source_kind text not null default '',
  source_item_id text not null default '',
  title text not null default '',
  description text not null default '',
  uri text not null default '',
  media_type text not null default 'image',
  captured_at timestamptz,
  access_scope jsonb not null default '{}'::jsonb,
  rights_status text not null default 'unknown',
  contains_people boolean not null default false,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists asset_usage_requests (
  id text primary key,
  asset_id text not null default '',
  purpose text not null default '',
  medium text not null default '',
  requested_by text not null default '',
  status text not null default 'proposed',
  needs_owner_check boolean not null default true,
  needs_people_check boolean not null default true,
  payload jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists member_profiles (
  id text primary key,
  display_name text not null default '',
  discord_user_id text not null default '',
  roles jsonb not null default '[]'::jsonb,
  skills jsonb not null default '[]'::jsonb,
  interests jsonb not null default '[]'::jsonb,
  past_assignments jsonb not null default '[]'::jsonb,
  evidence jsonb not null default '[]'::jsonb,
  access_scope jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table member_profiles
  add column if not exists evidence jsonb not null default '[]'::jsonb;

create table if not exists finance_records (
  id text primary key,
  record_type text not null default '',
  amount numeric not null default 0,
  currency text not null default 'JPY',
  status text not null default 'draft',
  payload jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists eval_sets (
  id text primary key,
  name text not null,
  description text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists eval_cases (
  id text primary key,
  eval_set_id text not null default '',
  input jsonb not null default '{}'::jsonb,
  expected jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists eval_runs (
  id text primary key,
  eval_set_id text not null default '',
  status text not null default 'running',
  metrics jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists eval_results (
  id text primary key,
  eval_run_id text not null default '',
  eval_case_id text not null default '',
  status text not null default 'unknown',
  scores jsonb not null default '{}'::jsonb,
  output jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists minecraft_wiki_articles (
  id text primary key,
  title text not null,
  url text not null default '',
  revision text not null default '',
  fetched_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists uq_action_runs_idempotency_key
  on action_runs (idempotency_key)
  where idempotency_key <> '';

create index if not exists idx_workflow_candidates_status
  on workflow_candidates (candidate_type, status, created_at desc);

create index if not exists idx_workflow_runs_status
  on workflow_runs (workflow_id, status, created_at desc);

create index if not exists idx_assets_source_kind
  on assets (source_kind, created_at desc);

create index if not exists idx_asset_usage_requests_status
  on asset_usage_requests (status, created_at desc);

create index if not exists idx_member_profiles_discord_user
  on member_profiles (discord_user_id);

create index if not exists idx_indexing_runs_status
  on indexing_runs (source_kind, status, created_at desc);
