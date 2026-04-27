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

create unique index if not exists uq_action_runs_idempotency_key
  on action_runs (idempotency_key)
  where idempotency_key <> '';

create index if not exists idx_workflow_candidates_status
  on workflow_candidates (candidate_type, status, created_at desc);

create index if not exists idx_workflow_runs_status
  on workflow_runs (workflow_id, status, created_at desc);
