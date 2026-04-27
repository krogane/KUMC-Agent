create table if not exists audit_logs (
  event_id uuid primary key,
  action text not null,
  actor_id text not null,
  actor_type text not null,
  target text not null default '',
  outcome text not null,
  risk_level text not null default 'low',
  trace_id text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_audit_logs_action_created_at
  on audit_logs (action, created_at desc);

create table if not exists job_runs (
  job_id uuid primary key,
  job_type text not null,
  status text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  error text not null default '',
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists idx_job_runs_type_status_started_at
  on job_runs (job_type, status, started_at desc);
