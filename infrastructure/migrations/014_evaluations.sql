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
