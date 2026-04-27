create table if not exists agent_runs (
  id uuid primary key,
  query text not null,
  status text not null,
  access_context jsonb not null default '{}'::jsonb,
  budget jsonb not null default '{}'::jsonb,
  citations jsonb not null default '[]'::jsonb,
  answer text not null default '',
  confidence text not null default 'low',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists agent_steps (
  id uuid primary key,
  run_id uuid not null references agent_runs(id) on delete cascade,
  state text not null,
  input_payload jsonb not null default '{}'::jsonb,
  output_payload jsonb not null default '{}'::jsonb,
  status text not null default 'succeeded',
  cost_usd double precision not null default 0,
  created_at timestamptz not null default now()
);

create table if not exists announcements (
  id text primary key,
  title text not null,
  body_markdown text not null,
  medium text not null default 'discord',
  audience text not null default '',
  status text not null default 'draft',
  fact_checks jsonb not null default '[]'::jsonb,
  citations jsonb not null default '[]'::jsonb,
  created_by text not null default 'agent',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_agent_runs_status_created_at
  on agent_runs (status, created_at desc);

create index if not exists idx_agent_steps_run_state
  on agent_steps (run_id, state, created_at);

create index if not exists idx_announcements_status_created_at
  on announcements (status, created_at desc);
