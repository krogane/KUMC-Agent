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
