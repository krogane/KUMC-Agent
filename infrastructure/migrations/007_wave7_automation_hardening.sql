create table if not exists automation_rules (
  id text primary key,
  name text not null,
  enabled boolean not null default false,
  trigger jsonb not null default '{}'::jsonb,
  conditions jsonb not null default '[]'::jsonb,
  actions jsonb not null default '[]'::jsonb,
  mode text not null default 'dry_run',
  risk_level text not null default 'low',
  created_by_user_id text not null default '',
  approved_by_user_id text not null default '',
  last_run_at timestamptz,
  next_run_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists automation_runs (
  id text primary key,
  rule_id text not null references automation_rules(id) on delete cascade,
  trigger_key text not null default '',
  mode text not null default 'dry_run',
  status text not null default 'dry_run',
  idempotency_key text not null unique,
  action_plan jsonb not null default '[]'::jsonb,
  warnings jsonb not null default '[]'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_automation_rules_enabled_mode
  on automation_rules (enabled, mode);

create index if not exists idx_automation_runs_rule_created_at
  on automation_runs (rule_id, created_at desc);

create index if not exists idx_automation_runs_status_created_at
  on automation_runs (status, created_at desc);
