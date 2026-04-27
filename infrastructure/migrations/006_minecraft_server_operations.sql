create table if not exists server_operations (
  id text primary key,
  server_name text not null,
  operation text not null,
  requested_by_user_id text not null default '',
  approved_by_user_ids jsonb not null default '[]'::jsonb,
  status text not null default 'waiting_approval',
  risk_level text not null default 'medium',
  action_run_id text,
  dry_run jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_server_operations_status_created_at
  on server_operations (status, created_at desc);

create index if not exists idx_server_operations_operation_risk
  on server_operations (operation, risk_level);
