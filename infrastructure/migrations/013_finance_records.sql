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
