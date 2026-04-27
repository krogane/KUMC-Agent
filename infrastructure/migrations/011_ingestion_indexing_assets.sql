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

create index if not exists idx_assets_source_kind
  on assets (source_kind, created_at desc);

create index if not exists idx_indexing_runs_status
  on indexing_runs (source_kind, status, created_at desc);
