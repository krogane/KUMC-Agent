create table if not exists event_candidates (
  id text primary key,
  title text not null,
  summary text,
  starts_at timestamptz,
  ends_at timestamptz,
  place text,
  related_source_ids jsonb not null default '[]'::jsonb,
  evidence jsonb not null default '[]'::jsonb,
  confidence text not null default 'low',
  status text not null default 'proposed',
  created_by text not null default 'agent',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists schedule_candidates (
  id text primary key,
  title text not null,
  starts_at timestamptz,
  ends_at timestamptz,
  place text,
  related_event_id text,
  evidence jsonb not null default '[]'::jsonb,
  confidence text not null default 'low',
  status text not null default 'proposed',
  created_by text not null default 'agent',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_event_candidates_status
  on event_candidates (status, created_at desc);

create index if not exists idx_schedule_candidates_status
  on schedule_candidates (status, created_at desc);
