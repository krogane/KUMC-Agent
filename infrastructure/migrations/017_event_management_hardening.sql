create table if not exists event_change_candidates (
  id text primary key,
  event_id text not null references events(id) on delete cascade,
  operation text not null,
  before_payload jsonb not null default '{}'::jsonb,
  after_payload jsonb not null default '{}'::jsonb,
  reason text not null default '',
  evidence jsonb not null default '[]'::jsonb,
  confidence text not null default 'medium',
  status text not null default 'proposed',
  created_by text not null default 'user',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists event_approval_batches (
  id text primary key,
  candidate_ids jsonb not null default '[]'::jsonb,
  change_candidate_ids jsonb not null default '[]'::jsonb,
  period_start timestamptz,
  period_end timestamptz,
  notification_channel_id text,
  notification_message_id text,
  status text not null default 'pending',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_events_status_starts_at
  on events (status, starts_at);

create index if not exists idx_events_place
  on events (place);

create index if not exists idx_event_candidates_created_by
  on event_candidates (created_by, created_at desc);

create index if not exists idx_event_candidates_confidence
  on event_candidates (confidence, created_at desc);

create index if not exists idx_event_candidates_starts_at
  on event_candidates (starts_at);

create index if not exists idx_event_change_candidates_status
  on event_change_candidates (status, created_at desc);

create index if not exists idx_event_change_candidates_event
  on event_change_candidates (event_id, created_at desc);

create index if not exists idx_event_approval_batches_status
  on event_approval_batches (status, created_at desc);
