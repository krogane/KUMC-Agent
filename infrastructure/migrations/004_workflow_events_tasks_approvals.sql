create table if not exists events (
  id text primary key,
  title text not null,
  summary text,
  starts_at timestamptz,
  ends_at timestamptz,
  place text,
  status text not null default 'planning',
  related_source_ids jsonb not null default '[]'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists meetings (
  id text primary key,
  title text not null,
  scheduled_at timestamptz,
  related_event_id text references events(id) on delete set null,
  agenda_markdown text not null default '',
  minutes_markdown text not null default '',
  decisions jsonb not null default '[]'::jsonb,
  open_questions jsonb not null default '[]'::jsonb,
  task_candidate_ids jsonb not null default '[]'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists task_candidates (
  id text primary key,
  title text not null,
  description text,
  proposed_assignee_user_id text,
  proposed_due_at timestamptz,
  related_event_id text references events(id) on delete set null,
  evidence jsonb not null default '[]'::jsonb,
  confidence text not null default 'low',
  status text not null default 'proposed',
  created_by text not null default 'agent',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists tasks (
  id text primary key,
  title text not null,
  description text,
  assignee_user_id text,
  due_at timestamptz,
  related_event_id text references events(id) on delete set null,
  source_candidate_id text references task_candidates(id) on delete set null,
  status text not null default 'todo',
  priority text not null default 'normal',
  evidence jsonb not null default '[]'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists schedule_events (
  id text primary key,
  title text not null,
  starts_at timestamptz,
  ends_at timestamptz,
  place text,
  related_event_id text references events(id) on delete set null,
  status text not null default 'planned',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists approval_records (
  id uuid primary key,
  target_type text not null,
  target_id text not null,
  action text not null,
  actor_id text not null default '',
  comment text not null default '',
  before_payload jsonb not null default '{}'::jsonb,
  after_payload jsonb not null default '{}'::jsonb,
  evidence jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_task_candidates_status
  on task_candidates (status, created_at desc);

create index if not exists idx_tasks_status_due
  on tasks (status, due_at);

create index if not exists idx_events_starts_at
  on events (starts_at);

create index if not exists idx_schedule_events_starts_at
  on schedule_events (starts_at);

create index if not exists idx_approval_records_target
  on approval_records (target_type, target_id, created_at desc);
