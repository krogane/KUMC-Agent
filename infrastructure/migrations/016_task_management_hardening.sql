create table if not exists task_change_candidates (
  id text primary key,
  task_id text not null references tasks(id) on delete cascade,
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

create table if not exists task_approval_batches (
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

create index if not exists idx_tasks_assignee_due
  on tasks (assignee_user_id, due_at);

create index if not exists idx_tasks_priority_due
  on tasks (priority, due_at);

create index if not exists idx_tasks_related_event_due
  on tasks (related_event_id, due_at);

create index if not exists idx_task_candidates_created_by
  on task_candidates (created_by, created_at desc);

create index if not exists idx_task_candidates_related_event
  on task_candidates (related_event_id, created_at desc);

create index if not exists idx_task_candidates_confidence
  on task_candidates (confidence, created_at desc);

create index if not exists idx_task_change_candidates_status
  on task_change_candidates (status, created_at desc);

create index if not exists idx_task_change_candidates_task
  on task_change_candidates (task_id, created_at desc);

create index if not exists idx_task_approval_batches_status
  on task_approval_batches (status, created_at desc);
