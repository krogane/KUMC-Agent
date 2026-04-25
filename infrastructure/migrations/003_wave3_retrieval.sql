create table if not exists embeddings (
  chunk_id text not null references chunks(id) on delete cascade,
  model text not null,
  dimensions integer not null,
  embedding jsonb not null,
  checksum text not null,
  created_at timestamptz not null default now(),
  primary key (chunk_id, model, dimensions)
);

create table if not exists search_runs (
  id uuid primary key,
  query text not null,
  actor_id text not null default '',
  guild_id text,
  source_filter text,
  mode text not null default 'answer',
  status text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists search_run_results (
  search_run_id uuid not null references search_runs(id) on delete cascade,
  chunk_id text not null references chunks(id) on delete cascade,
  rank integer not null,
  score double precision not null default 0,
  score_breakdown jsonb not null default '{}'::jsonb,
  primary key (search_run_id, chunk_id)
);

create index if not exists idx_embeddings_model_dimensions
  on embeddings (model, dimensions);

create index if not exists idx_search_runs_created_at
  on search_runs (created_at desc);
