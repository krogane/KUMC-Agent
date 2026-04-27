create table if not exists minecraft_wiki_articles (
  id text primary key,
  title text not null,
  url text not null default '',
  revision text not null default '',
  fetched_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
