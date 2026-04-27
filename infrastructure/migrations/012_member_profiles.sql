create table if not exists member_profiles (
  id text primary key,
  display_name text not null default '',
  discord_user_id text not null default '',
  roles jsonb not null default '[]'::jsonb,
  skills jsonb not null default '[]'::jsonb,
  interests jsonb not null default '[]'::jsonb,
  past_assignments jsonb not null default '[]'::jsonb,
  evidence jsonb not null default '[]'::jsonb,
  access_scope jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table member_profiles
  add column if not exists evidence jsonb not null default '[]'::jsonb;

create index if not exists idx_member_profiles_discord_user
  on member_profiles (discord_user_id);
