create table if not exists source_accounts (
  id text primary key,
  kind text not null,
  display_name text not null,
  enabled boolean not null default true,
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists source_items (
  id text primary key,
  source_account_id text not null,
  source_kind text not null,
  external_id text not null,
  canonical_url text,
  title text,
  author_id text,
  created_at timestamptz,
  updated_at timestamptz,
  deleted_at timestamptz,
  index_status text not null default 'active',
  access_scope jsonb not null default '{}'::jsonb,
  raw_object_key text not null default '',
  checksum text not null,
  metadata jsonb not null default '{}'::jsonb,
  ingested_at timestamptz not null default now(),
  unique (source_kind, external_id)
);

create index if not exists idx_source_items_kind_status
  on source_items (source_kind, index_status);

create table if not exists documents (
  id text primary key,
  source_item_id text not null references source_items(id) on delete cascade,
  version integer not null default 1,
  title text not null,
  normalized_text text not null,
  normalized_format text not null,
  language text,
  access_scope jsonb not null default '{}'::jsonb,
  checksum text not null,
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists chunks (
  id text primary key,
  document_id text not null references documents(id) on delete cascade,
  source_item_id text not null references source_items(id) on delete cascade,
  chunk_index integer not null,
  chunk_kind text not null,
  text text not null,
  token_count integer not null default 0,
  parent_chunk_id text,
  access_scope jsonb not null default '{}'::jsonb,
  index_status text not null default 'active',
  redaction_policy text not null default 'quote_allowed',
  checksum text not null,
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  unique (document_id, chunk_index, chunk_kind)
);

create index if not exists idx_chunks_source_status
  on chunks (source_item_id, index_status);

create table if not exists chunk_acl_entries (
  chunk_id text not null references chunks(id) on delete cascade,
  acl_type text not null,
  acl_value text not null,
  primary key (chunk_id, acl_type, acl_value)
);

create table if not exists secret_findings (
  id uuid primary key,
  source_item_id text not null references source_items(id) on delete cascade,
  chunk_id text references chunks(id) on delete cascade,
  secret_type text not null,
  severity text not null,
  redaction_policy text not null,
  detected_span_hash text not null,
  status text not null default 'active',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create unique index if not exists uq_secret_findings_item_span
  on secret_findings (source_item_id, detected_span_hash)
  where chunk_id is null;

create unique index if not exists uq_secret_findings_chunk_span
  on secret_findings (chunk_id, detected_span_hash)
  where chunk_id is not null;

create table if not exists sync_cursors (
  source_kind text primary key,
  cursor text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);
