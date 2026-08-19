#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# TrajProxy initialization
set -e

file_dir=$(realpath $(dirname $0))
root_dir=${file_dir}
traj_proxy_dir=${file_dir}/app

TRAJ_PROXY_DATA="${TRAJ_PROXY_DATA:-/traj_proxy/data}"
mkdir -p "${TRAJ_PROXY_DATA}/models"
mkdir -p "${TRAJ_PROXY_DATA}/logs"
mkdir -p "${TRAJ_PROXY_DATA}/postgresql"
mkdir -p "${TRAJ_PROXY_DATA}/archives"
path="${TRAJ_PROXY_DATA}"; while [[ "$path" != "/" && "$path" != "." ]]; do [ -d "$path" ] && chmod o+x "$path"; path=$(dirname "$path"); done
chmod o+rwx "${TRAJ_PROXY_DATA}/logs"
chown -R postgres:postgres "${TRAJ_PROXY_DATA}/postgresql"
chown -R postgres:postgres "${TRAJ_PROXY_DATA}/archives"

# ========================================
# Default environment variable values
# ========================================
PG_VERSION=14
PGDATA="${PGDATA:-${TRAJ_PROXY_DATA}/postgresql}"
POSTGRES_USER="${POSTGRES_USER:-llmproxy}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-dbpassword9090}"
TRAJ_PROXY_DB="${TRAJ_PROXY_DB:-traj_proxy}"
POSTGRES_DB="${POSTGRES_DB:-litellm}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-1234}"
LITELLM_SALT_KEY="${LITELLM_SALT_KEY:-sk-1234}"

# Auto-generate database connection URLs (user-set env vars take precedence)
export TRAJ_PROXY_DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${TRAJ_PROXY_DB}}"
export LITELLM_DATABASE_URL="${LITELLM_DATABASE_URL:-postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${POSTGRES_DB}}"
export STORE_MODEL_IN_DB="${STORE_MODEL_IN_DB:-True}"

# ========================================
# Full initialization mode (container entrypoint)
# ========================================
echo "=== TrajProxy All-in-One container startup ==="

# ========================================
# Phase 1: PostgreSQL data directory initialization
# ========================================
if [ ! -f "${PGDATA}/PG_VERSION" ]; then
    echo "--- Initializing PostgreSQL data directory ---"
    mkdir -p "${PGDATA}"
    chown postgres:postgres "${PGDATA}"
    cd "${PGDATA}"

    # Initialize the database cluster
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/initdb -D \"${PGDATA}\" -E UTF8 --locale=C --auth=trust"

    # Configure pg_hba.conf to allow local connections
    cat > "${PGDATA}/pg_hba.conf" <<EOF
# Local connections (all-in-one mode)
local   all             all                                     trust
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
EOF

    echo "PostgreSQL data directory initialized"
else
    echo "PostgreSQL data directory already exists, skipping initialization"
fi

# Ensure directory permissions are correct
chown -R postgres:postgres "${PGDATA}"
cd "${PGDATA}"

# ========================================
# Phase 2: Start PostgreSQL temporarily, create user and database
# ========================================
echo "--- Starting PostgreSQL for initialization ---"
if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q"; then
    echo "PostgreSQL is already running"
else
    echo "PostgreSQL is not running, starting..."
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl start -D \"${PGDATA}\" -l ${TRAJ_PROXY_DATA}/logs/postgresql_init.log -w -t 60"
fi

# Wait for PostgreSQL to accept connections
echo "--- Waiting for PostgreSQL to be ready ---"
for i in $(seq 1 30); do
    if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/psql -U postgres -c '\q' 2>/dev/null"; then
        echo "PostgreSQL is ready"
        break
    fi
    sleep 1
done

# Create user (if not exists)
echo "--- Creating database user ---"
su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/psql -c \"DO 'BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = ''${POSTGRES_USER}'') THEN CREATE USER ${POSTGRES_USER} WITH PASSWORD ''${POSTGRES_PASSWORD}'' LOGIN; END IF; END'\""

# Create databases (if not exists)
echo "--- Creating databases ---"
su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/createdb -O \"${POSTGRES_USER}\" \"${POSTGRES_DB}\"" 2>/dev/null || echo "Database ${POSTGRES_DB} already exists"

su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/createdb -O \"${POSTGRES_USER}\" \"${TRAJ_PROXY_DB}\"" 2>/dev/null || echo "Database ${TRAJ_PROXY_DB} already exists"

# ========================================
# Phase 3: Initialize traj_proxy tables
# ========================================
export DATABASE_URL=${TRAJ_PROXY_DATABASE_URL}
cd -
echo "--- Initializing traj_proxy tables ---"
python3 - <<'PYTHON_SCRIPT'
import os
import sys
import re
import psycopg

def init_database():
    """Initialize database: create tables, indexes"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set")
        sys.exit(1)

    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    db_name = parsed.path.lstrip("/")
    host = parsed.hostname
    port = parsed.port or 5432
    user = parsed.username
    password = parsed.password

    identifier_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    if not identifier_pattern.match(db_name):
        print(f"Error: Invalid database name: {db_name}")
        sys.exit(1)
    if not identifier_pattern.match(user):
        print(f"Error: Invalid user name: {user}")
        sys.exit(1)

    print("Creating tables and indexes...")
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            # request_metadata table
            print("  Creating request_metadata table...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS public.request_metadata (
                    id BIGSERIAL PRIMARY KEY,
                    unique_id TEXT NOT NULL UNIQUE,
                    request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    run_id TEXT,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cache_hit_tokens INTEGER DEFAULT 0,
                    processing_duration_ms FLOAT,
                    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    end_time TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    error TEXT,
                    archive_location TEXT,
                    archived_at TIMESTAMP WITH TIME ZONE
                )
            """)

            # Compatibility: add run_id column to request_metadata
            print("  Checking request_metadata column compatibility...")
            has_run_id = conn.execute("""
                SELECT EXISTS(SELECT 1 FROM information_schema.columns
                              WHERE table_schema='public'
                              AND table_name='request_metadata'
                              AND column_name='run_id')
            """).fetchone()[0]

            if not has_run_id:
                conn.execute("ALTER TABLE public.request_metadata ADD COLUMN run_id TEXT")
                print("  run_id column added")

            # request_metadata indexes
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_session_id_idx ON public.request_metadata (session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_run_id_idx ON public.request_metadata (run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_start_time_idx ON public.request_metadata (start_time DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_archive_location_idx ON public.request_metadata (archive_location) WHERE archive_location IS NOT NULL")

            # request_details_active partitioned table
            print("  Creating request_details_active partitioned table...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS public.request_details_active (
                    id BIGSERIAL,
                    unique_id TEXT NOT NULL
                        REFERENCES public.request_metadata(unique_id) ON DELETE CASCADE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    tokenizer_path TEXT,
                    messages JSONB NOT NULL,
                    raw_request JSONB,
                    raw_response JSONB,
                    text_request JSONB,
                    text_response JSONB,
                    prompt_text TEXT,
                    token_ids INTEGER[],
                    token_request JSONB,
                    token_response JSONB,
                    response_text TEXT,
                    response_ids INTEGER[],
                    full_conversation_text TEXT,
                    full_conversation_token_ids INTEGER[],
                    error_traceback TEXT,
                    PRIMARY KEY (id, created_at)
                ) PARTITION BY RANGE (created_at)
            """)

            # Create the current month partition
            import datetime as _dt
            _now = _dt.datetime.now()
            _month_start = _now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if _now.month == 12:
                _next_month = _month_start.replace(year=_now.year + 1, month=1)
            else:
                _next_month = _month_start.replace(month=_now.month + 1)
            _partition_name = f"request_details_active_{_now.strftime('%Y_%m')}"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS public.{_partition_name}
                    PARTITION OF public.request_details_active
                    FOR VALUES FROM ('{_month_start.isoformat()}') TO ('{_next_month.isoformat()}')
            """)

            # Default partition
            conn.execute("""
                CREATE TABLE IF NOT EXISTS public.request_details_active_default
                    PARTITION OF public.request_details_active DEFAULT
            """)

            # request_details_active indexes
            conn.execute("CREATE INDEX IF NOT EXISTS request_details_active_unique_id_idx ON public.request_details_active (unique_id)")

            # model_registry table
            print("  Creating model_registry table...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS public.model_registry (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT,
                    model_name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    tokenizer_path TEXT,
                    token_in_token_out BOOLEAN DEFAULT FALSE,
                    tool_parser TEXT NOT NULL DEFAULT '',
                    reasoning_parser TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT unique_run_model UNIQUE (run_id, model_name)
                )
            """)

            # model_registry indexes
            conn.execute("CREATE INDEX IF NOT EXISTS model_registry_run_id_idx ON public.model_registry (run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS model_registry_model_name_idx ON public.model_registry (model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS model_registry_updated_at_idx ON public.model_registry (updated_at DESC)")

            # Compatibility: add new columns to model_registry
            conn.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_schema='public' AND table_name='model_registry' AND column_name='tool_parser') THEN
                        ALTER TABLE public.model_registry ADD COLUMN tool_parser TEXT NOT NULL DEFAULT '';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                                   WHERE table_schema='public' AND table_name='model_registry' AND column_name='reasoning_parser') THEN
                        ALTER TABLE public.model_registry ADD COLUMN reasoning_parser TEXT NOT NULL DEFAULT '';
                    END IF;
                END $$;
            """)

            # Make tokenizer_path nullable
            conn.execute("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema='public'
                        AND table_name='model_registry'
                        AND column_name='tokenizer_path'
                        AND is_nullable='NO'
                    ) THEN
                        ALTER TABLE public.model_registry ALTER COLUMN tokenizer_path DROP NOT NULL;
                    END IF;
                END $$;
            """)

            print("Tables and indexes created successfully")
    except Exception as e:
        print(f"Error: Failed to create tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "Error: Table initialization failed"
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl stop -D \"${PGDATA}\"" 2>/dev/null || true
    exit 1
fi

echo "--- Table initialization complete ---"

# ========================================
# Phase 3B: Initialize LiteLLM tables (Prisma migration)
# ========================================
echo "--- Initializing LiteLLM tables ---"
LITELLM_PRISMA_DIR="${root_dir}/litellm-venv/lib/python3.11/site-packages/litellm/proxy"
if [ -f "${LITELLM_PRISMA_DIR}/schema.prisma" ]; then
    # Add litellm-venv/bin to PATH so the prisma-client-py generator can be found
    export PATH="${root_dir}/litellm-venv/bin:$PATH"

    export NODE_TLS_REJECT_UNAUTHORIZED=0
    export PRISMA_PYTHON_SKIP_NODEENV=1
    export PRISMA_CLI_BINARY=${root_dir}/node_modules/.bin/prisma

    DATABASE_URL="${LITELLM_DATABASE_URL}" \
    ${root_dir}/litellm-venv/bin/prisma db push \
        --schema "${LITELLM_PRISMA_DIR}/schema.prisma" \
        --skip-generate \
        --accept-data-loss 2>&1 || \
    echo "Warning: LiteLLM Prisma migration failed, some LiteLLM features may be unavailable"
    echo "LiteLLM table initialization complete"
else
    echo "Warning: LiteLLM Prisma schema not found, skipping migration"
fi

# ========================================
# Phase 4: Stop temporary PostgreSQL (supervisord will take over)
# ========================================
echo "--- Stopping temporary PostgreSQL ---"
su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl stop -D \"${PGDATA}\""

# ========================================
# Phase 5: Update config file database connection (overridden at runtime)
# ========================================
echo "--- Updating configuration file ---"
# Replace hardcoded values in config_allinone.yaml with runtime DATABASE_URL
CONFIG_FILE="${traj_proxy_dir}/dockers/allinone/configs/config.yaml"
if [ -f "${CONFIG_FILE}" ]; then
    sed -i "s|postgresql://[^@]*@[^/]*/${TRAJ_PROXY_DB}|${TRAJ_PROXY_DATABASE_URL}|g" "${CONFIG_FILE}"
    sed -i "s|models_dir: /app/models|models_dir: ${TRAJ_PROXY_DATA}/models|g" "${CONFIG_FILE}"
fi

# ========================================
# Phase 6: Start supervisord
# ========================================

echo "--- Init done! ---"
