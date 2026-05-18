#!/bin/bash
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
# Default values for environment variables
# ========================================
PG_VERSION=14
PGDATA="${PGDATA:-${TRAJ_PROXY_DATA}/postgresql}"
POSTGRES_USER="${POSTGRES_USER:-llmproxy}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-dbpassword9090}"
TRAJ_PROXY_DB="${TRAJ_PROXY_DB:-traj_proxy}"
POSTGRES_DB="${POSTGRES_DB:-litellm}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-1234}"
LITELLM_SALT_KEY="${LITELLM_SALT_KEY:-sk-1234}"

# Automatically generate database connection URL (user setting takes precedence)
export TRAJ_PROXY_DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${TRAJ_PROXY_DB}}"
export LITELLM_DATABASE_URL="${LITELLM_DATABASE_URL:-postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${POSTGRES_DB}}"
export STORE_MODEL_IN_DB="${STORE_MODEL_IN_DB:-True}"

# ========================================
# Full initialization mode (container entry)
# ========================================
echo "=== TrajProxy All-in-One 容器启动 ==="

# ========================================
# Phase One: PostgreSQL Data Directory Initialization
# ========================================
if [ ! -f "${PGDATA}/PG_VERSION" ]; then
    echo "--- 初始化 PostgreSQL 数据目录 ---"
    mkdir -p "${PGDATA}"
    chown postgres:postgres "${PGDATA}"
    cd "${PGDATA}"

    # 初始化数据库集群
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/initdb -D \"${PGDATA}\" -E UTF8 --locale=C --auth=trust"

    # 配置 pg_hba.conf 允许本地连接
    cat > "${PGDATA}/pg_hba.conf" <<EOF
# 本地连接（all-in-one 模式）
local   all             all                                     trust
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
EOF

    echo "PostgreSQL 数据目录初始化完成"
else
    echo "PostgreSQL 数据目录已存在，跳过初始化"
fi

chown -R postgres:postgres "${PGDATA}"
cd "${PGDATA}"

# ========================================
# Phase 2: Temporarily start PostgreSQL, create users and databases
# ========================================
echo "--- 启动 PostgreSQL 进行初始化 ---"
if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_isready -q"; then
    echo "PostgreSQL 服务已在运行"
else
    echo "PostgreSQL 服务未运行，正在启动..."
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl start -D \"${PGDATA}\" -l ${TRAJ_PROXY_DATA}/logs/postgresql_init.log -w -t 60"
fi

# Waiting for PostgreSQL to accept connections
echo "--- 等待 PostgreSQL 就绪 ---"
for i in $(seq 1 30); do
    if su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/psql -U postgres -c '\q' 2>/dev/null"; then
        echo "PostgreSQL 已就绪"
        break
    fi
    sleep 1
done

echo "--- 创建数据库用户 ---"
su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/psql -c \"DO 'BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = ''${POSTGRES_USER}'') THEN CREATE USER ${POSTGRES_USER} WITH PASSWORD ''${POSTGRES_PASSWORD}'' LOGIN; END IF; END'\""

echo "--- 创建数据库 ---"
su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/createdb -O \"${POSTGRES_USER}\" \"${POSTGRES_DB}\"" 2>/dev/null || echo "数据库 ${POSTGRES_DB} 已存在"

su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/createdb -O \"${POSTGRES_USER}\" \"${TRAJ_PROXY_DB}\"" 2>/dev/null || echo "数据库 ${TRAJ_PROXY_DB} 已存在"

# ========================================
# Phase 3: Initialize the traj_proxy data table
# ========================================
export DATABASE_URL=${TRAJ_PROXY_DATABASE_URL}
cd -
echo "--- 初始化 traj_proxy 数据表 ---"
python3 - <<'PYTHON_SCRIPT'
import os
import sys
import re
import psycopg

def init_database():
    """初始化数据库：创建表、索引"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("错误: 未设置 DATABASE_URL 环境变量")
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
        print(f"错误: 无效的数据库名称: {db_name}")
        sys.exit(1)
    if not identifier_pattern.match(user):
        print(f"错误: 无效的用户名称: {user}")
        sys.exit(1)

    print("创建表和索引...")
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            # request_metadata 表
            print("  创建 request_metadata 表...")
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

            # 兼容性：为 request_metadata 添加 run_id 列
            print("  检查 request_metadata 表列兼容性...")
            has_run_id = conn.execute("""
                SELECT EXISTS(SELECT 1 FROM information_schema.columns
                              WHERE table_schema='public'
                              AND table_name='request_metadata'
                              AND column_name='run_id')
            """).fetchone()[0]

            if not has_run_id:
                conn.execute("ALTER TABLE public.request_metadata ADD COLUMN run_id TEXT")
                print("  run_id 列添加完成")

            # request_metadata 索引
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_session_id_idx ON public.request_metadata (session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_run_id_idx ON public.request_metadata (run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_start_time_idx ON public.request_metadata (start_time DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS request_metadata_archive_location_idx ON public.request_metadata (archive_location) WHERE archive_location IS NOT NULL")

            # request_details_active 分区表
            print("  创建 request_details_active 分区表...")
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

            # 创建当月分区
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

            # 默认分区
            conn.execute("""
                CREATE TABLE IF NOT EXISTS public.request_details_active_default
                    PARTITION OF public.request_details_active DEFAULT
            """)

            # request_details_active 索引
            conn.execute("CREATE INDEX IF NOT EXISTS request_details_active_unique_id_idx ON public.request_details_active (unique_id)")

            # model_registry 表
            print("  创建 model_registry 表...")
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

            # model_registry 索引
            conn.execute("CREATE INDEX IF NOT EXISTS model_registry_run_id_idx ON public.model_registry (run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS model_registry_model_name_idx ON public.model_registry (model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS model_registry_updated_at_idx ON public.model_registry (updated_at DESC)")

            # 兼容性：为 model_registry 添加新列
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

            # tokenizer_path 改为可空
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

            print("表和索引创建完成")
    except Exception as e:
        print(f"错误: 创建表失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "错误: 数据表初始化失败"
    su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl stop -D \"${PGDATA}\"" 2>/dev/null || true
    exit 1
fi

echo "--- 数据表初始化完成 ---"

# ========================================
# Phase 3B: Initialize LiteLLM Data Tables (Prisma Migration)
# ========================================
echo "--- 初始化 LiteLLM 数据表 ---"
LITELLM_PRISMA_DIR="${root_dir}/litellm-venv/lib/python3.11/site-packages/litellm/proxy"
if [ -f "${LITELLM_PRISMA_DIR}/schema.prisma" ]; then
    export PATH="${root_dir}/litellm-venv/bin:$PATH"

    export HTTP_PROXY=http://10.50.113.120:3128
    export HTTPS_PROXY=http://10.50.113.120:3128
    export no_proxy="localhost,127.0.0.1,*.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.huawei.com"
    export NO_PROXY="$no_proxy"
    export NODE_TLS_REJECT_UNAUTHORIZED=0
    export PRISMA_PYTHON_SKIP_NODEENV=1
    export PRISMA_CLI_BINARY=${root_dir}/node_modules/.bin/prisma

    DATABASE_URL="${LITELLM_DATABASE_URL}" \
    ${root_dir}/litellm-venv/bin/prisma db push \
        --schema "${LITELLM_PRISMA_DIR}/schema.prisma" \
        --skip-generate \
        --accept-data-loss 2>&1 || \
    echo "警告: LiteLLM Prisma 迁移失败，LiteLLM 部分功能可能不可用"
    echo "LiteLLM 数据表初始化完成"
else
    echo "警告: 未找到 LiteLLM Prisma schema，跳过迁移"
fi

# ========================================
# Stage 4: Stop the temporary PostgreSQL (supervisord will take over)
# ========================================
echo "--- Stopping temporary PostgreSQL ---"
su postgres -c "/usr/lib/postgresql/${PG_VERSION}/bin/pg_ctl stop -D \"${PGDATA}\""

# ========================================
# Stage 5: Update database connections in configuration files (runtime override)
# ========================================
echo "--- Updating configuration files ---"
# Replace hardcoded values in config_allinone.yaml with runtime DATABASE_URL
CONFIG_FILE="${traj_proxy_dir}/dockers/allinone/configs/config.yaml"
if [ -f "${CONFIG_FILE}" ]; then
    sed -i "s|postgresql://[^@]*@[^/]*/${TRAJ_PROXY_DB}|${TRAJ_PROXY_DATABASE_URL}|g" "${CONFIG_FILE}"
    sed -i "s|models_dir: /app/models|models_dir: ${TRAJ_PROXY_DATA}/models|g" "${CONFIG_FILE}"
fi

# ========================================
# Stage 6: Start supervisord
# ========================================
#echo "=== Start supervisord (managing all services) ==="
#exec /usr/bin/supervisord -c /etc/supervisor/conf.d/trajproxy.conf
