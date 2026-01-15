-- Kappa OS PostgreSQL Schema Setup
-- Run with: psql -d kappa_db -f scripts/setup_db.sql

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom enum types
DO $$ BEGIN
    CREATE TYPE project_status AS ENUM (
        'pending', 'decomposing', 'running', 'resolving_conflicts', 'completed', 'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE task_status AS ENUM (
        'pending', 'running', 'completed', 'failed', 'skipped'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE session_status AS ENUM (
        'starting', 'running', 'completed', 'failed', 'timeout'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    specification TEXT NOT NULL,
    project_path VARCHAR(1024) NOT NULL,
    status project_status DEFAULT 'pending',
    config JSONB DEFAULT '{}',
    final_output TEXT,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(50) DEFAULT 'business_logic',
    complexity VARCHAR(20) DEFAULT 'medium',
    wave INTEGER DEFAULT 0,
    status task_status DEFAULT 'pending',
    dependencies JSONB DEFAULT '[]',
    file_targets JSONB DEFAULT '[]',
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    status session_status DEFAULT 'starting',
    output TEXT,
    error TEXT,
    files_modified JSONB DEFAULT '[]',
    token_usage JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Context snapshots table
CREATE TABLE IF NOT EXISTS context_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    context_type VARCHAR(50) NOT NULL,
    key VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    snapshot_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Decisions table
CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    category VARCHAR(50) NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT,
    alternatives_considered JSONB DEFAULT '[]',
    made_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conflicts table
CREATE TABLE IF NOT EXISTS conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path VARCHAR(1024) NOT NULL,
    session_a_id VARCHAR(36) NOT NULL,
    session_b_id VARCHAR(36) NOT NULL,
    conflict_type VARCHAR(50) DEFAULT 'merge',
    description TEXT NOT NULL,
    content_a TEXT,
    content_b TEXT,
    resolution TEXT,
    resolved_by VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS ix_tasks_project_wave ON tasks(project_id, wave);
CREATE INDEX IF NOT EXISTS ix_tasks_project_status ON tasks(project_id, status);
CREATE INDEX IF NOT EXISTS ix_sessions_project_status ON sessions(project_id, status);
CREATE INDEX IF NOT EXISTS ix_context_session_type ON context_snapshots(session_id, context_type);
CREATE INDEX IF NOT EXISTS ix_decisions_project_category ON decisions(project_id, category);
CREATE INDEX IF NOT EXISTS ix_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS ix_projects_created_at ON projects(created_at DESC);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO kappa_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO kappa_user;

-- Verification
SELECT 'Schema setup complete!' as status;
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
