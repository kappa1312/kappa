export interface Project {
  id: string;
  name: string;
  status: 'pending' | 'parsing' | 'planning' | 'executing' | 'validating' | 'completed' | 'failed' | 'running';
  progress: number;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  running_tasks: number;
  pending_tasks: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  specification?: string;
  project_path?: string;
  config?: Record<string, unknown>;
  final_output?: string;
  error?: string;
}

export interface Wave {
  id: number;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  total_tasks: number;
  completed_tasks: number;
  tasks: WaveTask[];
}

export interface WaveTask {
  id: string;
  name: string;
  status: string;
  category: string;
}

export interface Task {
  id: string;
  project_id: string;
  name: string;
  description: string;
  category: string;
  complexity: string;
  wave: number;
  status: string;
  dependencies: string[];
  file_targets: string[];
  session_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: Record<string, unknown>;
  error?: string;
}

export interface Session {
  id: string;
  project_id: string;
  task_id?: string;
  status: 'active' | 'completed' | 'failed' | 'killed' | 'running';
  files_modified: string[];
  token_usage: Record<string, unknown>;
  started_at: string;
  completed_at?: string;
  output?: string;
  error?: string;
  metrics?: Record<string, unknown>;
}

export interface SystemMetrics {
  total_projects: number;
  active_projects: number;
  completed_projects: number;
  failed_projects: number;
  total_tasks: number;
  total_sessions: number;
  total_conflicts: number;
  resolved_conflicts: number;
}

export interface PerformanceMetrics {
  active_connections: number;
  cache_stats: Record<string, unknown>;
  database_status: string;
  uptime_seconds: number;
}

export interface ExecutionMetrics {
  avg_task_duration_seconds: number;
  avg_session_duration_seconds: number;
  avg_project_duration_seconds: number;
  tasks_per_hour: number;
  parallelism_factor: number;
}

export interface ProjectMetrics {
  project_id: string;
  task_metrics: Record<string, number>;
  wave_metrics: Array<{
    wave: number;
    total_tasks: number;
    completed_tasks: number;
    progress: number;
  }>;
  session_metrics: Record<string, unknown>;
  conflict_metrics: Record<string, number>;
  timing_metrics: Record<string, number>;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

export interface TaskLogEntry {
  timestamp: string;
  level: string;
  message: string;
}

export interface WebSocketMessage {
  type: string;
  project_id?: string;
  task_id?: string;
  session_id?: string;
  wave_id?: number;
  conflict_id?: string;
  status?: string;
  progress?: number;
  output?: string;
  message?: string;
  conflict_type?: string;
  description?: string;
}

export type TabType = 'overview' | 'waves' | 'sessions' | 'logs';
