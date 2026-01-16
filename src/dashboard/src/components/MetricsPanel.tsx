import { Activity, Clock, Database, Layers, Zap } from 'lucide-react';
import type { SystemMetrics, PerformanceMetrics, ExecutionMetrics } from '../types';

interface MetricsPanelProps {
  systemMetrics: SystemMetrics | null;
  performanceMetrics: PerformanceMetrics | null;
  executionMetrics: ExecutionMetrics | null;
}

const formatUptime = (seconds: number) => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const formatDuration = (seconds: number) => {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
};

export function MetricsPanel({
  systemMetrics,
  performanceMetrics,
  executionMetrics,
}: MetricsPanelProps) {
  const maxSessions = 10;
  const activeSessions = performanceMetrics?.active_connections ?? 0;

  return (
    <aside className="glass-panel p-4 flex flex-col h-full space-y-4">
      <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
        System Metrics
      </h2>

      {/* Active Sessions */}
      <div className="glass-panel-light p-3 rounded-xl">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-neon-green" />
            <span className="text-sm text-gray-300">Active Sessions</span>
          </div>
          <span className="text-sm font-mono text-neon-green">
            {activeSessions}/{maxSessions}
          </span>
        </div>
        <div className="metric-bar">
          <div
            className="metric-bar-fill bg-neon-green progress-glow"
            style={{ width: `${(activeSessions / maxSessions) * 100}%` }}
          />
        </div>
      </div>

      {/* Database Status */}
      <div className="glass-panel-light p-3 rounded-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-neon-cyan" />
            <span className="text-sm text-gray-300">Database</span>
          </div>
          <span
            className={`text-sm font-medium ${
              performanceMetrics?.database_status === 'healthy'
                ? 'text-neon-green'
                : 'text-neon-red'
            }`}
          >
            {performanceMetrics?.database_status?.toUpperCase() ?? 'UNKNOWN'}
          </span>
        </div>
      </div>

      {/* Uptime */}
      <div className="glass-panel-light p-3 rounded-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-neon-purple" />
            <span className="text-sm text-gray-300">Uptime</span>
          </div>
          <span className="text-sm font-mono text-neon-purple">
            {formatUptime(performanceMetrics?.uptime_seconds ?? 0)}
          </span>
        </div>
      </div>

      {/* Execution Metrics */}
      <div className="glass-panel-light p-3 rounded-xl space-y-3">
        <div className="flex items-center gap-2 mb-2">
          <Zap className="w-4 h-4 text-neon-orange" />
          <span className="text-sm text-gray-300">Performance</span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-400">Avg Task Duration</span>
          <span className="font-mono text-white">
            {formatDuration(executionMetrics?.avg_task_duration_seconds ?? 0)}
          </span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-400">Tasks/Hour</span>
          <span className="font-mono text-white">
            {(executionMetrics?.tasks_per_hour ?? 0).toFixed(1)}
          </span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-400">Parallelism</span>
          <span className="font-mono text-white">
            {(executionMetrics?.parallelism_factor ?? 1).toFixed(1)}x
          </span>
        </div>
      </div>

      {/* Today's Stats */}
      <div className="flex-1" />
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Today's Activity
        </h3>
        <div className="grid grid-cols-2 gap-2">
          <div className="glass-panel-light p-3 rounded-xl text-center">
            <Layers className="w-5 h-5 text-neon-cyan mx-auto mb-1" />
            <div className="text-lg font-bold text-white">
              {systemMetrics?.total_projects ?? 0}
            </div>
            <div className="text-xs text-gray-400">Projects</div>
          </div>
          <div className="glass-panel-light p-3 rounded-xl text-center">
            <Activity className="w-5 h-5 text-neon-purple mx-auto mb-1" />
            <div className="text-lg font-bold text-white">
              {systemMetrics?.total_tasks ?? 0}
            </div>
            <div className="text-xs text-gray-400">Tasks</div>
          </div>
        </div>
      </div>

      {/* Conflicts */}
      {(systemMetrics?.total_conflicts ?? 0) > 0 && (
        <div className="glass-panel-light p-3 rounded-xl border border-neon-orange/30">
          <div className="flex items-center justify-between text-xs">
            <span className="text-neon-orange">Conflicts</span>
            <span className="font-mono text-white">
              {systemMetrics?.resolved_conflicts ?? 0}/{systemMetrics?.total_conflicts ?? 0}{' '}
              resolved
            </span>
          </div>
        </div>
      )}
    </aside>
  );
}

export default MetricsPanel;
