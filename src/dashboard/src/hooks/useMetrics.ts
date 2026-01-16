import { useState, useEffect, useCallback } from 'react';
import type { SystemMetrics, PerformanceMetrics, ExecutionMetrics } from '../types';

interface UseMetricsReturn {
  systemMetrics: SystemMetrics | null;
  performanceMetrics: PerformanceMetrics | null;
  executionMetrics: ExecutionMetrics | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

const defaultSystemMetrics: SystemMetrics = {
  total_projects: 0,
  active_projects: 0,
  completed_projects: 0,
  failed_projects: 0,
  total_tasks: 0,
  total_sessions: 0,
  total_conflicts: 0,
  resolved_conflicts: 0,
};

const defaultPerformanceMetrics: PerformanceMetrics = {
  active_connections: 0,
  cache_stats: {},
  database_status: 'unknown',
  uptime_seconds: 0,
};

const defaultExecutionMetrics: ExecutionMetrics = {
  avg_task_duration_seconds: 0,
  avg_session_duration_seconds: 0,
  avg_project_duration_seconds: 0,
  tasks_per_hour: 0,
  parallelism_factor: 1,
};

export function useMetrics(): UseMetricsReturn {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(defaultSystemMetrics);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(defaultPerformanceMetrics);
  const [executionMetrics, setExecutionMetrics] = useState<ExecutionMetrics | null>(defaultExecutionMetrics);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const [systemRes, perfRes, execRes] = await Promise.all([
        fetch('/api/metrics/'),
        fetch('/api/metrics/performance'),
        fetch('/api/metrics/execution'),
      ]);

      if (systemRes.ok) {
        const data = await systemRes.json();
        setSystemMetrics(data);
      }

      if (perfRes.ok) {
        const data = await perfRes.json();
        setPerformanceMetrics(data);
      }

      if (execRes.ok) {
        const data = await execRes.json();
        setExecutionMetrics(data);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load and refresh every 2 seconds
  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  const refresh = useCallback(async () => {
    setLoading(true);
    await fetchMetrics();
  }, [fetchMetrics]);

  return {
    systemMetrics,
    performanceMetrics,
    executionMetrics,
    loading,
    error,
    refresh,
  };
}

export default useMetrics;
