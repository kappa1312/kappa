import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Clock, Layers, AlertTriangle } from 'lucide-react';
import Header from './components/Header';
import ProjectSidebar from './components/ProjectSidebar';
import MetricsPanel from './components/MetricsPanel';
import ProgressRing from './components/ProgressRing';
import TabNavigation from './components/TabNavigation';
import WaveTimeline from './components/WaveTimeline';
import SessionGrid from './components/SessionGrid';
import LogViewer from './components/LogViewer';
import useWebSocket from './hooks/useWebSocket';
import useProjects from './hooks/useProjects';
import useMetrics from './hooks/useMetrics';
import type { TabType, Session } from './types';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [sessions, setSessions] = useState<Session[]>([]);

  const { isConnected, lastMessage, subscribe, unsubscribe } = useWebSocket();
  const {
    projects,
    selectedProject,
    waves,
    chatHistory,
    loading: projectsLoading,
    selectProject,
    refreshProjects,
  } = useProjects();
  const { systemMetrics, performanceMetrics, executionMetrics } = useMetrics();

  // Subscribe to selected project for real-time updates
  useEffect(() => {
    if (selectedProject) {
      subscribe(selectedProject.id);
      return () => unsubscribe(selectedProject.id);
    }
  }, [selectedProject, subscribe, unsubscribe]);

  // Fetch sessions when project is selected
  useEffect(() => {
    if (selectedProject) {
      fetch(`/api/projects/${selectedProject.id}/sessions`)
        .then((res) => res.json())
        .then(setSessions)
        .catch(() => setSessions([]));
    } else {
      setSessions([]);
    }
  }, [selectedProject]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      // Refresh data based on message type
      if (
        lastMessage.type === 'task_update' ||
        lastMessage.type === 'wave_update' ||
        lastMessage.type === 'project_update'
      ) {
        refreshProjects();
      }
    }
  }, [lastMessage, refreshProjects]);

  return (
    <div className="min-h-screen grid-bg relative">
      {/* Animated background orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      {/* Main layout */}
      <div className="relative z-10 flex flex-col h-screen p-4 gap-4">
        {/* Header */}
        <Header isConnected={isConnected} />

        {/* Main content */}
        <div className="flex-1 grid grid-cols-[280px_1fr_280px] gap-4 min-h-0">
          {/* Left sidebar - Projects */}
          <ProjectSidebar
            projects={projects}
            selectedProjectId={selectedProject?.id ?? null}
            onSelectProject={selectProject}
            loading={projectsLoading}
            onRefresh={refreshProjects}
          />

          {/* Center - Main content */}
          <main className="glass-panel p-6 flex flex-col min-h-0 overflow-hidden">
            {selectedProject ? (
              <>
                {/* Project header */}
                <div className="flex items-start gap-8 mb-6">
                  <ProgressRing
                    progress={selectedProject.progress}
                    status={selectedProject.status}
                  />
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold holographic-text mb-2">
                      {selectedProject.name}
                    </h2>
                    <p className="text-gray-400 text-sm mb-4">
                      {selectedProject.specification?.slice(0, 150) || 'No description'}
                      {(selectedProject.specification?.length ?? 0) > 150 && '...'}
                    </p>

                    {/* Quick stats */}
                    <div className="grid grid-cols-4 gap-4">
                      <div className="glass-panel-light p-3 rounded-xl text-center">
                        <CheckCircle className="w-5 h-5 text-neon-cyan mx-auto mb-1" />
                        <div className="text-lg font-bold text-white">
                          {selectedProject.completed_tasks}
                        </div>
                        <div className="text-xs text-gray-400">Completed</div>
                      </div>
                      <div className="glass-panel-light p-3 rounded-xl text-center">
                        <Layers className="w-5 h-5 text-neon-purple mx-auto mb-1" />
                        <div className="text-lg font-bold text-white">
                          {selectedProject.total_tasks}
                        </div>
                        <div className="text-xs text-gray-400">Total</div>
                      </div>
                      <div className="glass-panel-light p-3 rounded-xl text-center">
                        <XCircle className="w-5 h-5 text-neon-red mx-auto mb-1" />
                        <div className="text-lg font-bold text-white">
                          {selectedProject.failed_tasks}
                        </div>
                        <div className="text-xs text-gray-400">Failed</div>
                      </div>
                      <div className="glass-panel-light p-3 rounded-xl text-center">
                        <Clock className="w-5 h-5 text-neon-orange mx-auto mb-1" />
                        <div className="text-lg font-bold text-white">
                          {selectedProject.pending_tasks}
                        </div>
                        <div className="text-xs text-gray-400">Pending</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Tabs */}
                <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

                {/* Tab content */}
                <div className="flex-1 mt-6 overflow-auto">
                  {activeTab === 'overview' && (
                    <OverviewTab project={selectedProject} waves={waves} />
                  )}
                  {activeTab === 'waves' && <WaveTimeline waves={waves} />}
                  {activeTab === 'sessions' && <SessionGrid sessions={sessions} />}
                  {activeTab === 'logs' && <LogViewer messages={chatHistory} />}
                </div>
              </>
            ) : (
              <EmptyState />
            )}
          </main>

          {/* Right sidebar - Metrics */}
          <MetricsPanel
            systemMetrics={systemMetrics}
            performanceMetrics={performanceMetrics}
            executionMetrics={executionMetrics}
          />
        </div>
      </div>
    </div>
  );
}

// Overview tab component
function OverviewTab({
  project,
  waves,
}: {
  project: NonNullable<ReturnType<typeof useProjects>['selectedProject']>;
  waves: ReturnType<typeof useProjects>['waves'];
}) {
  const completedWaves = waves.filter((w) => w.status === 'completed').length;
  const currentWave = waves.find((w) => w.status === 'executing');

  return (
    <div className="space-y-6">
      {/* Status banner */}
      <div
        className={`p-4 rounded-xl flex items-center gap-4 ${
          project.status === 'completed'
            ? 'bg-neon-cyan/10 border border-neon-cyan/30'
            : project.status === 'failed'
            ? 'bg-neon-red/10 border border-neon-red/30'
            : project.status === 'running' || project.status === 'executing'
            ? 'bg-neon-green/10 border border-neon-green/30'
            : 'bg-neon-orange/10 border border-neon-orange/30'
        }`}
      >
        {project.status === 'completed' ? (
          <CheckCircle className="w-6 h-6 text-neon-cyan" />
        ) : project.status === 'failed' ? (
          <XCircle className="w-6 h-6 text-neon-red" />
        ) : project.status === 'running' || project.status === 'executing' ? (
          <div className="status-orb running w-6 h-6" />
        ) : (
          <Clock className="w-6 h-6 text-neon-orange" />
        )}
        <div>
          <div className="font-medium text-white">
            {project.status === 'completed'
              ? 'Project Completed Successfully'
              : project.status === 'failed'
              ? 'Project Failed'
              : project.status === 'running' || project.status === 'executing'
              ? 'Project Running'
              : 'Project Pending'}
          </div>
          <div className="text-sm text-gray-400">
            {currentWave
              ? `Currently executing Wave ${currentWave.id}`
              : `${completedWaves}/${waves.length} waves completed`}
          </div>
        </div>
      </div>

      {/* Wave progress overview */}
      <div className="grid grid-cols-2 gap-4">
        <div className="glass-panel-light p-4 rounded-xl">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Wave Progress</h3>
          <div className="space-y-2">
            {waves.slice(0, 5).map((wave) => (
              <div key={wave.id} className="flex items-center gap-3">
                <span className="text-xs font-mono text-gray-400 w-12">
                  Wave {wave.id}
                </span>
                <div className="flex-1 metric-bar">
                  <div
                    className={`metric-bar-fill ${
                      wave.status === 'completed'
                        ? 'bg-neon-cyan'
                        : wave.status === 'executing'
                        ? 'bg-neon-green'
                        : wave.status === 'failed'
                        ? 'bg-neon-red'
                        : 'bg-gray-600'
                    }`}
                    style={{
                      width: `${
                        wave.total_tasks > 0
                          ? (wave.completed_tasks / wave.total_tasks) * 100
                          : 0
                      }%`,
                    }}
                  />
                </div>
                <span className="text-xs text-gray-400 w-10 text-right">
                  {wave.completed_tasks}/{wave.total_tasks}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel-light p-4 rounded-xl">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Task Breakdown</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Completed</span>
              <span className="text-sm font-mono text-neon-cyan">
                {project.completed_tasks}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Running</span>
              <span className="text-sm font-mono text-neon-green">
                {project.running_tasks}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Pending</span>
              <span className="text-sm font-mono text-neon-orange">
                {project.pending_tasks}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Failed</span>
              <span className="text-sm font-mono text-neon-red">
                {project.failed_tasks}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Error message if failed */}
      {project.error && (
        <div className="p-4 rounded-xl bg-neon-red/10 border border-neon-red/30">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-neon-red" />
            <span className="font-medium text-neon-red">Error</span>
          </div>
          <p className="text-sm text-gray-300 font-mono">{project.error}</p>
        </div>
      )}

      {/* Final output if completed */}
      {project.final_output && (
        <div className="p-4 rounded-xl bg-neon-cyan/10 border border-neon-cyan/30">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5 text-neon-cyan" />
            <span className="font-medium text-neon-cyan">Output</span>
          </div>
          <p className="text-sm text-gray-300 font-mono whitespace-pre-wrap">
            {project.final_output}
          </p>
        </div>
      )}
    </div>
  );
}

// Empty state when no project selected
function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center max-w-md">
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-neon-cyan/20 to-neon-purple/20 flex items-center justify-center mx-auto mb-6">
          <Layers className="w-10 h-10 text-neon-cyan" />
        </div>
        <h3 className="text-xl font-bold holographic-text mb-3">Select a Project</h3>
        <p className="text-gray-400">
          Choose a project from the sidebar to view its details, progress, and execution
          logs.
        </p>
      </div>
    </div>
  );
}

export default App;
