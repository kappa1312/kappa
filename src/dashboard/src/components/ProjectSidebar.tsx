import { Folder, FolderOpen, RefreshCw } from 'lucide-react';
import type { Project } from '../types';

interface ProjectSidebarProps {
  projects: Project[];
  selectedProjectId: string | null;
  onSelectProject: (projectId: string | null) => void;
  loading: boolean;
  onRefresh: () => void;
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'text-neon-cyan';
    case 'running':
    case 'executing':
      return 'text-neon-green';
    case 'failed':
      return 'text-neon-red';
    case 'pending':
    case 'parsing':
    case 'planning':
      return 'text-neon-orange';
    default:
      return 'text-gray-400';
  }
};

const getStatusOrb = (status: string) => {
  switch (status) {
    case 'completed':
      return 'completed';
    case 'running':
    case 'executing':
      return 'running';
    case 'failed':
      return 'failed';
    default:
      return 'pending';
  }
};

export function ProjectSidebar({
  projects,
  selectedProjectId,
  onSelectProject,
  loading,
  onRefresh,
}: ProjectSidebarProps) {
  return (
    <aside className="glass-panel p-4 flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Projects
        </h2>
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
          title="Refresh projects"
        >
          <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Projects list */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {projects.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Folder className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No projects yet</p>
          </div>
        ) : (
          projects.map((project) => {
            const isSelected = project.id === selectedProjectId;
            return (
              <button
                key={project.id}
                onClick={() => onSelectProject(isSelected ? null : project.id)}
                className={`w-full p-3 rounded-xl text-left transition-all hover-glow ${
                  isSelected
                    ? 'bg-gradient-to-r from-neon-cyan/20 to-neon-purple/20 border border-neon-cyan/30'
                    : 'glass-panel-light hover:bg-white/10'
                }`}
              >
                <div className="flex items-start gap-3">
                  {isSelected ? (
                    <FolderOpen className="w-5 h-5 text-neon-cyan flex-shrink-0 mt-0.5" />
                  ) : (
                    <Folder className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`status-orb ${getStatusOrb(project.status)}`} />
                      <span className="font-medium text-white truncate">
                        {project.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-xs ${getStatusColor(project.status)}`}>
                        {project.status.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">
                        {project.completed_tasks}/{project.total_tasks} tasks
                      </span>
                    </div>
                    {/* Progress bar */}
                    <div className="metric-bar mt-2">
                      <div
                        className={`metric-bar-fill ${
                          project.status === 'completed'
                            ? 'bg-neon-cyan'
                            : project.status === 'failed'
                            ? 'bg-neon-red'
                            : 'bg-neon-green'
                        }`}
                        style={{ width: `${project.progress}%` }}
                      />
                    </div>
                  </div>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* Stats footer */}
      <div className="mt-4 pt-4 border-t border-white/10">
        <div className="grid grid-cols-2 gap-2 text-center">
          <div className="glass-panel-light p-2 rounded-lg">
            <div className="text-lg font-bold text-neon-cyan">{projects.length}</div>
            <div className="text-xs text-gray-400">Total</div>
          </div>
          <div className="glass-panel-light p-2 rounded-lg">
            <div className="text-lg font-bold text-neon-green">
              {projects.filter((p) => p.status === 'running' || p.status === 'executing').length}
            </div>
            <div className="text-xs text-gray-400">Active</div>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default ProjectSidebar;
