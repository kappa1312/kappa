import { Terminal, Clock, FileCode, AlertCircle } from 'lucide-react';
import type { Session } from '../types';

interface SessionGridProps {
  sessions: Session[];
  onSelectSession?: (session: Session) => void;
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'text-neon-cyan border-neon-cyan/30 bg-neon-cyan/10';
    case 'active':
    case 'running':
      return 'text-neon-green border-neon-green/30 bg-neon-green/10';
    case 'failed':
      return 'text-neon-red border-neon-red/30 bg-neon-red/10';
    case 'killed':
      return 'text-neon-orange border-neon-orange/30 bg-neon-orange/10';
    default:
      return 'text-gray-400 border-gray-600/30 bg-gray-600/10';
  }
};

const formatTime = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
};

const formatDuration = (startStr: string, endStr?: string) => {
  const start = new Date(startStr);
  const end = endStr ? new Date(endStr) : new Date();
  const seconds = Math.floor((end.getTime() - start.getTime()) / 1000);

  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
};

export function SessionGrid({ sessions, onSelectSession }: SessionGridProps) {
  if (sessions.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No sessions to display</p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {sessions.map((session) => (
        <button
          key={session.id}
          onClick={() => onSelectSession?.(session)}
          className={`glass-panel-light p-4 rounded-xl text-left hover-glow border ${getStatusColor(
            session.status
          )}`}
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4" />
              <span className="font-mono text-sm truncate max-w-[150px]">
                {session.id.slice(0, 8)}...
              </span>
            </div>
            <span className={`status-orb ${session.status === 'active' || session.status === 'running' ? 'running' : session.status === 'completed' ? 'completed' : session.status === 'failed' ? 'failed' : 'pending'}`} />
          </div>

          <div className="space-y-2">
            {/* Task ID */}
            {session.task_id && (
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <FileCode className="w-3 h-3" />
                <span className="truncate">Task: {session.task_id.slice(0, 8)}...</span>
              </div>
            )}

            {/* Timing */}
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1 text-gray-400">
                <Clock className="w-3 h-3" />
                <span>{formatTime(session.started_at)}</span>
              </div>
              <span className="font-mono text-gray-300">
                {formatDuration(session.started_at, session.completed_at)}
              </span>
            </div>

            {/* Files modified */}
            {session.files_modified && session.files_modified.length > 0 && (
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <FileCode className="w-3 h-3" />
                <span>{session.files_modified.length} files modified</span>
              </div>
            )}

            {/* Error indicator */}
            {session.error && (
              <div className="flex items-center gap-1 text-xs text-neon-red">
                <AlertCircle className="w-3 h-3" />
                <span className="truncate">{session.error.slice(0, 50)}...</span>
              </div>
            )}
          </div>

          {/* Status badge */}
          <div className="mt-3 pt-3 border-t border-white/10">
            <span
              className={`text-xs font-medium uppercase ${
                session.status === 'completed'
                  ? 'text-neon-cyan'
                  : session.status === 'active' || session.status === 'running'
                  ? 'text-neon-green'
                  : session.status === 'failed'
                  ? 'text-neon-red'
                  : 'text-gray-400'
              }`}
            >
              {session.status}
            </span>
          </div>
        </button>
      ))}
    </div>
  );
}

export default SessionGrid;
