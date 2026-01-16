import { ChevronDown, ChevronRight, Circle, CheckCircle, XCircle, Loader } from 'lucide-react';
import { useState } from 'react';
import type { Wave, WaveTask } from '../types';

interface WaveTimelineProps {
  waves: Wave[];
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="w-4 h-4 text-neon-cyan" />;
    case 'running':
    case 'executing':
      return <Loader className="w-4 h-4 text-neon-green animate-spin" />;
    case 'failed':
      return <XCircle className="w-4 h-4 text-neon-red" />;
    default:
      return <Circle className="w-4 h-4 text-gray-500" />;
  }
};

const getWaveColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'border-neon-cyan bg-neon-cyan/10';
    case 'executing':
      return 'border-neon-green bg-neon-green/10';
    case 'failed':
      return 'border-neon-red bg-neon-red/10';
    default:
      return 'border-gray-600 bg-gray-600/10';
  }
};

const getTaskStatusColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'text-neon-cyan';
    case 'running':
      return 'text-neon-green';
    case 'failed':
      return 'text-neon-red';
    case 'skipped':
      return 'text-gray-500';
    default:
      return 'text-neon-orange';
  }
};

function WaveItem({ wave, isLast }: { wave: Wave; isLast: boolean }) {
  const [isExpanded, setIsExpanded] = useState(wave.status === 'executing');
  const progress = wave.total_tasks > 0
    ? Math.round((wave.completed_tasks / wave.total_tasks) * 100)
    : 0;

  return (
    <div className="relative">
      {/* Connector line */}
      {!isLast && (
        <div className="wave-connector" />
      )}

      {/* Wave node */}
      <div className="flex gap-4">
        {/* Circle indicator */}
        <div
          className={`w-10 h-10 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${getWaveColor(
            wave.status
          )}`}
        >
          <span className="text-sm font-bold">{wave.id}</span>
        </div>

        {/* Wave content */}
        <div className="flex-1 pb-6">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full glass-panel-light p-4 rounded-xl hover:bg-white/10 transition-colors text-left"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                )}
                <span className="font-medium text-white">Wave {wave.id}</span>
                <span className={`text-xs uppercase ${getTaskStatusColor(wave.status)}`}>
                  {wave.status}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-400">
                  {wave.completed_tasks}/{wave.total_tasks} tasks
                </span>
                <span className="text-sm font-mono text-white">{progress}%</span>
              </div>
            </div>

            {/* Progress bar */}
            <div className="metric-bar mt-3">
              <div
                className={`metric-bar-fill ${
                  wave.status === 'completed'
                    ? 'bg-neon-cyan'
                    : wave.status === 'failed'
                    ? 'bg-neon-red'
                    : wave.status === 'executing'
                    ? 'bg-neon-green'
                    : 'bg-gray-600'
                }`}
                style={{ width: `${progress}%` }}
              />
            </div>
          </button>

          {/* Expanded tasks */}
          {isExpanded && wave.tasks && wave.tasks.length > 0 && (
            <div className="mt-2 ml-8 space-y-1">
              {wave.tasks.map((task: WaveTask) => (
                <div
                  key={task.id}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg bg-black/20"
                >
                  {getStatusIcon(task.status)}
                  <span className="text-sm text-gray-300 flex-1 truncate">
                    {task.name}
                  </span>
                  <span className="text-xs text-gray-500">{task.category}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function WaveTimeline({ waves }: WaveTimelineProps) {
  if (waves.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <Circle className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No waves to display</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-0">
      {waves.map((wave, index) => (
        <WaveItem
          key={wave.id}
          wave={wave}
          isLast={index === waves.length - 1}
        />
      ))}
    </div>
  );
}

export default WaveTimeline;
