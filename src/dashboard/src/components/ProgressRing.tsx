interface ProgressRingProps {
  progress: number;
  size?: number;
  strokeWidth?: number;
  status?: string;
}

const getColor = (status: string) => {
  switch (status) {
    case 'completed':
      return { stroke: '#22d3ee', shadow: '0 0 20px rgba(34, 211, 238, 0.5)' };
    case 'running':
    case 'executing':
      return { stroke: '#10b981', shadow: '0 0 20px rgba(16, 185, 129, 0.5)' };
    case 'failed':
      return { stroke: '#ef4444', shadow: '0 0 20px rgba(239, 68, 68, 0.5)' };
    default:
      return { stroke: '#f97316', shadow: '0 0 20px rgba(249, 115, 22, 0.5)' };
  }
};

export function ProgressRing({
  progress,
  size = 180,
  strokeWidth = 12,
  status = 'running',
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (progress / 100) * circumference;
  const colors = getColor(status);

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        className="progress-ring"
        width={size}
        height={size}
      >
        {/* Background circle */}
        <circle
          className="text-white/10"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        {/* Progress circle */}
        <circle
          stroke={colors.stroke}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset,
            filter: `drop-shadow(${colors.shadow})`,
            transition: 'stroke-dashoffset 0.5s ease',
          }}
        />
      </svg>
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="text-4xl font-bold"
          style={{ color: colors.stroke }}
        >
          {Math.round(progress)}%
        </span>
        <span className="text-xs text-gray-400 uppercase tracking-wider mt-1">
          {status}
        </span>
      </div>
    </div>
  );
}

export default ProgressRing;
