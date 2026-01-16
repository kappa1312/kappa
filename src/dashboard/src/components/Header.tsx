import { Activity, Wifi, WifiOff } from 'lucide-react';
import { useState, useEffect } from 'react';

interface HeaderProps {
  isConnected: boolean;
}

export function Header({ isConnected }: HeaderProps) {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <header className="glass-panel px-6 py-4 flex items-center justify-between">
      {/* Logo */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-cyan to-neon-purple flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold holographic-text">KAPPA OS</h1>
            <p className="text-xs text-gray-400">Autonomous Development System</p>
          </div>
        </div>
      </div>

      {/* Center - Live indicator */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <>
              <Wifi className="w-4 h-4 text-neon-green" />
              <span className="text-neon-green text-sm font-medium">LIVE</span>
              <span className="status-orb running" />
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-neon-red" />
              <span className="text-neon-red text-sm font-medium">OFFLINE</span>
              <span className="status-orb failed" />
            </>
          )}
        </div>
      </div>

      {/* Right - Time and version */}
      <div className="flex items-center gap-6">
        <div className="text-right">
          <div className="font-mono text-lg text-white">{formatTime(currentTime)}</div>
          <div className="text-xs text-gray-400">{formatDate(currentTime)}</div>
        </div>
        <div className="px-3 py-1 rounded-full bg-neon-purple/20 border border-neon-purple/30">
          <span className="text-xs font-mono text-neon-purple">v0.1.0-beta</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
