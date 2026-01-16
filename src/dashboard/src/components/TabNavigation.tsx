import { LayoutDashboard, Waves, Terminal, MessageSquare } from 'lucide-react';
import type { TabType } from '../types';

interface TabNavigationProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
  { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="w-4 h-4" /> },
  { id: 'waves', label: 'Waves', icon: <Waves className="w-4 h-4" /> },
  { id: 'sessions', label: 'Sessions', icon: <Terminal className="w-4 h-4" /> },
  { id: 'logs', label: 'Logs', icon: <MessageSquare className="w-4 h-4" /> },
];

export function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  return (
    <div className="flex gap-1 p-1 glass-panel-light rounded-xl">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
            activeTab === tab.id
              ? 'tab-active text-white'
              : 'text-gray-400 hover:text-white hover:bg-white/5'
          }`}
        >
          {tab.icon}
          <span className="text-sm font-medium">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}

export default TabNavigation;
