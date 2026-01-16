import { useRef, useEffect } from 'react';
import { User, Bot, Settings, MessageSquare } from 'lucide-react';
import type { ChatMessage } from '../types';

interface LogViewerProps {
  messages: ChatMessage[];
}

const getMessageIcon = (role: string) => {
  switch (role) {
    case 'user':
      return <User className="w-4 h-4 text-neon-cyan" />;
    case 'assistant':
      return <Bot className="w-4 h-4 text-neon-purple" />;
    case 'system':
      return <Settings className="w-4 h-4 text-neon-orange" />;
    default:
      return <MessageSquare className="w-4 h-4 text-gray-400" />;
  }
};

const getRoleName = (role: string) => {
  switch (role) {
    case 'user':
      return 'User';
    case 'assistant':
      return 'Kappa';
    case 'system':
      return 'System';
    default:
      return role;
  }
};

const formatTimestamp = (timestamp: string) => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
};

export function LogViewer({ messages }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No logs to display</p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-[400px] overflow-y-auto space-y-4 pr-2"
    >
      {messages.map((message, index) => (
        <div
          key={index}
          className={`chat-message ${message.role} p-4 rounded-xl`}
        >
          {/* Header */}
          <div className="flex items-center gap-2 mb-2">
            {getMessageIcon(message.role)}
            <span
              className={`text-sm font-medium ${
                message.role === 'user'
                  ? 'text-neon-cyan'
                  : message.role === 'assistant'
                  ? 'text-neon-purple'
                  : 'text-neon-orange'
              }`}
            >
              {getRoleName(message.role)}
            </span>
            <span className="text-xs text-gray-500 ml-auto font-mono">
              {formatTimestamp(message.timestamp)}
            </span>
          </div>

          {/* Content */}
          <div className="text-sm text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
            {message.content}
          </div>
        </div>
      ))}
    </div>
  );
}

export default LogViewer;
