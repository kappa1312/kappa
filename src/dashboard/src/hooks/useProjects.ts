import { useState, useEffect, useCallback } from 'react';
import type { Project, Wave, ChatMessage } from '../types';

interface UseProjectsReturn {
  projects: Project[];
  selectedProject: Project | null;
  waves: Wave[];
  chatHistory: ChatMessage[];
  loading: boolean;
  error: string | null;
  selectProject: (projectId: string | null) => void;
  refreshProjects: () => Promise<void>;
  refreshWaves: () => Promise<void>;
  refreshChatHistory: () => Promise<void>;
}

export function useProjects(): UseProjectsReturn {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [waves, setWaves] = useState<Wave[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProjects = useCallback(async () => {
    try {
      const response = await fetch('/api/projects/');
      if (!response.ok) throw new Error('Failed to fetch projects');
      const data = await response.json();
      setProjects(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch projects');
    }
  }, []);

  const fetchProjectDetails = useCallback(async (projectId: string) => {
    try {
      const response = await fetch(`/api/projects/${projectId}`);
      if (!response.ok) throw new Error('Failed to fetch project details');
      const data = await response.json();
      setSelectedProject(data);
    } catch (err) {
      console.error('Failed to fetch project details:', err);
    }
  }, []);

  const fetchWaves = useCallback(async (projectId: string) => {
    try {
      const response = await fetch(`/api/projects/${projectId}/waves`);
      if (!response.ok) throw new Error('Failed to fetch waves');
      const data = await response.json();
      setWaves(data);
    } catch (err) {
      console.error('Failed to fetch waves:', err);
      setWaves([]);
    }
  }, []);

  const fetchChatHistory = useCallback(async (projectId: string) => {
    try {
      const response = await fetch(`/api/projects/${projectId}/chat-history`);
      if (!response.ok) throw new Error('Failed to fetch chat history');
      const data = await response.json();
      setChatHistory(data);
    } catch (err) {
      console.error('Failed to fetch chat history:', err);
      setChatHistory([]);
    }
  }, []);

  // Initial load and refresh every 5 seconds
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await fetchProjects();
      setLoading(false);
    };

    loadData();

    const interval = setInterval(fetchProjects, 5000);
    return () => clearInterval(interval);
  }, [fetchProjects]);

  // Load selected project details
  useEffect(() => {
    if (selectedProjectId) {
      fetchProjectDetails(selectedProjectId);
      fetchWaves(selectedProjectId);
      fetchChatHistory(selectedProjectId);

      // Refresh selected project data every 3 seconds
      const interval = setInterval(() => {
        fetchProjectDetails(selectedProjectId);
        fetchWaves(selectedProjectId);
        fetchChatHistory(selectedProjectId);
      }, 3000);

      return () => clearInterval(interval);
    } else {
      setSelectedProject(null);
      setWaves([]);
      setChatHistory([]);
    }
  }, [selectedProjectId, fetchProjectDetails, fetchWaves, fetchChatHistory]);

  const selectProject = useCallback((projectId: string | null) => {
    setSelectedProjectId(projectId);
  }, []);

  const refreshProjects = useCallback(async () => {
    await fetchProjects();
  }, [fetchProjects]);

  const refreshWaves = useCallback(async () => {
    if (selectedProjectId) {
      await fetchWaves(selectedProjectId);
    }
  }, [selectedProjectId, fetchWaves]);

  const refreshChatHistory = useCallback(async () => {
    if (selectedProjectId) {
      await fetchChatHistory(selectedProjectId);
    }
  }, [selectedProjectId, fetchChatHistory]);

  return {
    projects,
    selectedProject,
    waves,
    chatHistory,
    loading,
    error,
    selectProject,
    refreshProjects,
    refreshWaves,
    refreshChatHistory,
  };
}

export default useProjects;
