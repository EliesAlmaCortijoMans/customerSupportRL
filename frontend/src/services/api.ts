import axios, { AxiosInstance, AxiosResponse } from 'axios';
import toast from 'react-hot-toast';

// API Base URL - point directly to backend in development
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

// Create axios instance
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    // Handle common errors
    if (error.response) {
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          toast.error('Unauthorized access. Please login again.');
          // Could redirect to login here
          break;
        case 403:
          toast.error('Access forbidden.');
          break;
        case 404:
          toast.error('Resource not found.');
          break;
        case 500:
          toast.error('Server error. Please try again later.');
          break;
        default:
          toast.error(data?.message || 'An error occurred.');
      }
    } else if (error.request) {
      toast.error('Network error. Please check your connection.');
    } else {
      toast.error('An unexpected error occurred.');
    }
    
    return Promise.reject(error);
  }
);

// API Types
export interface Environment {
  environment_id: string;
  industry: 'bfsi' | 'retail' | 'tech' | 'mixed';
  environment_type: 'standard' | 'vectorized' | 'advanced';
  max_conversation_length: number;
  created_at: string;
  num_envs: number;
  is_active: boolean;
  total_episodes: number;
  current_episode?: number;
}

export interface CreateEnvironmentRequest {
  industry: 'bfsi' | 'retail' | 'tech' | 'mixed';
  max_conversation_length?: number;
  environment_type?: 'standard' | 'vectorized' | 'advanced';
  num_envs?: number;
  add_curriculum?: boolean;
  add_noise?: boolean;
}

export interface TrainingSession {
  session_id: string;
  industry: 'bfsi' | 'retail' | 'tech' | 'mixed';
  algorithm: 'ppo' | 'a2c' | 'dqn';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  total_timesteps: number;
  current_timesteps: number;
  progress: number;
  num_envs: number;
  current_reward?: number;
  best_reward?: number;
}

export interface TrainingRequest {
  industry: 'bfsi' | 'retail' | 'tech' | 'mixed';
  algorithm: 'ppo' | 'a2c' | 'dqn';
  total_timesteps?: number;
  num_envs?: number;
  learning_rate?: number;
  eval_freq?: number;
  save_freq?: number;
  model_params?: Record<string, any>;
}

export interface Model {
  model_id: string;
  name: string;
  industry: 'bfsi' | 'retail' | 'tech' | 'mixed';
  algorithm: 'ppo' | 'a2c' | 'dqn';
  created_at: string;
  training_session_id: string;
  total_timesteps: number;
  final_reward: number;
  final_satisfaction: number;
  model_size_mb: number;
  is_default: boolean;
}

export interface EvaluationRequest {
  n_episodes?: number;
  industry?: 'bfsi' | 'retail' | 'tech' | 'mixed';
  render?: boolean;
  deterministic?: boolean;
}

export interface EvaluationResult {
  model_id: string;
  n_episodes: number;
  mean_reward: number;
  std_reward: number;
  mean_satisfaction: number;
  std_satisfaction: number;
  mean_length: number;
  success_rate: number;
  escalation_rate: number;
  strategy_distribution: Record<string, number>;
  tier_performance: Record<string, number>;
  evaluation_time: number;
}

// Environment API
export const environmentApi = {
  // List all environments
  list: (): Promise<AxiosResponse<Environment[]>> =>
    apiClient.get('/environments'),

  // Create new environment
  create: (data: CreateEnvironmentRequest): Promise<AxiosResponse<{ environment_id: string }>> =>
    apiClient.post('/environments', data),

  // Get environment details
  get: (envId: string): Promise<AxiosResponse<Environment>> =>
    apiClient.get(`/environments/${envId}`),

  // Delete environment
  delete: (envId: string): Promise<AxiosResponse<void>> =>
    apiClient.delete(`/environments/${envId}`),

  // Reset environment
  reset: (envId: string): Promise<AxiosResponse<any>> =>
    apiClient.post(`/environments/${envId}/reset`),

  // Step environment
  step: (envId: string, action: number): Promise<AxiosResponse<any>> =>
    apiClient.post(`/environments/${envId}/step`, { action }),

  // Get environment metrics
  getMetrics: (envId: string): Promise<AxiosResponse<any>> =>
    apiClient.get(`/environments/${envId}/metrics`),
};

// Training API
export const trainingApi = {
  // List training sessions
  list: (): Promise<AxiosResponse<TrainingSession[]>> =>
    apiClient.get('/training/sessions'),

  // Start training
  start: (data: TrainingRequest): Promise<AxiosResponse<{ session_id: string }>> =>
    apiClient.post('/training/start', data),

  // Get training session details
  get: (sessionId: string): Promise<AxiosResponse<TrainingSession>> =>
    apiClient.get(`/training/sessions/${sessionId}`),

  // Stop training
  stop: (sessionId: string): Promise<AxiosResponse<void>> =>
    apiClient.post(`/training/sessions/${sessionId}/stop`),

  // Get training metrics
  getMetrics: (sessionId: string): Promise<AxiosResponse<any>> =>
    apiClient.get(`/training/sessions/${sessionId}/metrics`),
};

// Model API
export const modelApi = {
  // List models
  list: (): Promise<AxiosResponse<Model[]>> =>
    apiClient.get('/models'),

  // Evaluate model
  evaluate: (modelId: string, data: EvaluationRequest): Promise<AxiosResponse<EvaluationResult>> =>
    apiClient.post(`/models/${modelId}/evaluate`, data),

  // Predict action
  predict: (modelId: string, observation: number[]): Promise<AxiosResponse<any>> =>
    apiClient.post(`/models/${modelId}/predict`, {
      observation,
      deterministic: true,
    }),
};

// Analytics API
export const analyticsApi = {
  // Get overview
  getOverview: (): Promise<AxiosResponse<any>> =>
    apiClient.get('/analytics/overview'),
};

// Scenarios API
export const scenariosApi = {
  // List scenarios
  list: (): Promise<AxiosResponse<any[]>> =>
    apiClient.get('/scenarios'),

  // Run scenario
  run: (scenarioId: string, data: any): Promise<AxiosResponse<any>> =>
    apiClient.post(`/scenarios/${scenarioId}/run`, data),
};

// Configuration API
export const configApi = {
  // Get configuration
  get: (): Promise<AxiosResponse<any>> =>
    apiClient.get('/config'),
};

// Health check API
export const healthApi = {
  // Check health
  check: (): Promise<AxiosResponse<any>> =>
    apiClient.get('/health'),
};

// Export default API client
export default apiClient;
