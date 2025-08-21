import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import toast from 'react-hot-toast';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface WebSocketContextType {
  socket: WebSocket | null;
  isConnected: boolean;
  subscribe: (resourceType: string, resourceId: string) => void;
  unsubscribe: (resourceType: string, resourceId: string) => void;
  sendMessage: (message: any) => void;
  lastMessage: WebSocketMessage | null;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  useEffect(() => {
    // Generate unique client ID
    const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Message handler functions
    const handleEnvironmentUpdate = (data: any) => {
      switch (data.update_type) {
        case 'reset':
          toast.success(`Environment ${data.environment_id} reset`);
          break;
        case 'step':
          // Don't show toast for every step, just log
          console.log(`Environment ${data.environment_id} step completed`);
          break;
        case 'episode_completed':
          const satisfaction = data.data.satisfaction || 0;
          toast.success(`Episode completed! Satisfaction: ${(satisfaction * 100).toFixed(1)}%`);
          break;
        default:
          console.log('Environment update:', data);
      }
    };

    const handleTrainingUpdate = (data: any) => {
      switch (data.update_type) {
        case 'started':
          toast.success(`Training session ${data.session_id} started`);
          break;
        case 'progress':
          const progress = data.data.progress || 0;
          console.log(`Training progress: ${(progress * 100).toFixed(1)}%`);
          break;
        case 'completed':
          toast.success(`Training session ${data.session_id} completed!`);
          break;
        case 'failed':
          toast.error(`Training session ${data.session_id} failed`);
          break;
        case 'stopped':
          toast(`Training session ${data.session_id} stopped`);
          break;
        default:
          console.log('Training update:', data);
      }
    };

    const handleModelUpdate = (data: any) => {
      switch (data.update_type) {
        case 'created':
          toast.success(`New model ${data.model_id} created`);
          break;
        case 'evaluated':
          const satisfaction = data.data.mean_satisfaction || 0;
          toast.success(`Model evaluation completed: ${(satisfaction * 100).toFixed(1)}% satisfaction`);
          break;
        default:
          console.log('Model update:', data);
      }
    };

    const handleSystemAlert = (data: any) => {
      const { level, message, alert_type } = data;
      
      switch (level) {
        case 'error':
          toast.error(`${alert_type}: ${message}`);
          break;
        case 'warning':
          toast.error(`${alert_type}: ${message}`, { icon: '⚠️' });
          break;
        case 'info':
          toast.success(`${alert_type}: ${message}`, { icon: 'ℹ️' });
          break;
        default:
          toast(message);
      }
    };

    const handleMetricsUpdate = (data: any) => {
      console.log('Metrics update:', data);
      // Could trigger global state updates here
    };

    const handleProgressUpdate = (data: any) => {
      const { resource_type, resource_id, progress, details } = data;
      console.log(`Progress update for ${resource_type}:${resource_id}: ${(progress * 100).toFixed(1)}%`);
      
      // Could show progress notifications for long-running operations
      if (progress >= 1.0) {
        toast.success(`${resource_type} ${resource_id} completed!`);
      }
    };

    const handleMessage = (message: WebSocketMessage) => {
      switch (message.type) {
        case 'welcome':
          console.log('Welcome message received:', message.data);
          break;

        case 'environment_update':
          handleEnvironmentUpdate(message.data);
          break;

        case 'training_update':
          handleTrainingUpdate(message.data);
          break;

        case 'model_update':
          handleModelUpdate(message.data);
          break;

        case 'system_alert':
          handleSystemAlert(message.data);
          break;

        case 'metrics_update':
          handleMetricsUpdate(message.data);
          break;

        case 'progress_update':
          handleProgressUpdate(message.data);
          break;

        case 'subscription_confirmed':
          toast.success(`Subscribed to ${message.data.resource_type}:${message.data.resource_id}`);
          break;

        case 'unsubscription_confirmed':
          toast.success(`Unsubscribed from ${message.data.resource_type}:${message.data.resource_id}`);
          break;

        case 'pong':
          // Heartbeat response
          break;

        default:
          console.log('Unknown message type:', message.type, message.data);
      }
    };

    // Create WebSocket connection
    const wsUrl = `ws://localhost:8000/ws/${clientId}`;
    const newSocket = new WebSocket(wsUrl);

    // Connection event handlers
    newSocket.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      toast.success('Connected to server');
    };

    newSocket.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      toast.error('Disconnected from server');
    };

    newSocket.onerror = (error: Event) => {
      console.error('WebSocket connection error:', error);
      setIsConnected(false);
      toast.error('Connection error');
    };

    // Message handlers
    newSocket.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        const message: WebSocketMessage = {
          type: data.type || 'message',
          data: data.data || data,
          timestamp: data.timestamp || new Date().toISOString(),
        };
        setLastMessage(message);
        handleMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      newSocket.close();
    };
  }, []);

  const subscribe = useCallback((resourceType: string, resourceId: string) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify({
        type: 'subscribe',
        resource_type: resourceType,
        resource_id: resourceId,
      }));
    }
  }, [socket, isConnected]);

  const unsubscribe = useCallback((resourceType: string, resourceId: string) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify({
        type: 'unsubscribe',
        resource_type: resourceType,
        resource_id: resourceId,
      }));
    }
  }, [socket, isConnected]);

  const sendMessage = useCallback((message: any) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message));
    }
  }, [socket, isConnected]);

  // Periodic heartbeat
  useEffect(() => {
    if (socket && isConnected) {
      const interval = setInterval(() => {
        sendMessage({ type: 'ping' });
      }, 30000); // 30 seconds

      return () => clearInterval(interval);
    }
  }, [socket, isConnected, sendMessage]);

  const contextValue: WebSocketContextType = {
    socket,
    isConnected,
    subscribe,
    unsubscribe,
    sendMessage,
    lastMessage,
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};