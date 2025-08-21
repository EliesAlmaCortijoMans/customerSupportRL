import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Alert,
  Button,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Star as StarIcon,
  Psychology as PsychologyIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { motion } from 'framer-motion';

import { apiClient } from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';

interface DashboardMetrics {
  total_environments: number;
  total_training_sessions: number;
  total_episodes: number;
  average_satisfaction: number;
  average_resolution_time: number;
  active_connections: number;
}

interface EnvironmentInfo {
  environment_id: string;
  industry: string;
  environment_type: string;
  is_active: boolean;
  total_episodes: number;
  created_at: string;
}

interface TrainingSession {
  session_id: string;
  industry: string;
  algorithm: string;
  status: string;
  progress: number;
  current_reward?: number;
  created_at: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const Dashboard: React.FC = () => {
  const { isConnected } = useWebSocket();
  const [refreshKey, setRefreshKey] = useState(0);

  // Fetch dashboard metrics
  const { data: metrics, isLoading: metricsLoading, error: metricsError } = useQuery({
    queryKey: ['dashboard-metrics', refreshKey],
    queryFn: async (): Promise<DashboardMetrics> => {
      const response = await apiClient.get('/analytics/overview');
      return response.data;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch environments
  const { data: environments, isLoading: environmentsLoading } = useQuery({
    queryKey: ['environments', refreshKey],
    queryFn: async (): Promise<EnvironmentInfo[]> => {
      const response = await apiClient.get('/environments');
      return response.data;
    },
    refetchInterval: 15000,
  });

  // Fetch training sessions
  const { data: trainingSessions, isLoading: trainingLoading } = useQuery({
    queryKey: ['training-sessions', refreshKey],
    queryFn: async (): Promise<TrainingSession[]> => {
      const response = await apiClient.get('/training/sessions');
      return response.data;
    },
    refetchInterval: 10000,
  });

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  // Mock data for charts
  const satisfactionTrendData = [
    { time: '00:00', satisfaction: 0.65 },
    { time: '04:00', satisfaction: 0.72 },
    { time: '08:00', satisfaction: 0.78 },
    { time: '12:00', satisfaction: 0.82 },
    { time: '16:00', satisfaction: 0.79 },
    { time: '20:00', satisfaction: 0.85 },
  ];

  const industryDistributionData = [
    { name: 'BFSI', value: 35, episodes: 450 },
    { name: 'Retail', value: 40, episodes: 520 },
    { name: 'Tech', value: 25, episodes: 325 },
  ];

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running': return 'success';
      case 'completed': return 'info';
      case 'failed': return 'error';
      case 'pending': return 'warning';
      default: return 'default';
    }
  };

  const MetricCard = ({ title, value, subtitle, icon, color = 'primary' }: any) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Box sx={{ color: `${color}.main`, mr: 2 }}>
              {icon}
            </Box>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              {title}
            </Typography>
          </Box>
          <Typography variant="h4" component="div" color="text.primary" sx={{ mb: 1 }}>
            {value}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        </CardContent>
      </Card>
    </motion.div>
  );

  if (metricsError) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to load dashboard data. Please check your connection.
      </Alert>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          Dashboard Overview
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={metricsLoading}
        >
          Refresh
        </Button>
      </Box>

      {/* Connection Status Alert */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Real-time updates are disabled. Check your connection to the server.
        </Alert>
      )}

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Environments"
            value={metrics?.total_environments || 0}
            subtitle="Active environments"
            icon={<PsychologyIcon />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Training Sessions"
            value={metrics?.total_training_sessions || 0}
            subtitle="All-time sessions"
            icon={<TrendingUpIcon />}
            color="secondary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Episodes Completed"
            value={metrics?.total_episodes || 0}
            subtitle="Across all environments"
            icon={<SpeedIcon />}
            color="info"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Satisfaction"
            value={`${((metrics?.average_satisfaction || 0) * 100).toFixed(1)}%`}
            subtitle="Customer satisfaction"
            icon={<StarIcon />}
            color="success"
          />
        </Grid>
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Satisfaction Trend */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                Customer Satisfaction Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={satisfactionTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 1]} />
                  <RechartsTooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Satisfaction']} />
                  <Line
                    type="monotone"
                    dataKey="satisfaction"
                    stroke="#1976d2"
                    strokeWidth={2}
                    dot={{ fill: '#1976d2' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Industry Distribution */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                Episodes by Industry
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={industryDistributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {industryDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tables Row */}
      <Grid container spacing={3}>
        {/* Active Environments */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                Active Environments
              </Typography>
              {environmentsLoading ? (
                <LinearProgress />
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Industry</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Episodes</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {environments?.slice(0, 5).map((env) => (
                        <TableRow key={env.environment_id}>
                          <TableCell>
                            <Chip
                              label={env.industry.toUpperCase()}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>{env.environment_type}</TableCell>
                          <TableCell>{env.total_episodes}</TableCell>
                          <TableCell>
                            <Chip
                              label={env.is_active ? 'Active' : 'Inactive'}
                              size="small"
                              color={env.is_active ? 'success' : 'default'}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Training Sessions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                Recent Training Sessions
              </Typography>
              {trainingLoading ? (
                <LinearProgress />
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Algorithm</TableCell>
                        <TableCell>Industry</TableCell>
                        <TableCell>Progress</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {trainingSessions?.slice(0, 5).map((session) => (
                        <TableRow key={session.session_id}>
                          <TableCell>
                            <Chip
                              label={session.algorithm.toUpperCase()}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>{session.industry.toUpperCase()}</TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={session.progress * 100}
                                sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                              />
                              <Typography variant="caption">
                                {(session.progress * 100).toFixed(0)}%
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={session.status}
                              size="small"
                              color={getStatusColor(session.status) as any}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
