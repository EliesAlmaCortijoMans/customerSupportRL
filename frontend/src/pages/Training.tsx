import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Chip,
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
  LinearProgress,
  Slider,
} from '@mui/material';
import {
  Add as AddIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import toast from 'react-hot-toast';

import { trainingApi, TrainingRequest, TrainingSession } from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';

const Training: React.FC = () => {
  const queryClient = useQueryClient();
  const { isConnected, subscribe } = useWebSocket();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedSession, setSelectedSession] = useState<TrainingSession | null>(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);

  // Form state for new training
  const [newTraining, setNewTraining] = useState<TrainingRequest>({
    industry: 'mixed',
    algorithm: 'ppo',
    total_timesteps: 50000,
    num_envs: 4,
    learning_rate: 0.0003,
    eval_freq: 5000,
    save_freq: 10000,
  });

  // Fetch training sessions
  const { data: sessions, isLoading, error, refetch } = useQuery({
    queryKey: ['training-sessions'],
    queryFn: async () => {
      const response = await trainingApi.list();
      return response.data;
    },
    refetchInterval: 10000,
  });

  // Start training mutation
  const startMutation = useMutation({
    mutationFn: (data: TrainingRequest) => trainingApi.start(data),
    onSuccess: (response) => {
      toast.success('Training started successfully!');
      setCreateDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ['training-sessions'] });
      
      // Subscribe to training updates
      if (isConnected) {
        subscribe('training', response.data.session_id);
      }
      
      // Reset form
      setNewTraining({
        industry: 'mixed',
        algorithm: 'ppo',
        total_timesteps: 50000,
        num_envs: 4,
        learning_rate: 0.0003,
        eval_freq: 5000,
        save_freq: 10000,
      });
    },
    onError: () => {
      toast.error('Failed to start training');
    },
  });

  // Stop training mutation
  const stopMutation = useMutation({
    mutationFn: (sessionId: string) => trainingApi.stop(sessionId),
    onSuccess: () => {
      toast.success('Training stopped successfully!');
      queryClient.invalidateQueries({ queryKey: ['training-sessions'] });
    },
    onError: () => {
      toast.error('Failed to stop training');
    },
  });

  const handleStartTraining = () => {
    startMutation.mutate(newTraining);
  };

  const handleStopTraining = (sessionId: string) => {
    if (window.confirm('Are you sure you want to stop this training session?')) {
      stopMutation.mutate(sessionId);
    }
  };

  const handleViewDetails = (session: TrainingSession) => {
    setSelectedSession(session);
    setDetailsDialogOpen(true);
    
    // Subscribe to training updates
    if (isConnected) {
      subscribe('training', session.session_id);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success';
      case 'completed': return 'info';
      case 'failed': return 'error';
      case 'pending': return 'warning';
      case 'stopped': return 'default';
      default: return 'default';
    }
  };

  const getAlgorithmColor = (algorithm: string) => {
    switch (algorithm) {
      case 'ppo': return 'primary';
      case 'a2c': return 'secondary';
      case 'dqn': return 'success';
      default: return 'default';
    }
  };

  // Mock training progress data
  const mockProgressData = [
    { step: 0, reward: 0.5, satisfaction: 0.6 },
    { step: 10000, reward: 1.2, satisfaction: 0.68 },
    { step: 20000, reward: 1.8, satisfaction: 0.75 },
    { step: 30000, reward: 2.1, satisfaction: 0.78 },
    { step: 40000, reward: 2.4, satisfaction: 0.82 },
    { step: 50000, reward: 2.6, satisfaction: 0.85 },
  ];

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to load training sessions. Please check your connection.
      </Alert>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          Training Management
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetch()}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
          >
            Start Training
          </Button>
        </Box>
      </Box>

      {/* Connection Status */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Real-time training updates are disabled. Training progress may not be reflected immediately.
        </Alert>
      )}

      {/* Training Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Total Sessions
              </Typography>
              <Typography variant="h4" color="primary.main">
                {sessions?.length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Running Sessions
              </Typography>
              <Typography variant="h4" color="success.main">
                {sessions?.filter(s => s.status === 'running').length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Completed Sessions
              </Typography>
              <Typography variant="h4" color="info.main">
                {sessions?.filter(s => s.status === 'completed').length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Failed Sessions
              </Typography>
              <Typography variant="h4" color="error.main">
                {sessions?.filter(s => s.status === 'failed').length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Training Progress Chart */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" component="div" sx={{ mb: 2 }}>
            Training Progress Overview
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockProgressData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <RechartsTooltip />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="reward"
                stroke="#1976d2"
                strokeWidth={2}
                name="Reward"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="satisfaction"
                stroke="#2e7d32"
                strokeWidth={2}
                name="Satisfaction"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Training Sessions Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" component="div" sx={{ mb: 2 }}>
            Training Sessions
          </Typography>
          {isLoading ? (
            <LinearProgress />
          ) : (
            <TableContainer component={Paper} elevation={0}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Algorithm</TableCell>
                    <TableCell>Industry</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>Reward</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sessions?.map((session) => (
                    <TableRow key={session.session_id} hover>
                      <TableCell>
                        <Chip
                          label={session.algorithm.toUpperCase()}
                          color={getAlgorithmColor(session.algorithm) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={session.industry.toUpperCase()}
                          variant="outlined"
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={session.status}
                          color={getStatusColor(session.status) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 120 }}>
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
                        {session.current_reward?.toFixed(2) || 'N/A'}
                      </TableCell>
                      <TableCell>
                        {format(new Date(session.created_at), 'MMM dd, HH:mm')}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => handleViewDetails(session)}
                            >
                              <VisibilityIcon />
                            </IconButton>
                          </Tooltip>
                          {session.status === 'running' && (
                            <Tooltip title="Stop Training">
                              <IconButton
                                size="small"
                                color="error"
                                onClick={() => handleStopTraining(session.session_id)}
                                disabled={stopMutation.isPending}
                              >
                                <StopIcon />
                              </IconButton>
                            </Tooltip>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                  {sessions?.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No training sessions found. Start your first training session to get started.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Create Training Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Start New Training Session</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, pt: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Industry</InputLabel>
              <Select
                value={newTraining.industry}
                label="Industry"
                onChange={(e) => setNewTraining({ ...newTraining, industry: e.target.value as any })}
              >
                <MenuItem value="bfsi">Banking & Financial Services</MenuItem>
                <MenuItem value="retail">Retail & E-commerce</MenuItem>
                <MenuItem value="tech">Technology & SaaS</MenuItem>
                <MenuItem value="mixed">Mixed Industries</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={newTraining.algorithm}
                label="Algorithm"
                onChange={(e) => setNewTraining({ ...newTraining, algorithm: e.target.value as any })}
              >
                <MenuItem value="ppo">PPO (Proximal Policy Optimization)</MenuItem>
                <MenuItem value="a2c">A2C (Advantage Actor-Critic)</MenuItem>
                <MenuItem value="dqn">DQN (Deep Q-Network)</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Total Timesteps"
              type="number"
              value={newTraining.total_timesteps}
              onChange={(e) => setNewTraining({ ...newTraining, total_timesteps: parseInt(e.target.value) })}
              inputProps={{ min: 1000, max: 1000000, step: 1000 }}
              fullWidth
            />

            <TextField
              label="Number of Environments"
              type="number"
              value={newTraining.num_envs}
              onChange={(e) => setNewTraining({ ...newTraining, num_envs: parseInt(e.target.value) })}
              inputProps={{ min: 1, max: 16 }}
              fullWidth
            />

            <Box>
              <Typography gutterBottom>Learning Rate</Typography>
              <Slider
                value={newTraining.learning_rate || 0.0003}
                onChange={(_, value) => setNewTraining({ ...newTraining, learning_rate: value as number })}
                min={0.00001}
                max={0.01}
                step={0.00001}
                scale={(x) => x * 1000}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value.toFixed(5)}`}
              />
            </Box>

            <TextField
              label="Evaluation Frequency"
              type="number"
              value={newTraining.eval_freq}
              onChange={(e) => setNewTraining({ ...newTraining, eval_freq: parseInt(e.target.value) })}
              inputProps={{ min: 100, max: 50000, step: 100 }}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleStartTraining}
            variant="contained"
            disabled={startMutation.isPending}
          >
            {startMutation.isPending ? 'Starting...' : 'Start Training'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Training Details Dialog */}
      <Dialog
        open={detailsDialogOpen}
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Training Session Details</DialogTitle>
        <DialogContent>
          {selectedSession && (
            <Box sx={{ pt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Session ID</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedSession.session_id}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Status</Typography>
                  <Chip
                    label={selectedSession.status}
                    color={getStatusColor(selectedSession.status) as any}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Algorithm</Typography>
                  <Chip
                    label={selectedSession.algorithm.toUpperCase()}
                    color={getAlgorithmColor(selectedSession.algorithm) as any}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Industry</Typography>
                  <Chip
                    label={selectedSession.industry.toUpperCase()}
                    variant="outlined"
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Progress</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={selectedSession.progress * 100}
                      sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="body2">
                      {(selectedSession.progress * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Current Timesteps</Typography>
                  <Typography variant="body2">
                    {selectedSession.current_timesteps.toLocaleString()} / {selectedSession.total_timesteps.toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Current Reward</Typography>
                  <Typography variant="body2">
                    {selectedSession.current_reward?.toFixed(3) || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Best Reward</Typography>
                  <Typography variant="body2">
                    {selectedSession.best_reward?.toFixed(3) || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Parallel Environments</Typography>
                  <Typography variant="body2">{selectedSession.num_envs}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Created</Typography>
                  <Typography variant="body2">
                    {format(new Date(selectedSession.created_at), 'PPpp')}
                  </Typography>
                </Grid>
                {selectedSession.started_at && (
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Started</Typography>
                    <Typography variant="body2">
                      {format(new Date(selectedSession.started_at), 'PPpp')}
                    </Typography>
                  </Grid>
                )}
                {selectedSession.completed_at && (
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Completed</Typography>
                    <Typography variant="body2">
                      {format(new Date(selectedSession.completed_at), 'PPpp')}
                    </Typography>
                  </Grid>
                )}
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
          {selectedSession?.status === 'running' && (
            <Button
              color="error"
              onClick={() => {
                handleStopTraining(selectedSession.session_id);
                setDetailsDialogOpen(false);
              }}
              disabled={stopMutation.isPending}
            >
              Stop Training
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Training;
