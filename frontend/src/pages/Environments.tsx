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
  Switch,
  FormControlLabel,
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
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

import { environmentApi, Environment, CreateEnvironmentRequest } from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';

const Environments: React.FC = () => {
  const queryClient = useQueryClient();
  const { isConnected, subscribe, unsubscribe } = useWebSocket();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedEnv, setSelectedEnv] = useState<Environment | null>(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);

  // Form state for creating new environment
  const [newEnv, setNewEnv] = useState<CreateEnvironmentRequest>({
    industry: 'mixed',
    max_conversation_length: 10,
    environment_type: 'standard',
    num_envs: 1,
    add_curriculum: false,
    add_noise: false,
  });

  // Fetch environments
  const { data: environments, isLoading, error, refetch } = useQuery({
    queryKey: ['environments'],
    queryFn: async () => {
      const response = await environmentApi.list();
      return response.data;
    },
    refetchInterval: 15000,
  });

  // Create environment mutation
  const createMutation = useMutation({
    mutationFn: (data: CreateEnvironmentRequest) => environmentApi.create(data),
    onSuccess: (response) => {
      toast.success('Environment created successfully!');
      setCreateDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ['environments'] });
      
      // Subscribe to new environment updates
      if (isConnected) {
        subscribe('environment', response.data.environment_id);
      }
      
      // Reset form
      setNewEnv({
        industry: 'mixed',
        max_conversation_length: 10,
        environment_type: 'standard',
        num_envs: 1,
        add_curriculum: false,
        add_noise: false,
      });
    },
    onError: (error) => {
      toast.error('Failed to create environment');
      console.error(error);
    },
  });

  // Delete environment mutation
  const deleteMutation = useMutation({
    mutationFn: (envId: string) => environmentApi.delete(envId),
    onSuccess: (_, envId) => {
      toast.success('Environment deleted successfully!');
      queryClient.invalidateQueries({ queryKey: ['environments'] });
      
      // Unsubscribe from environment updates
      if (isConnected) {
        unsubscribe('environment', envId);
      }
    },
    onError: () => {
      toast.error('Failed to delete environment');
    },
  });

  // Reset environment mutation
  const resetMutation = useMutation({
    mutationFn: (envId: string) => environmentApi.reset(envId),
    onSuccess: () => {
      toast.success('Environment reset successfully!');
      queryClient.invalidateQueries({ queryKey: ['environments'] });
    },
    onError: () => {
      toast.error('Failed to reset environment');
    },
  });

  const handleCreateEnvironment = () => {
    createMutation.mutate(newEnv);
  };

  const handleDeleteEnvironment = (envId: string) => {
    if (window.confirm('Are you sure you want to delete this environment?')) {
      deleteMutation.mutate(envId);
    }
  };

  const handleResetEnvironment = (envId: string) => {
    resetMutation.mutate(envId);
  };

  const handleViewDetails = async (env: Environment) => {
    setSelectedEnv(env);
    setDetailsDialogOpen(true);
    
    // Subscribe to environment updates
    if (isConnected) {
      subscribe('environment', env.environment_id);
    }
  };

  const getIndustryColor = (industry: string) => {
    switch (industry) {
      case 'bfsi': return 'primary';
      case 'retail': return 'secondary';
      case 'tech': return 'success';
      case 'mixed': return 'info';
      default: return 'default';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'standard': return 'default';
      case 'vectorized': return 'warning';
      case 'advanced': return 'error';
      default: return 'default';
    }
  };

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to load environments. Please check your connection.
      </Alert>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          Environment Management
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
            Create Environment
          </Button>
        </Box>
      </Box>

      {/* Connection Status */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Real-time updates are disabled. Environment changes may not be reflected immediately.
        </Alert>
      )}

      {/* Environment Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Total Environments
              </Typography>
              <Typography variant="h4" color="primary.main">
                {environments?.length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Active Environments
              </Typography>
              <Typography variant="h4" color="success.main">
                {environments?.filter(env => env.is_active).length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Total Episodes
              </Typography>
              <Typography variant="h4" color="info.main">
                {environments?.reduce((sum, env) => sum + env.total_episodes, 0) || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Vectorized Environments
              </Typography>
              <Typography variant="h4" color="warning.main">
                {environments?.filter(env => env.environment_type === 'vectorized').length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Environments Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" component="div" sx={{ mb: 2 }}>
            All Environments
          </Typography>
          {isLoading ? (
            <LinearProgress />
          ) : (
            <TableContainer component={Paper} elevation={0}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Industry</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Episodes</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {environments?.map((env) => (
                    <TableRow key={env.environment_id} hover>
                      <TableCell>
                        <Chip
                          label={env.industry.toUpperCase()}
                          color={getIndustryColor(env.industry) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={env.environment_type}
                          color={getTypeColor(env.environment_type) as any}
                          variant="outlined"
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{env.total_episodes}</TableCell>
                      <TableCell>
                        <Chip
                          label={env.is_active ? 'Active' : 'Inactive'}
                          color={env.is_active ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {format(new Date(env.created_at), 'MMM dd, yyyy HH:mm')}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => handleViewDetails(env)}
                            >
                              <VisibilityIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Reset Environment">
                            <IconButton
                              size="small"
                              onClick={() => handleResetEnvironment(env.environment_id)}
                              disabled={resetMutation.isPending}
                            >
                              <RefreshIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete Environment">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleDeleteEnvironment(env.environment_id)}
                              disabled={deleteMutation.isPending}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                  {environments?.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No environments found. Create your first environment to get started.
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

      {/* Create Environment Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Create New Environment</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, pt: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Industry</InputLabel>
              <Select
                value={newEnv.industry}
                label="Industry"
                onChange={(e) => setNewEnv({ ...newEnv, industry: e.target.value as any })}
              >
                <MenuItem value="bfsi">Banking & Financial Services</MenuItem>
                <MenuItem value="retail">Retail & E-commerce</MenuItem>
                <MenuItem value="tech">Technology & SaaS</MenuItem>
                <MenuItem value="mixed">Mixed Industries</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Environment Type</InputLabel>
              <Select
                value={newEnv.environment_type}
                label="Environment Type"
                onChange={(e) => setNewEnv({ ...newEnv, environment_type: e.target.value as any })}
              >
                <MenuItem value="standard">Standard</MenuItem>
                <MenuItem value="vectorized">Vectorized (Parallel)</MenuItem>
                <MenuItem value="advanced">Advanced (Curriculum)</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Max Conversation Length"
              type="number"
              value={newEnv.max_conversation_length}
              onChange={(e) => setNewEnv({ ...newEnv, max_conversation_length: parseInt(e.target.value) })}
              inputProps={{ min: 1, max: 20 }}
              fullWidth
            />

            {newEnv.environment_type === 'vectorized' && (
              <TextField
                label="Number of Parallel Environments"
                type="number"
                value={newEnv.num_envs}
                onChange={(e) => setNewEnv({ ...newEnv, num_envs: parseInt(e.target.value) })}
                inputProps={{ min: 1, max: 16 }}
                fullWidth
              />
            )}

            {newEnv.environment_type === 'advanced' && (
              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={newEnv.add_curriculum}
                      onChange={(e) => setNewEnv({ ...newEnv, add_curriculum: e.target.checked })}
                    />
                  }
                  label="Enable Curriculum Learning"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={newEnv.add_noise}
                      onChange={(e) => setNewEnv({ ...newEnv, add_noise: e.target.checked })}
                    />
                  }
                  label="Add Observation Noise"
                />
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateEnvironment}
            variant="contained"
            disabled={createMutation.isPending}
          >
            {createMutation.isPending ? 'Creating...' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Environment Details Dialog */}
      <Dialog
        open={detailsDialogOpen}
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Environment Details</DialogTitle>
        <DialogContent>
          {selectedEnv && (
            <Box sx={{ pt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Environment ID</Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedEnv.environment_id}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Industry</Typography>
                  <Chip
                    label={selectedEnv.industry.toUpperCase()}
                    color={getIndustryColor(selectedEnv.industry) as any}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Type</Typography>
                  <Chip
                    label={selectedEnv.environment_type}
                    color={getTypeColor(selectedEnv.environment_type) as any}
                    variant="outlined"
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Status</Typography>
                  <Chip
                    label={selectedEnv.is_active ? 'Active' : 'Inactive'}
                    color={selectedEnv.is_active ? 'success' : 'default'}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Total Episodes</Typography>
                  <Typography variant="body2">{selectedEnv.total_episodes}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Max Conversation Length</Typography>
                  <Typography variant="body2">{selectedEnv.max_conversation_length}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Parallel Environments</Typography>
                  <Typography variant="body2">{selectedEnv.num_envs}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2">Created</Typography>
                  <Typography variant="body2">
                    {format(new Date(selectedEnv.created_at), 'PPpp')}
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Environments;
