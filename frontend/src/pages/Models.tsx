import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
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
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Assessment as AssessmentIcon,
  Download as DownloadIcon,
  Star as StarIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';

import { modelApi } from '../services/api';

const Models: React.FC = () => {
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await modelApi.list();
      return response.data;
    },
  });

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 3 }}>
        Model Management
      </Typography>

      {/* Model Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Total Models
              </Typography>
              <Typography variant="h4" color="primary.main">
                {models?.length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Default Models
              </Typography>
              <Typography variant="h4" color="success.main">
                {models?.filter(m => m.is_default).length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Best Satisfaction
              </Typography>
              <Typography variant="h4" color="info.main">
                {models?.length ? `${(Math.max(...models.map(m => m.final_satisfaction)) * 100).toFixed(1)}%` : 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Total Size
              </Typography>
              <Typography variant="h4" color="warning.main">
                {models?.reduce((sum, m) => sum + m.model_size_mb, 0).toFixed(1) || 0} MB
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Models Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" component="div" sx={{ mb: 2 }}>
            Available Models
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Algorithm</TableCell>
                  <TableCell>Industry</TableCell>
                  <TableCell>Satisfaction</TableCell>
                  <TableCell>Reward</TableCell>
                  <TableCell>Size</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {models?.map((model) => (
                  <TableRow key={model.model_id} hover>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {model.is_default && <StarIcon color="warning" fontSize="small" />}
                        {model.name}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip label={model.algorithm.toUpperCase()} size="small" />
                    </TableCell>
                    <TableCell>
                      <Chip label={model.industry.toUpperCase()} variant="outlined" size="small" />
                    </TableCell>
                    <TableCell>{(model.final_satisfaction * 100).toFixed(1)}%</TableCell>
                    <TableCell>{model.final_reward.toFixed(2)}</TableCell>
                    <TableCell>{model.model_size_mb.toFixed(1)} MB</TableCell>
                    <TableCell>{format(new Date(model.created_at), 'MMM dd, yyyy')}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Evaluate Model">
                          <IconButton size="small">
                            <AssessmentIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Test Model">
                          <IconButton size="small">
                            <PlayIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Download Model">
                          <IconButton size="small">
                            <DownloadIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
                {models?.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body2" color="text.secondary">
                        No models found. Train your first model to get started.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Models;
