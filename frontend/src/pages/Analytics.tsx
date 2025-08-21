import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
} from '@mui/material';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const Analytics: React.FC = () => {
  // Mock data for charts
  const industryData = [
    { name: 'BFSI', value: 40, episodes: 1200 },
    { name: 'Retail', value: 35, episodes: 1050 },
    { name: 'Tech', value: 25, episodes: 750 },
  ];

  const satisfactionTrend = [
    { month: 'Jan', satisfaction: 0.65 },
    { month: 'Feb', satisfaction: 0.68 },
    { month: 'Mar', satisfaction: 0.72 },
    { month: 'Apr', satisfaction: 0.75 },
    { month: 'May', satisfaction: 0.78 },
    { month: 'Jun', satisfaction: 0.82 },
  ];

  const strategyEffectiveness = [
    { strategy: 'Empathetic', effectiveness: 0.85 },
    { strategy: 'Technical', effectiveness: 0.78 },
    { strategy: 'Quick Resolution', effectiveness: 0.92 },
    { strategy: 'Educational', effectiveness: 0.76 },
    { strategy: 'Apologetic', effectiveness: 0.68 },
  ];

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 3 }}>
        Analytics Dashboard
      </Typography>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Overall Satisfaction
              </Typography>
              <Typography variant="h4" color="success.main">
                82.3%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                +5.2% from last month
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Avg Resolution Time
              </Typography>
              <Typography variant="h4" color="info.main">
                4.2
              </Typography>
              <Typography variant="body2" color="text.secondary">
                interactions per episode
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div">
                Escalation Rate
              </Typography>
              <Typography variant="h4" color="warning.main">
                12.5%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                -2.1% from last month
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
              <Typography variant="h4" color="primary.main">
                3,000
              </Typography>
              <Typography variant="body2" color="text.secondary">
                across all environments
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
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
                    data={industryData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {industryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Satisfaction Trend */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                Customer Satisfaction Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={satisfactionTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Satisfaction']} />
                  <Line
                    type="monotone"
                    dataKey="satisfaction"
                    stroke="#2e7d32"
                    strokeWidth={2}
                    dot={{ fill: '#2e7d32' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Strategy Effectiveness */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                Strategy Effectiveness
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={strategyEffectiveness}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="strategy" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Effectiveness']} />
                  <Bar dataKey="effectiveness" fill="#1976d2" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Analytics;
