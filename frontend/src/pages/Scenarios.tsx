import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Chip,
  Avatar,
} from '@mui/material';
import {
  Business as BusinessIcon,
  Store as StoreIcon,
  Computer as ComputerIcon,
  PlayArrow as PlayIcon,
} from '@mui/icons-material';

const Scenarios: React.FC = () => {
  const scenarios = [
    {
      id: 'bfsi_fraud_detection',
      title: 'BFSI Fraud Detection',
      description: 'Customer reports suspicious transaction activity and needs immediate assistance with fraud investigation.',
      industry: 'BFSI',
      difficulty: 'High',
      icon: <BusinessIcon />,
      color: 'primary',
      expectedStrategies: ['Empathetic', 'Technical', 'Escalate'],
    },
    {
      id: 'retail_product_return',
      title: 'Retail Product Return',
      description: 'Customer wants to return a defective product purchased online and is frustrated with the initial response.',
      industry: 'Retail',
      difficulty: 'Medium',
      icon: <StoreIcon />,
      color: 'secondary',
      expectedStrategies: ['Apologetic', 'Quick Resolution', 'Product Recommend'],
    },
    {
      id: 'tech_integration_help',
      title: 'Tech Integration Support',
      description: 'Developer needs help integrating API services and is experiencing authentication errors.',
      industry: 'Tech',
      difficulty: 'High',
      icon: <ComputerIcon />,
      color: 'success',
      expectedStrategies: ['Technical', 'Educational', 'Escalate'],
    },
    {
      id: 'bfsi_investment_advice',
      title: 'Investment Advisory',
      description: 'High-value customer seeking investment recommendations for retirement planning.',
      industry: 'BFSI',
      difficulty: 'Medium',
      icon: <BusinessIcon />,
      color: 'primary',
      expectedStrategies: ['Educational', 'Product Recommend', 'Upsell'],
    },
    {
      id: 'retail_discount_inquiry',
      title: 'Discount Inquiry',
      description: 'Customer asking about available promotions and discounts for upcoming purchases.',
      industry: 'Retail',
      difficulty: 'Low',
      icon: <StoreIcon />,
      color: 'secondary',
      expectedStrategies: ['Product Recommend', 'Upsell', 'Empathetic'],
    },
    {
      id: 'tech_billing_issue',
      title: 'Billing Dispute',
      description: 'SaaS customer disputes unexpected charges on their monthly subscription bill.',
      industry: 'Tech',
      difficulty: 'Medium',
      icon: <ComputerIcon />,
      color: 'success',
      expectedStrategies: ['Apologetic', 'Technical', 'Quick Resolution'],
    },
  ];

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Low': return 'success';
      case 'Medium': return 'warning';
      case 'High': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 3 }}>
        Demo Scenarios
      </Typography>

      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Test your trained models against realistic customer support scenarios across different industries.
        Each scenario is designed to challenge different aspects of customer interaction.
      </Typography>

      <Grid container spacing={3}>
        {scenarios.map((scenario) => (
          <Grid item xs={12} md={6} lg={4} key={scenario.id}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: `${scenario.color}.main`, mr: 2 }}>
                    {scenario.icon}
                  </Avatar>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" component="div">
                      {scenario.title}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                      <Chip
                        label={scenario.industry}
                        color={scenario.color as any}
                        size="small"
                      />
                      <Chip
                        label={scenario.difficulty}
                        color={getDifficultyColor(scenario.difficulty) as any}
                        size="small"
                        variant="outlined"
                      />
                    </Box>
                  </Box>
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {scenario.description}
                </Typography>

                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Expected Strategies:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                  {scenario.expectedStrategies.map((strategy) => (
                    <Chip
                      key={strategy}
                      label={strategy}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </CardContent>

              <Box sx={{ p: 2, pt: 0 }}>
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={<PlayIcon />}
                  color={scenario.color as any}
                >
                  Run Scenario
                </Button>
              </Box>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mt: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary.main">
                6
              </Typography>
              <Typography variant="h6" component="div">
                Total Scenarios
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                3
              </Typography>
              <Typography variant="h6" component="div">
                Industries
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                2
              </Typography>
              <Typography variant="h6" component="div">
                High Difficulty
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info.main">
                8
              </Typography>
              <Typography variant="h6" component="div">
                Strategy Types
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Scenarios;
