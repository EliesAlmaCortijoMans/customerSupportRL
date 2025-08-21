import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon,
  Settings as SettingsIcon,
  Code as CodeIcon,
} from '@mui/icons-material';

const Documentation: React.FC = () => {
  const sections = [
    {
      id: 'overview',
      title: 'Project Overview',
      icon: <PsychologyIcon />,
      content: (
        <Box>
          <Typography variant="body1" paragraph>
            This project evaluates Gymnasium as a framework for modeling interaction-based workflows 
            relevant to UltraLab's products. The prototype demonstrates a customer support agent 
            training environment that simulates realistic customer service interactions.
          </Typography>
          <Typography variant="h6" gutterBottom>Key Features:</Typography>
          <List>
            <ListItem>
              <ListItemText primary="Multi-industry support (BFSI, Retail, Tech)" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Comprehensive state and action spaces" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Business-relevant reward functions" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Vectorized and advanced environment variants" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Real-time training monitoring" />
            </ListItem>
          </List>
        </Box>
      ),
    },
    {
      id: 'environment',
      title: 'Environment Architecture',
      icon: <SettingsIcon />,
      content: (
        <Box>
          <Typography variant="h6" gutterBottom>State Space:</Typography>
          <List dense>
            <ListItem>
              <ListItemText 
                primary="Inquiry Type" 
                secondary="Customer inquiry category (15 types across industries)" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Customer Sentiment" 
                secondary="Emotional state (Angry, Frustrated, Neutral, Satisfied, Delighted)" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Customer Tier" 
                secondary="Value tier (Basic, Premium, VIP)" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Conversation Context" 
                secondary="Length, history, and encoded conversation state" 
              />
            </ListItem>
          </List>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Action Space:</Typography>
          <Grid container spacing={1}>
            {[
              'Empathetic', 'Technical', 'Escalate', 'Product Recommend',
              'Apologetic', 'Educational', 'Quick Resolution', 'Upsell'
            ].map((strategy) => (
              <Grid item key={strategy}>
                <Chip label={strategy} size="small" variant="outlined" />
              </Grid>
            ))}
          </Grid>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Reward Function:</Typography>
          <Typography variant="body2">
            Multi-component reward based on customer satisfaction, resolution efficiency, 
            and business impact. Scales with customer tier and includes penalties for poor outcomes.
          </Typography>
        </Box>
      ),
    },
    {
      id: 'training',
      title: 'Training & Algorithms',
      icon: <TrendingUpIcon />,
      content: (
        <Box>
          <Typography variant="h6" gutterBottom>Supported Algorithms:</Typography>
          <List>
            <ListItem>
              <ListItemText 
                primary="PPO (Proximal Policy Optimization)" 
                secondary="Default choice for stable training with good sample efficiency" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="A2C (Advantage Actor-Critic)" 
                secondary="Faster training for simpler scenarios" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="DQN (Deep Q-Network)" 
                secondary="Value-based approach for discrete action spaces" 
              />
            </ListItem>
          </List>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Training Features:</Typography>
          <List dense>
            <ListItem>
              <ListItemText primary="Vectorized environments for parallel training" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Curriculum learning for progressive difficulty" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Real-time metrics and progress tracking" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Automatic model saving and evaluation" />
            </ListItem>
          </List>
        </Box>
      ),
    },
    {
      id: 'api',
      title: 'API Reference',
      icon: <CodeIcon />,
      content: (
        <Box>
          <Typography variant="h6" gutterBottom>Core Endpoints:</Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" color="primary">Environment Management</Typography>
            <Box className="code-block" sx={{ mt: 1 }}>
              <Typography variant="body2" component="div" sx={{ fontFamily: 'monospace' }}>
                GET /environments - List all environments<br/>
                POST /environments - Create new environment<br/>
                POST /environments/&#123;id&#125;/reset - Reset environment<br/>
                POST /environments/&#123;id&#125;/step - Take action in environment
              </Typography>
            </Box>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" color="primary">Training Management</Typography>
            <Box className="code-block" sx={{ mt: 1 }}>
              <Typography variant="body2" component="div" sx={{ fontFamily: 'monospace' }}>
                GET /training/sessions - List training sessions<br/>
                POST /training/start - Start new training<br/>
                POST /training/sessions/&#123;id&#125;/stop - Stop training<br/>
                GET /training/sessions/&#123;id&#125;/metrics - Get training metrics
              </Typography>
            </Box>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" color="primary">Model Operations</Typography>
            <Box className="code-block" sx={{ mt: 1 }}>
              <Typography variant="body2" component="div" sx={{ fontFamily: 'monospace' }}>
                GET /models - List available models<br/>
                POST /models/&#123;id&#125;/evaluate - Evaluate model performance<br/>
                POST /models/&#123;id&#125;/predict - Get action prediction
              </Typography>
            </Box>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>WebSocket Events:</Typography>
          <List dense>
            <ListItem>
              <ListItemText 
                primary="environment_update" 
                secondary="Real-time environment state changes" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="training_update" 
                secondary="Training progress and completion events" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="model_update" 
                secondary="Model creation and evaluation results" 
              />
            </ListItem>
          </List>
        </Box>
      ),
    },
    {
      id: 'findings',
      title: 'Key Findings & Assessment',
      icon: <PsychologyIcon />,
      content: (
        <Box>
          <Typography variant="h6" gutterBottom>Gymnasium Strengths:</Typography>
          <List>
            <ListItem>
              <ListItemText 
                primary="Excellent framework structure" 
                secondary="Well-designed API that's easy to extend and customize" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Strong ecosystem integration" 
                secondary="Works seamlessly with stable-baselines3 and other RL libraries" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Vectorization support" 
                secondary="Built-in support for parallel environment execution" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Flexible observation/action spaces" 
                secondary="Supports complex business logic and state representations" 
              />
            </ListItem>
          </List>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Potential Limitations:</Typography>
          <List>
            <ListItem>
              <ListItemText 
                primary="Learning curve" 
                secondary="Requires RL expertise for optimal environment design" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Training time" 
                secondary="Complex environments may require significant computational resources" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="Debugging complexity" 
                secondary="Reward shaping and environment tuning can be challenging" 
              />
            </ListItem>
          </List>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Recommendations:</Typography>
          <Typography variant="body2" paragraph>
            Gymnasium is well-suited for UltraLab's prototyping needs, particularly for:
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText primary="Agent-user interaction modeling" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Behavior-based evaluation systems" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Adaptive testing and optimization workflows" />
            </ListItem>
          </List>
        </Box>
      ),
    },
  ];

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 3 }}>
        Project Documentation
      </Typography>

      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Comprehensive documentation for the Customer Support RL Environment prototype, 
        including technical details, findings, and recommendations for UltraLab.
      </Typography>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary.main">
                3
              </Typography>
              <Typography variant="h6" component="div">
                Story Points
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                15
              </Typography>
              <Typography variant="h6" component="div">
                Inquiry Types
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                8
              </Typography>
              <Typography variant="h6" component="div">
                Response Strategies
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info.main">
                3
              </Typography>
              <Typography variant="h6" component="div">
                RL Algorithms
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Documentation Sections */}
      <Box>
        {sections.map((section, index) => (
          <Accordion key={section.id} defaultExpanded={index === 0}>
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls={`${section.id}-content`}
              id={`${section.id}-header`}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {section.icon}
                <Typography variant="h6">{section.title}</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {section.content}
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>

      {/* Next Steps */}
      <Card sx={{ mt: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Next Steps & Future Development
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Short Term (1-2 weeks):
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="Expand scenario library" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Add model comparison tools" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Implement A/B testing framework" />
                </ListItem>
              </List>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Long Term (1-3 months):
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText primary="Multi-agent environments" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Integration with real customer data" />
                </ListItem>
                <ListItem>
                  <ListItemText primary="Production deployment framework" />
                </ListItem>
              </List>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Documentation;
