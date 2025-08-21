# Deployment Guide

This guide will help you deploy the Customer Support RL application to Vercel (frontend) and Railway (backend).

## Prerequisites

- GitHub account with repository access
- Vercel account
- Railway account

## Backend Deployment (Railway)

### 1. Deploy to Railway

1. **Sign up/Login to Railway**: Go to [railway.app](https://railway.app)

2. **Connect GitHub**: Link your GitHub account and select the `customerSupportRL` repository

3. **Deploy Backend**:
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect the Python app

4. **Configure Environment Variables** in Railway dashboard:
   ```
   ENVIRONMENT=production
   DEBUG=false
   LOG_LEVEL=info
   ALLOWED_ORIGINS=["https://your-vercel-app.vercel.app"]
   ENABLE_TRAINING=true
   ENABLE_EVALUATION=true
   MAX_ENVIRONMENTS=10
   MAX_TRAINING_SESSIONS=5
   ```

5. **Custom Start Command** (if needed):
   ```
   python run_server.py serve --host 0.0.0.0 --port $PORT --reload false --workers 1
   ```

6. **Domain**: Railway will provide a domain like `https://your-app.railway.app`

### 2. Railway Configuration Files

The following files are already configured:
- `railway.toml` - Railway deployment configuration
- `nixpacks.toml` - Build configuration
- `requirements-prod.txt` - Production dependencies
- `production.env` - Environment variables template

## Frontend Deployment (Vercel)

### 1. Deploy to Vercel

1. **Sign up/Login to Vercel**: Go to [vercel.com](https://vercel.com)

2. **Import Project**:
   - Click "New Project"
   - Import from your GitHub repository
   - Select the repository

3. **Configure Build Settings**:
   - **Build Command**: `cd frontend && npm ci && npm run build`
   - **Output Directory**: `frontend/build`
   - **Install Command**: `cd frontend && npm install`

4. **Environment Variables** in Vercel dashboard:
   ```
   REACT_APP_API_URL=https://your-backend.railway.app
   ```

5. **Deploy**: Click "Deploy"

### 2. Vercel Configuration

The `vercel.json` file is already configured with the correct settings.

## Post-Deployment Steps

### 1. Update CORS Settings

After deployment, update the Railway environment variables with your actual Vercel domain:

```
ALLOWED_ORIGINS=["https://your-actual-vercel-app.vercel.app"]
```

### 2. Test the Deployment

1. **Backend Health Check**: Visit `https://your-backend.railway.app/health`
2. **API Documentation**: Visit `https://your-backend.railway.app/docs`
3. **Frontend**: Visit your Vercel app URL

### 3. Connect Frontend to Backend

The frontend will automatically connect to the backend using the `REACT_APP_API_URL` environment variable.

## Environment Variables Reference

### Railway (Backend)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Environment name | `development` | No |
| `DEBUG` | Enable debug mode | `true` | No |
| `LOG_LEVEL` | Logging level | `info` | No |
| `ALLOWED_ORIGINS` | CORS allowed origins | `["*"]` | Yes (Production) |
| `ENABLE_TRAINING` | Enable training endpoints | `true` | No |
| `ENABLE_EVALUATION` | Enable evaluation endpoints | `true` | No |
| `MAX_ENVIRONMENTS` | Max concurrent environments | `10` | No |
| `MAX_TRAINING_SESSIONS` | Max concurrent training | `5` | No |

### Vercel (Frontend)

| Variable | Description | Required |
|----------|-------------|----------|
| `REACT_APP_API_URL` | Backend API URL | Yes |

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure `ALLOWED_ORIGINS` includes your Vercel domain
2. **Build Failures**: Check that all dependencies are in `requirements-prod.txt`
3. **Memory Issues**: Railway has memory limits; use production requirements
4. **Timeout Issues**: Increase Railway timeout settings if needed

### Logs

- **Railway**: Check deployment logs in Railway dashboard
- **Vercel**: Check build logs in Vercel dashboard
- **Application**: Check runtime logs in Railway dashboard

## Production Optimizations

### Backend (Railway)

1. Use production dependencies (`requirements-prod.txt`)
2. Disable debug mode and auto-reload
3. Set appropriate log levels
4. Configure memory and CPU limits
5. Set up health checks

### Frontend (Vercel)

1. Optimize build output
2. Enable gzip compression (automatic on Vercel)
3. Use environment-specific API URLs
4. Configure caching headers

## Monitoring and Maintenance

1. **Health Checks**: Both platforms provide built-in monitoring
2. **Error Tracking**: Consider adding Sentry or similar
3. **Performance**: Monitor API response times and frontend performance
4. **Costs**: Monitor usage on both platforms

## Custom Domains (Optional)

### Railway
1. Go to Settings → Domains
2. Add your custom domain
3. Configure DNS records

### Vercel
1. Go to Project Settings → Domains
2. Add your custom domain
3. Configure DNS records

---

Your application should now be live! 🚀

- **Frontend**: https://your-app.vercel.app
- **Backend**: https://your-app.railway.app
- **API Docs**: https://your-app.railway.app/docs
