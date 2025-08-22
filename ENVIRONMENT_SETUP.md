# Environment Variables Setup Guide

This guide shows you how to configure environment variables for both Railway (backend) and Vercel (frontend).

## 🚂 Railway Backend Variables

### How to Set Variables:
1. Go to [railway.app](https://railway.app)
2. Open your `customerSupportRL` project
3. Click on **Variables** tab
4. Click **+ New Variable** and add each one:

### Required Variables:
```
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
```

### CORS Configuration:
```
ALLOWED_ORIGINS=["https://your-vercel-app.vercel.app"]
```
⚠️ **Important**: Replace `your-vercel-app` with your actual Vercel domain

### Optional Variables:
```
ENABLE_TRAINING=true
ENABLE_EVALUATION=true
MAX_ENVIRONMENTS=10
MAX_TRAINING_SESSIONS=5
```

## 🔺 Vercel Frontend Variables

### How to Set Variables:
1. Go to [vercel.com](https://vercel.com)
2. Open your project
3. Go to **Settings** → **Environment Variables**
4. Click **Add New**

### Required Variable:
```
Name: REACT_APP_API_URL
Value: https://your-railway-app.up.railway.app
Environments: ✅ Production ✅ Preview ✅ Development
```

⚠️ **Important**: Replace `your-railway-app` with your actual Railway domain

## 📋 Step-by-Step Setup Process

### 1. Deploy Backend First
1. Set Railway variables (above)
2. Wait for deployment to complete
3. Note your Railway URL (e.g., `https://web-production-1234.up.railway.app`)

### 2. Configure Frontend
1. Set `REACT_APP_API_URL` in Vercel to your Railway URL
2. Redeploy frontend

### 3. Update CORS
1. Get your Vercel URL (e.g., `https://customer-support-rl.vercel.app`)
2. Update `ALLOWED_ORIGINS` in Railway to include your Vercel URL
3. Railway will auto-redeploy

## 🔗 Example URLs

### Railway Backend URLs:
- `https://customersupportrl-production.up.railway.app`
- `https://web-production-1234.up.railway.app`
- `https://shimmering-kindness-production.up.railway.app`

### Vercel Frontend URLs:
- `https://customer-support-rl.vercel.app`
- `https://customer-support-rl-git-main-username.vercel.app`
- `https://customer-support-rl-username.vercel.app`

## ✅ Testing Your Setup

### Backend Health Check:
Visit: `https://your-railway-url/health`
Should return: `{"status": "healthy", ...}`

### API Documentation:
Visit: `https://your-railway-url/docs`
Should show FastAPI documentation

### Frontend Connection:
1. Open your Vercel app
2. Check browser console for errors
3. Look for successful API calls to your Railway backend

## 🔧 Troubleshooting

### CORS Errors:
- Ensure `ALLOWED_ORIGINS` includes your exact Vercel domain
- Check for typos in URLs
- Make sure to include `https://`

### API Connection Errors:
- Verify `REACT_APP_API_URL` is set correctly
- Ensure Railway app is running
- Check Railway logs for errors

### Environment Variables Not Working:
- **Railway**: Variables take effect on next deployment
- **Vercel**: Must redeploy after adding variables
