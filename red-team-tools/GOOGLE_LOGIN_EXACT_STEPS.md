# 🔍 Google Login - EXACT Steps (No Confusion)

## ⚡ FASTEST METHOD (Recommended)

You **DON'T** need to enable any API! Just create the OAuth client directly.

---

## 📋 **Step-by-Step (EXACT)**

### **1. Go to Google Cloud Console**
https://console.cloud.google.com/

### **2. Create/Select Project**
- Click project dropdown (top left, says "Select a project")
- Click **"NEW PROJECT"**
- Project name: `Red Team Tools`
- Click **CREATE**
- Wait 10 seconds for project to be created
- Make sure the new project is selected (check top bar)

### **3. Configure OAuth Consent Screen**
- In left sidebar, click **"APIs & Services"**
- Click **"OAuth consent screen"**
- Choose **"External"** (allows anyone to login)
- Click **CREATE**

Fill in ONLY these fields:
```
App name: Red Team Tools
User support email: [your email]
Developer contact information: [your email]
```

- Click **SAVE AND CONTINUE**
- On "Scopes" screen: Click **SAVE AND CONTINUE** (don't add any)
- On "Test users" screen: Click **SAVE AND CONTINUE** (skip this)
- Click **BACK TO DASHBOARD**

### **4. Create OAuth Client ID**
- In left sidebar: **"APIs & Services"** → **"Credentials"**
- Click **"+ CREATE CREDENTIALS"** (at top)
- Select **"OAuth client ID"**

Fill in:
```
Application type: Web application
Name: Red Team Tools Web Client
```

Under **"Authorized redirect URIs"**:
- Click **"+ ADD URI"**
- Paste: `https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback`
- Click **CREATE**

### **5. Copy Credentials**
You'll see a popup with:
- **Client ID** (long string starting with numbers)
- **Client secret** (shorter string)

**Copy both!** (You can also view them later)

### **6. Configure in Supabase**
Back in Supabase (where you are):
1. Find **Google** provider
2. Toggle it **ON**
3. Paste **Client ID** in the "Client ID" field
4. Paste **Client Secret** in the "Client Secret" field
5. Click **SAVE**

---

## ✅ **That's It!**

No APIs needed. No Google+ API. No Identity Services API.

Just OAuth consent screen + OAuth client = Done.

---

## 🐛 **Common Issues**

### **"Access blocked: This app's request is invalid"**
→ You didn't configure the OAuth consent screen
→ Go back to step 3

### **"Redirect URI mismatch"**
→ Make sure the callback URL is EXACTLY:
```
https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback
```
(no trailing slash, no extra spaces)

### **"This app hasn't been verified by Google"**
→ This is normal for testing!
→ Click "Advanced" → "Go to Red Team Tools (unsafe)"
→ To remove this warning, publish the app (optional, not needed for testing)

---

## 📸 **Visual Reference**

**OAuth Consent Screen should look like:**
```
App name: Red Team Tools
User support email: your@email.com
App domain: (leave blank)
Authorized domains: (leave blank)
Developer contact: your@email.com
```

**OAuth Client should look like:**
```
Application type: Web application
Name: Red Team Tools Web Client
Authorized redirect URIs:
  https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback
```

---

## 🚀 **Quick Links**

- **Google Cloud Console**: https://console.cloud.google.com/
- **Direct to Credentials**: https://console.cloud.google.com/apis/credentials
- **Direct to OAuth Consent**: https://console.cloud.google.com/apis/credentials/consent

---

## ⏱️ **Time to Complete**

- Configure consent screen: 1 minute
- Create OAuth client: 30 seconds
- Configure in Supabase: 30 seconds
- **Total: 2 minutes**

---

**You do NOT need to enable any APIs!** Just consent screen + OAuth client.
