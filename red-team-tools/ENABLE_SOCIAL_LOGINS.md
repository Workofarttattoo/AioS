# 🔐 Enable Social Logins (GitHub, Apple, Google)

You're at the **Authentication → Providers** window in Supabase. Perfect!

---

## 🎯 **Quick Overview**

Each provider requires OAuth credentials from their platform. Here's the order to do them (easiest to hardest):

1. **GitHub** (Easiest - 2 minutes)
2. **Google** (Medium - 5 minutes)
3. **Apple** (Hardest - 10+ minutes, requires Apple Developer account)

---

## 🐙 **GitHub Login (EASIEST)**

### **Step 1: Create GitHub OAuth App**

1. Go to: https://github.com/settings/developers
2. Click **"OAuth Apps"** in left sidebar
3. Click **"New OAuth App"**
4. Fill in:
   - **Application name:** Red Team Tools
   - **Homepage URL:** `https://red-team-tools.aios.is`
   - **Authorization callback URL:** `https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback`
5. Click **"Register application"**

### **Step 2: Get Client ID & Secret**

1. After creating, you'll see **Client ID** - copy it
2. Click **"Generate a new client secret"** - copy the secret immediately (only shown once!)

### **Step 3: Configure in Supabase**

Back in Supabase (where you are now):

1. Find **GitHub** provider
2. Toggle it **ON**
3. Paste **Client ID**
4. Paste **Client Secret**
5. Click **Save**

✅ **Done!** GitHub login is now enabled.

---

## 🔍 **Google Login (MEDIUM)**

### **Step 1: Create Google OAuth App**

1. Go to: https://console.cloud.google.com/
2. Create a new project (or select existing):
   - Click project dropdown at top
   - Click **"New Project"**
   - Name: "Red Team Tools"
   - Click **Create**

### **Step 2: Enable Google+ API**

1. In left sidebar, go to **"APIs & Services"** → **"Library"**
2. Search for: **"Google+ API"**
3. Click it, then click **"Enable"**

### **Step 3: Create OAuth Credentials**

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"+ CREATE CREDENTIALS"** → **"OAuth client ID"**
3. If prompted, configure OAuth consent screen first:
   - Click **"CONFIGURE CONSENT SCREEN"**
   - Choose **External**
   - Fill in:
     - App name: Red Team Tools
     - User support email: your email
     - Developer contact: your email
   - Click **Save and Continue**
   - Skip scopes (click **Save and Continue**)
   - Click **Back to Dashboard**

4. Now create OAuth client ID:
   - Application type: **Web application**
   - Name: Red Team Tools
   - Authorized redirect URIs: `https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback`
   - Click **Create**

### **Step 4: Get Client ID & Secret**

1. You'll see a popup with **Client ID** and **Client Secret** - copy both

### **Step 5: Configure in Supabase**

Back in Supabase:

1. Find **Google** provider
2. Toggle it **ON**
3. Paste **Client ID**
4. Paste **Client Secret**
5. Click **Save**

✅ **Done!** Google login is now enabled.

---

## 🍎 **Apple Login (HARDEST - Requires Apple Developer Account)**

**Prerequisites:**
- Apple Developer account ($99/year)
- If you don't have one, skip this for now

### **Step 1: Create Service ID**

1. Go to: https://developer.apple.com/account/resources/identifiers/list
2. Click **"+"** to add new identifier
3. Select **"Services IDs"**, click **Continue**
4. Fill in:
   - Description: Red Team Tools
   - Identifier: `com.corporationoflight.redteamtools` (must be unique)
5. Click **Continue**, then **Register**

### **Step 2: Configure Service ID**

1. Click on the Service ID you just created
2. Check **"Sign In with Apple"**
3. Click **Configure**
4. Fill in:
   - Primary App ID: (create one if you don't have)
   - Domains and Subdomains: `red-team-tools.aios.is`
   - Return URLs: `https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback`
5. Click **Save**, then **Continue**, then **Save**

### **Step 3: Create Private Key**

1. Go to: https://developer.apple.com/account/resources/authkeys/list
2. Click **"+"** to add new key
3. Fill in:
   - Key Name: Red Team Tools Auth Key
   - Check **"Sign In with Apple"**
   - Click **Configure**
   - Select your Service ID
   - Click **Save**
4. Click **Continue**, then **Register**
5. **Download the key file** (.p8 file) - you can only download once!
6. Note the **Key ID** shown

### **Step 4: Get Team ID**

1. Go to: https://developer.apple.com/account
2. Your **Team ID** is shown in top right corner

### **Step 5: Configure in Supabase**

Back in Supabase:

1. Find **Apple** provider
2. Toggle it **ON**
3. Fill in:
   - **Services ID:** `com.corporationoflight.redteamtools` (what you created)
   - **Team ID:** (from step 4)
   - **Key ID:** (from step 3)
   - **Private Key:** Open the .p8 file in a text editor, copy the entire contents
4. Click **Save**

✅ **Done!** Apple login is now enabled.

---

## 🎬 **After Enabling Providers**

### **Update Your Login Page**

Add social login buttons to `login.html`:

```html
<!-- After email/password form, before register link -->
<div style="margin: 20px 0; text-align: center; color: #999;">
    <span>── OR ──</span>
</div>

<button onclick="loginWithGitHub()" class="btn" style="background: #333;">
    🐙 Continue with GitHub
</button>

<button onclick="loginWithGoogle()" class="btn" style="background: #4285f4;">
    🔍 Continue with Google
</button>

<!-- Only if you enabled Apple -->
<button onclick="loginWithApple()" class="btn" style="background: #000;">
    🍎 Continue with Apple
</button>

<script>
async function loginWithGitHub() {
    const { data, error } = await supabaseClient.auth.signInWithOAuth({
        provider: 'github'
    });
}

async function loginWithGoogle() {
    const { data, error } = await supabaseClient.auth.signInWithOAuth({
        provider: 'google'
    });
}

async function loginWithApple() {
    const { data, error } = await supabaseClient.auth.signInWithOAuth({
        provider: 'apple'
    });
}
</script>
```

---

## ✅ **Verification**

Test each provider:

1. Go to your login page
2. Click "Continue with GitHub" (or Google/Apple)
3. Should redirect to OAuth provider
4. Authorize the app
5. Should redirect back to your dashboard
6. Check Supabase **Authentication → Users** - you should see the new user!

---

## 📋 **Quick Reference**

### **Callback URLs (same for all providers):**
```
https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback
```

### **Your URLs:**
- Homepage: `https://red-team-tools.aios.is`
- Login: `https://red-team-tools.aios.is/login.html`
- Dashboard: `https://red-team-tools.aios.is/dashboard.html`

---

## 🎯 **Recommended Order**

1. ✅ Email (you already did this)
2. ✅ **Start with GitHub** - easiest and fastest
3. ✅ **Then Google** - widely used
4. ⏸️ **Skip Apple for now** unless you have Apple Developer account

---

## 🚨 **Common Issues**

### **"Redirect URI mismatch"**
→ Make sure callback URL is exactly: `https://trokobwiphidmrmhwkni.supabase.co/auth/v1/callback`

### **"Client ID invalid"**
→ Double-check you copied the correct Client ID (no extra spaces)

### **Google: "Access blocked"**
→ You need to publish your OAuth consent screen or add test users

### **Apple: "Invalid client"**
→ Verify Service ID and Team ID are correct

---

## ✅ **You're Done When:**

- [ ] GitHub provider enabled in Supabase
- [ ] Google provider enabled in Supabase  
- [ ] (Optional) Apple provider enabled
- [ ] Social login buttons added to login.html
- [ ] Tested each provider successfully
- [ ] New users appear in Supabase Authentication → Users

---

**Start with GitHub - it's the quickest and easiest!** 🚀
