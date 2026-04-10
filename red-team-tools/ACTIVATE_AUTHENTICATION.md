# 🔐 Activate Authentication for Red Team Tools

**Status**: Authentication code is ready, but Supabase backend needs activation.

---

## 🎯 Quick Fix

Your authentication system IS already coded and configured. The issue is that Supabase needs to have the database tables created. Here's how to activate it:

---

## ✅ **Step 1: Go to Supabase Dashboard**

1. Visit: https://supabase.com/dashboard
2. Login to your account
3. Select project: **red-team-tools** (or the project with URL `trokobwiphidmrmhwkni.supabase.co`)

---

## ✅ **Step 2: Enable Email Authentication**

1. In Supabase dashboard, go to **Authentication** → **Providers**
2. Find **Email** provider
3. Toggle it **ON** (if not already enabled)
4. Settings:
   - ✅ Enable email provider
   - ✅ Confirm email: **Enabled** (recommended)
   - ✅ Double confirm email: Optional
   - ✅ Secure email change: Enabled

5. **Save Changes**

---

## ✅ **Step 3: Create Database Tables**

1. In Supabase dashboard, go to **SQL Editor**
2. Click **New query**
3. Copy and paste the entire contents of `setup_supabase_auth.sql` (in this directory)
4. Click **Run**

This will create:
- `profiles` table - User profile data
- `user_logins` table - Login tracking
- `analytics` table - Event logging
- RLS (Row Level Security) policies
- Auto-triggers for profile creation

---

## ✅ **Step 4: Configure Email Templates (Optional but Recommended)**

1. Go to **Authentication** → **Email Templates**
2. Customize these templates:
   - **Confirm signup** - Sent when user registers
   - **Reset password** - Sent for password resets
   - **Magic Link** - For passwordless login (if you enable it later)

Example confirm signup template:
```html
<h2>Welcome to Red Team Tools!</h2>
<p>Click the link below to verify your email:</p>
<p><a href="{{ .ConfirmationURL }}">Verify Email</a></p>
```

---

## ✅ **Step 5: Test the System**

### **Option A: Test Locally**

1. Open `login.html` in a browser:
   ```bash
   cd ./red-team-tools
   open login.html
   ```

2. Try to register a new account
3. Check your email for verification link
4. Click verification link
5. Login with your credentials

### **Option B: Test on Live Site**

1. Visit: https://red-team-tools.aios.is/login.html
2. Register → Verify → Login

---

## 🐛 **Troubleshooting**

### **"Failed to fetch" error at login:**
✅ **FIXED!** The login.html and auth.js files have been updated to handle IP address fetch timeouts.

### **"Table 'profiles' does not exist":**
→ Run the `setup_supabase_auth.sql` script in Supabase SQL Editor (Step 3 above)

### **Email verification not working:**
→ Check Supabase **Authentication** → **Email Templates**
→ Verify SMTP settings (Supabase uses their SMTP by default)

### **RLS policy errors:**
→ The SQL script creates all necessary policies
→ If you get errors, go to Supabase → **Authentication** → **Policies** and verify they exist

### **Can't login after registering:**
→ Check if email confirmation is required
→ Go to **Authentication** → **Settings** and check "Enable email confirmations"
→ If disabled, users can login immediately after registering

---

## 📋 **What's Already Done**

✅ Authentication JavaScript code (`auth.js`)
✅ Configuration file (`auth-config.js`)
✅ Login page (`login.html`) - **FIXED fetch issue**
✅ Register page (`register.html`)
✅ Dashboard page (`dashboard.html`)
✅ Password reset page (`reset-password.html`)
✅ Email verification page (`verify-email.html`)
✅ Supabase project created
✅ API keys configured

---

## 🎬 **What You Need to Do**

1. ⏹️ Run `setup_supabase_auth.sql` in Supabase SQL Editor
2. ⏹️ Enable Email provider in Supabase (if not already)
3. ⏹️ Test registration/login flow

**That's it!** Once you run the SQL script, authentication will be fully functional.

---

## 🔑 **Your Supabase Credentials**

**Project URL**: https://trokobwiphidmrmhwkni.supabase.co
**Anon Key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (already in auth-config.js)

---

## 🚀 **Advanced: Add More Features**

Once basic auth is working, you can add:

### **1. Social Login (Google, GitHub, etc.)**
- Go to **Authentication** → **Providers**
- Enable Google/GitHub/etc.
- Add OAuth credentials

### **2. Magic Links (Passwordless)**
- Already supported by Supabase
- Just enable in **Authentication** → **Providers**

### **3. Two-Factor Authentication (2FA)**
- Enable in **Authentication** → **Settings**
- Requires SMS or TOTP app

### **4. Session Management**
- Configure session timeout in **Authentication** → **Settings**
- Default: 7 days

---

## 📞 **Need Help?**

If you run the SQL script and still get errors:

1. Check Supabase logs: **Logs** → **Postgres Logs**
2. Verify your project is on the Free tier (has database access)
3. Make sure you're running the SQL in the correct project

---

## ✅ **Verification Checklist**

After running setup:

- [ ] SQL script ran without errors
- [ ] Tables exist: `profiles`, `user_logins`, `analytics`
- [ ] RLS policies are enabled
- [ ] Email provider is ON
- [ ] Can register a new user
- [ ] Receive verification email
- [ ] Can verify email
- [ ] Can login successfully
- [ ] Dashboard loads after login

---

**Once you complete Step 3 (run SQL script), your authentication system will be 100% functional!** 🎉
