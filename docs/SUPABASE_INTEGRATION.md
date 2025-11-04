# Supabase Integration Guide for Ai:oS Website

## Setup Instructions

### 1. Get Your Supabase Credentials

1. Go to [https://supabase.com](https://supabase.com)
2. Create a project or select existing project
3. Go to Project Settings > API
4. Copy your:
   - **Project URL**: `https://yourproject.supabase.co`
   - **Anon/Public Key**: `eyJhbGciOi...`

### 2. Environment Variables

Create a `.env` file (or add to your hosting platform):

```bash
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
```

### 3. Integration Code

Add this script to any page that needs authentication:

```html
<!-- Add to <head> -->
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>

<script>
// Initialize Supabase client
const SUPABASE_URL = 'YOUR_SUPABASE_URL';
const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY';
const supabase = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Check if user is logged in
async function checkAuth() {
    const { data: { user } } = await supabase.auth.getUser();
    if (user) {
        console.log('User is logged in:', user.email);
        return user;
    } else {
        console.log('User is not logged in');
        return null;
    }
}

// Sign up function
async function signUp(email, password) {
    const { data, error } = await supabase.auth.signUp({
        email: email,
        password: password,
    });
    if (error) {
        console.error('Error signing up:', error.message);
        return { success: false, error: error.message };
    }
    return { success: true, user: data.user };
}

// Sign in function
async function signIn(email, password) {
    const { data, error } = await supabase.auth.signInWithPassword({
        email: email,
        password: password,
    });
    if (error) {
        console.error('Error signing in:', error.message);
        return { success: false, error: error.message };
    }
    return { success: true, user: data.user };
}

// Sign out function
async function signOut() {
    const { error } = await supabase.auth.signOut();
    if (error) {
        console.error('Error signing out:', error.message);
        return { success: false, error: error.message };
    }
    return { success: true };
}

// Check auth on page load
window.addEventListener('DOMContentLoaded', async () => {
    const user = await checkAuth();
    if (!user) {
        // Redirect to login if needed
        // window.location.href = 'login.html';
    }
});
</script>
```

### 4. Example Login Form

```html
<div class="login-form">
    <h2>Login to Ai:oS</h2>
    <input type="email" id="email" placeholder="Email" />
    <input type="password" id="password" placeholder="Password" />
    <button onclick="handleLogin()">Login</button>
    <button onclick="handleSignUp()">Sign Up</button>
</div>

<script>
async function handleLogin() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    const result = await signIn(email, password);
    if (result.success) {
        alert('Login successful!');
        window.location.href = 'index.html';
    } else {
        alert('Login failed: ' + result.error);
    }
}

async function handleSignUp() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    const result = await signUp(email, password);
    if (result.success) {
        alert('Sign up successful! Check your email to confirm.');
    } else {
        alert('Sign up failed: ' + result.error);
    }
}
</script>
```

### 5. Database Setup (Optional)

Create tables for storing user data:

```sql
-- User profiles table
CREATE TABLE profiles (
    id UUID REFERENCES auth.users PRIMARY KEY,
    email TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own profile
CREATE POLICY "Users can view own profile"
ON profiles FOR SELECT
USING (auth.uid() = id);

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profile"
ON profiles FOR UPDATE
USING (auth.uid() = id);
```

### 6. Advanced Features

#### Email Confirmation

Enable in Supabase Dashboard: Authentication > Settings > Enable email confirmations

#### OAuth Providers

Enable in Supabase Dashboard: Authentication > Providers

Example GitHub OAuth:

```javascript
async function signInWithGitHub() {
    const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'github',
    });
}
```

#### Protected Routes

```javascript
// Add to each protected page
window.addEventListener('DOMContentLoaded', async () => {
    const user = await checkAuth();
    if (!user) {
        window.location.href = 'login.html';
    }
});
```

## Where to Add Supabase

- **quantum-visualizer.html** - Save user's circuit designs
- **index.html** - Track user engagement
- **algorithms.html** - Personalized content

## Security Best Practices

1. **Never expose your service_role key** - Only use anon/public key in browser
2. **Use Row Level Security (RLS)** - Protect your database tables
3. **Enable email confirmation** - Verify user emails
4. **Use HTTPS** - Always serve over secure connection
5. **Implement rate limiting** - Protect against abuse

## Testing

1. Sign up with test email
2. Check Supabase Dashboard > Authentication > Users
3. Verify email confirmation works
4. Test login/logout flow
5. Test protected routes

---

**Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.**
