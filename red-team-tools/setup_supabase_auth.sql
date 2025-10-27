-- Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
--
-- Supabase Authentication Setup for Red Team Tools
-- Run this in your Supabase SQL Editor to enable authentication
--

-- 1. Enable Row Level Security (RLS) on all tables
ALTER TABLE IF EXISTS profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS user_logins ENABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS analytics ENABLE ROW LEVEL SECURITY;

-- 2. Create profiles table (if doesn't exist)
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    company TEXT,
    role TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

-- 3. Create user_logins table (if doesn't exist)
CREATE TABLE IF NOT EXISTS user_logins (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    logged_in_at TIMESTAMPTZ DEFAULT NOW(),
    ip_address TEXT DEFAULT 'unknown'
);

-- 4. Create analytics table (if doesn't exist)
CREATE TABLE IF NOT EXISTS analytics (
    id SERIAL PRIMARY KEY,
    event_name TEXT NOT NULL,
    event_data JSONB,
    ip_address TEXT DEFAULT 'unknown',
    user_agent TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- 5. RLS Policies for profiles table
DROP POLICY IF EXISTS "Users can view own profile" ON profiles;
CREATE POLICY "Users can view own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON profiles;
CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Enable insert for authenticated users only" ON profiles;
CREATE POLICY "Enable insert for authenticated users only"
    ON profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- 6. RLS Policies for user_logins (only users can see their own logins)
DROP POLICY IF EXISTS "Users can view own logins" ON user_logins;
CREATE POLICY "Users can view own logins"
    ON user_logins FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Enable insert for authenticated users only" ON user_logins;
CREATE POLICY "Enable insert for authenticated users only"
    ON user_logins FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- 7. RLS Policies for analytics (public insert, authenticated read)
DROP POLICY IF EXISTS "Enable insert for all users" ON analytics;
CREATE POLICY "Enable insert for all users"
    ON analytics FOR INSERT
    WITH CHECK (true);

DROP POLICY IF EXISTS "Enable read for authenticated users" ON analytics;
CREATE POLICY "Enable read for authenticated users"
    ON analytics FOR SELECT
    USING (auth.role() = 'authenticated');

-- 8. Function to create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, created_at)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
        NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 9. Trigger to auto-create profile on user signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();

-- 10. Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 11. Trigger to update timestamp on profile changes
DROP TRIGGER IF EXISTS update_profiles_updated_at ON profiles;
CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- 12. Grant permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;

-- Success message
SELECT 'Supabase authentication setup complete! ✅' AS status;
SELECT 'Tables created: profiles, user_logins, analytics' AS info;
SELECT 'RLS policies enabled and configured' AS security;
SELECT 'Triggers created for auto-profile creation' AS automation;
