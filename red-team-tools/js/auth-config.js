/**
 * Supabase Authentication Configuration - Red Team Tools
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 */

const AUTH_CONFIG = {
    PROJECT_NAME: 'red-team-tools',
    SUPABASE_URL: 'https://trokobwiphidmrmhwkni.supabase.co',
    SUPABASE_ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRyb2tvYndpcGhpZG1ybWh3a25pIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2NTk4MTQsImV4cCI6MjA3NjIzNTgxNH0.D1iTVxtL481Tk6Jr7qSInjOOCZhWmuHT8g-cE_ZT-dM',
    REDIRECT_URLS: {
        LOGIN: '/dashboard.html',
        VERIFY: '/verify-email.html',
        RESET: '/reset-password.html'
    },
    EMAIL_CONFIG: {
        FROM: 'noreply@red-team-tools.aios.is',
        VERIFICATION_SUBJECT: 'Verify your Red Team Tools account',
        RESET_SUBJECT: 'Reset your Red Team Tools password'
    }
};
