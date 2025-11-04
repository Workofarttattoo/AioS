/**
 * Supabase Authentication Configuration - Ai:oS
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 */

const AUTH_CONFIG = {
    PROJECT_NAME: 'aios',
    SUPABASE_URL: 'https://cszoklkfdszqsxhufhhj.supabase.co',
    SUPABASE_ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzem9rbGtmZHN6cXN4aHVmaGhqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjExNzI0MzAsImV4cCI6MjA3Njc0ODQzMH0.HdqXrWVTPCQ2NYH-5ED_nx91a38UGPvTHjva4NzBG8I',
    REDIRECT_URLS: {
        LOGIN: '/index.html',
        VERIFY: '/docs/verify-email.html',
        RESET: '/docs/reset-password.html'
    },
    EMAIL_CONFIG: {
        FROM: 'noreply@aios.is',
        VERIFICATION_SUBJECT: 'Verify your Ai:oS account',
        RESET_SUBJECT: 'Reset your Ai:oS password'
    },
    TRIAL: {
        DAYS: 30,
        CASES_LIMIT: 1000
    }
};
