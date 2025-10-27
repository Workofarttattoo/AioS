/**
 * Tool Authentication Helper
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 *
 * This script provides authentication for individual tool pages.
 * Include this after auth-config.js in your HTML.
 */

(function() {
    'use strict';

    // Check if auth-config is loaded
    if (typeof AUTH_CONFIG === 'undefined' || !supabaseClient) {
        console.error('❌ AUTH_CONFIG or supabaseClient not found. Make sure auth-config.js is loaded first.');
        return;
    }

    // Check authentication on page load
    async function checkAuth() {
        try {
            const { data: { session }, error } = await supabaseClient.auth.getSession();

            if (error) {
                console.error('Auth check error:', error);
                redirectToLogin();
                return;
            }

            if (!session) {
                console.log('No active session - redirecting to login');
                redirectToLogin();
                return;
            }

            // User is authenticated
            console.log('✅ User authenticated:', session.user.email);

            // Display user info if element exists
            const userEmailEl = document.getElementById('userEmail');
            if (userEmailEl) {
                userEmailEl.textContent = session.user.email;
            }

            const userNameEl = document.getElementById('userName');
            if (userNameEl && session.user.user_metadata?.full_name) {
                userNameEl.textContent = session.user.user_metadata.full_name;
            }

            // Setup logout button if exists
            const logoutBtn = document.getElementById('logoutBtn');
            if (logoutBtn) {
                logoutBtn.addEventListener('click', handleLogout);
            }

        } catch (err) {
            console.error('Authentication error:', err);
            redirectToLogin();
        }
    }

    function redirectToLogin() {
        // Store current page for redirect after login
        sessionStorage.setItem('redirectAfterLogin', window.location.pathname);
        window.location.href = '/login.html';
    }

    async function handleLogout() {
        try {
            const { error } = await supabaseClient.auth.signOut();
            if (error) throw error;

            console.log('✅ Logged out successfully');
            window.location.href = '/index.html';
        } catch (err) {
            console.error('Logout error:', err);
            // Force redirect anyway
            window.location.href = '/index.html';
        }
    }

    // Listen for auth state changes
    supabaseClient.auth.onAuthStateChange((event, session) => {
        console.log('Auth state changed:', event);

        if (event === 'SIGNED_OUT') {
            window.location.href = '/index.html';
        } else if (event === 'TOKEN_REFRESHED') {
            console.log('✅ Token refreshed');
        } else if (event === 'SIGNED_IN') {
            console.log('✅ User signed in');
        }
    });

    // Run auth check when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', checkAuth);
    } else {
        checkAuth();
    }

    // Export logout function for manual use
    window.toolAuth = {
        logout: handleLogout,
        checkAuth: checkAuth
    };

})();
