/**
 * Visitor Counter for Red Team Tools
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 * Starts at 1000
 */

(function() {
    // Configuration
    const START_COUNT = 1000;
    const SITE_NAME = window.location.hostname || 'red-team-tools';
    const STORAGE_KEY = 'visitor_count_' + SITE_NAME.replace(/\./g, '_');
    const SESSION_KEY = 'visitor_session_' + SITE_NAME.replace(/\./g, '_');

    // Get or initialize count
    let visitorCount = localStorage.getItem(STORAGE_KEY);
    visitorCount = visitorCount ? parseInt(visitorCount) : START_COUNT;

    // Check if new visitor (no session)
    if (!sessionStorage.getItem(SESSION_KEY)) {
        visitorCount++;
        localStorage.setItem(STORAGE_KEY, visitorCount);
        sessionStorage.setItem(SESSION_KEY, 'visited');
    }

    // Create and inject counter element
    function createCounter() {
        const counterDiv = document.createElement('div');
        counterDiv.id = 'visitor-counter';
        counterDiv.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
            <span>Visitors: <strong>${visitorCount.toLocaleString()}</strong></span>
        `;

        // Style the counter
        counterDiv.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 50px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            font-family: -apple-system, system-ui, sans-serif;
            font-size: 14px;
            z-index: 9999;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideInUp 0.5s ease-out;
            cursor: pointer;
            transition: transform 0.3s;
        `;

        // Add hover effect
        counterDiv.onmouseover = function() {
            this.style.transform = 'scale(1.05)';
        };
        counterDiv.onmouseout = function() {
            this.style.transform = 'scale(1)';
        };

        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInUp {
                from {
                    transform: translateY(100px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            #visitor-counter:hover {
                animation: pulse 1s infinite;
            }
        `;
        document.head.appendChild(style);

        // Add to page
        document.body.appendChild(counterDiv);

        // Animate after load
        setTimeout(() => {
            counterDiv.style.animation = 'pulse 2s';
        }, 1000);
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createCounter);
    } else {
        createCounter();
    }

    // Expose count globally for debugging
    window.visitorCount = visitorCount;
})();