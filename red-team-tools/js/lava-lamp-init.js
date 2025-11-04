/**
 * Lava Lamp Background Initialization
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 */

(function() {
    'use strict';

    // Create lava lamp container
    function initLavaLamp() {
        // Check if already initialized
        if (document.querySelector('.lava-lamp-container')) {
            return;
        }

        const container = document.createElement('div');
        container.className = 'lava-lamp-container';
        container.innerHTML = `
            <!-- Blacklight Base -->
            <div class="blacklight-glow"></div>

            <!-- Lava Blobs - Layer 1 (Back) -->
            <div class="lava-blob blob-1 depth-layer-1"></div>
            <div class="lava-blob blob-4 depth-layer-1"></div>

            <!-- Lava Blobs - Layer 2 (Middle) -->
            <div class="lava-blob blob-2 depth-layer-2"></div>
            <div class="lava-blob blob-5 depth-layer-2"></div>

            <!-- Lava Blobs - Layer 3 (Front) -->
            <div class="lava-blob blob-3 depth-layer-3"></div>
            <div class="lava-blob blob-6 depth-layer-3"></div>

            <!-- UV Scanlines -->
            <div class="scanlines"></div>

            <!-- Blacklight Vignette -->
            <div class="blacklight-vignette"></div>
        `;

        // Insert at the beginning of body
        document.body.insertBefore(container, document.body.firstChild);
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initLavaLamp);
    } else {
        initLavaLamp();
    }
})();

// Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
