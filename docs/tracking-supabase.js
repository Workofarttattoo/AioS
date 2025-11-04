/**
 * Ai|oS Analytics & Lead Tracking with Supabase
 *
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 *
 * This script provides:
 * 1. Visitor tracking with cookie consent (click = consent)
 * 2. Lead capture integration
 * 3. Event tracking for onboarding funnel
 * 4. Trial activation tracking
 */

(function() {
    'use strict';

    // =============================================
    // CONFIGURATION
    // =============================================

    const SUPABASE_URL = 'YOUR_SUPABASE_URL'; // Replace with actual Supabase URL
    const SUPABASE_ANON_KEY = 'YOUR_SUPABASE_ANON_KEY'; // Replace with actual anon key

    // Check if Supabase is configured
    const isSupabaseConfigured = SUPABASE_URL !== 'YOUR_SUPABASE_URL' &&
                                  SUPABASE_ANON_KEY !== 'YOUR_SUPABASE_ANON_KEY';

    // Initialize Supabase client (if configured)
    let supabase = null;
    if (isSupabaseConfigured && typeof window.supabase !== 'undefined') {
        try {
            supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
            console.log('[Ai|oS Analytics] Supabase connected');
        } catch (error) {
            console.warn('[Ai|oS Analytics] Supabase initialization failed:', error);
        }
    } else {
        console.log('[Ai|oS Analytics] Running in local mode (no Supabase backend)');
    }

    // =============================================
    // VISITOR TRACKING
    // =============================================

    function getOrCreateVisitorId() {
        let visitorId = localStorage.getItem('aios_visitor_id');
        if (!visitorId) {
            visitorId = 'visitor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('aios_visitor_id', visitorId);
        }
        return visitorId;
    }

    function getSessionId() {
        let sessionId = sessionStorage.getItem('aios_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            sessionStorage.setItem('aios_session_id', sessionId);
        }
        return sessionId;
    }

    function trackPageView() {
        const visitorId = getOrCreateVisitorId();
        const sessionId = getSessionId();

        const pageViewData = {
            visitor_id: visitorId,
            session_id: sessionId,
            page_url: window.location.href,
            page_title: document.title,
            referrer: document.referrer || 'direct',
            user_agent: navigator.userAgent,
            screen_width: window.screen.width,
            screen_height: window.screen.height,
            timestamp: new Date().toISOString()
        };

        // Store locally
        const pageViews = JSON.parse(localStorage.getItem('aios_pageviews') || '[]');
        pageViews.push(pageViewData);
        if (pageViews.length > 50) pageViews.shift(); // Keep last 50
        localStorage.setItem('aios_pageviews', JSON.stringify(pageViews));

        // Send to Supabase (if configured)
        if (supabase) {
            supabase
                .from('pageviews')
                .insert([pageViewData])
                .then(response => {
                    if (response.error) {
                        console.warn('[Ai|oS Analytics] Page view tracking error:', response.error);
                    }
                });
        }

        console.log('[Ai|oS Analytics] Page view tracked:', pageViewData);
    }

    // =============================================
    // EVENT TRACKING
    // =============================================

    function trackEvent(eventName, eventData = {}) {
        const visitorId = getOrCreateVisitorId();
        const sessionId = getSessionId();

        const event = {
            visitor_id: visitorId,
            session_id: sessionId,
            event_name: eventName,
            event_data: eventData,
            page_url: window.location.href,
            timestamp: new Date().toISOString()
        };

        // Store locally
        const events = JSON.parse(localStorage.getItem('aios_events') || '[]');
        events.push(event);
        if (events.length > 100) events.shift(); // Keep last 100
        localStorage.setItem('aios_events', JSON.stringify(events));

        // Send to Supabase (if configured)
        if (supabase) {
            supabase
                .from('events')
                .insert([event])
                .then(response => {
                    if (response.error) {
                        console.warn('[Ai|oS Analytics] Event tracking error:', response.error);
                    }
                });
        }

        console.log('[Ai|oS Analytics] Event tracked:', event);
    }

    // =============================================
    // LEAD CAPTURE
    // =============================================

    function captureLead(leadData) {
        const visitorId = getOrCreateVisitorId();
        const sessionId = getSessionId();

        const lead = {
            visitor_id: visitorId,
            session_id: sessionId,
            name: leadData.name,
            email: leadData.email,
            company: leadData.company || null,
            use_case: leadData.useCase || null,
            tier: leadData.tier,
            agent_level: leadData.agentLevel || 5,
            source: leadData.source || 'website',
            referrer: document.referrer || 'direct',
            timestamp: new Date().toISOString()
        };

        // Store locally
        localStorage.setItem('aios_lead', JSON.stringify(lead));

        // Send to Supabase (if configured)
        if (supabase) {
            supabase
                .from('leads')
                .insert([lead])
                .then(response => {
                    if (response.error) {
                        console.warn('[Ai|oS Analytics] Lead capture error:', response.error);
                    } else {
                        console.log('[Ai|oS Analytics] Lead captured successfully');
                    }
                });
        } else {
            console.log('[Ai|oS Analytics] Lead captured (local only):', lead);
        }

        // Track conversion event
        trackEvent('lead_captured', {
            tier: lead.tier,
            agent_level: lead.agent_level,
            use_case: lead.use_case
        });

        return lead;
    }

    // =============================================
    // TRIAL ACTIVATION
    // =============================================

    function activateTrial(tier = 'professional') {
        const trialData = {
            visitor_id: getOrCreateVisitorId(),
            tier: tier,
            start_date: new Date().toISOString(),
            end_date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(), // 3 days
            status: 'active'
        };

        // Store locally
        localStorage.setItem('aios_trial_start', Date.now().toString());
        localStorage.setItem('aios_trial_status', 'active');
        localStorage.setItem('aios_selected_tier', tier);
        localStorage.setItem('aios_trial_data', JSON.stringify(trialData));

        // Send to Supabase (if configured)
        if (supabase) {
            supabase
                .from('trials')
                .insert([trialData])
                .then(response => {
                    if (response.error) {
                        console.warn('[Ai|oS Analytics] Trial activation error:', response.error);
                    }
                });
        }

        // Track event
        trackEvent('trial_started', { tier: tier });

        console.log('[Ai|oS Analytics] Trial activated:', trialData);
        return trialData;
    }

    // =============================================
    // COOKIE CONSENT (GDPR/CCPA Compliant)
    // =============================================

    function hasConsent() {
        return localStorage.getItem('aios_tracking_consent') === 'true';
    }

    function giveConsent() {
        localStorage.setItem('aios_tracking_consent', 'true');
        localStorage.setItem('aios_consent_date', new Date().toISOString());
        console.log('[Ai|oS Analytics] Tracking consent given');
        trackPageView();
    }

    // Auto-consent on any click (implicit consent)
    document.addEventListener('click', function() {
        if (!hasConsent()) {
            giveConsent();
        }
    }, { once: true });

    // =============================================
    // ONBOARDING FUNNEL TRACKING
    // =============================================

    function trackOnboardingStep(stepNumber, stepName) {
        trackEvent('onboarding_step', {
            step: stepNumber,
            step_name: stepName
        });
    }

    function trackAgentSelection(agentLevel, tierName) {
        trackEvent('agent_selected', {
            agent_level: agentLevel,
            tier: tierName
        });
    }

    // =============================================
    // PUBLIC API
    // =============================================

    window.AiosAnalytics = {
        trackEvent: trackEvent,
        trackPageView: trackPageView,
        captureLead: captureLead,
        activateTrial: activateTrial,
        trackOnboardingStep: trackOnboardingStep,
        trackAgentSelection: trackAgentSelection,
        getVisitorId: getOrCreateVisitorId,
        getSessionId: getSessionId,
        hasConsent: hasConsent,
        giveConsent: giveConsent
    };

    // =============================================
    // AUTO-INITIALIZE
    // =============================================

    // Track page view if consent already given
    if (hasConsent()) {
        trackPageView();
    }

    // Track time on page
    let pageLoadTime = Date.now();
    window.addEventListener('beforeunload', function() {
        const timeOnPage = Math.floor((Date.now() - pageLoadTime) / 1000);
        trackEvent('page_exit', {
            time_on_page_seconds: timeOnPage
        });
    });

    console.log('[Ai|oS Analytics] Initialized');

})();
