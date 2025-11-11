/**
 * Security Tools API Client
 * Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
 *
 * JavaScript client for connecting React components to the backend API.
 */

const API_BASE_URL = 'http://localhost:5000/api';
const API_KEY = 'demo_key_12345'; // Demo key - replace with user's actual key

/**
 * Make authenticated API request
 */
async function apiRequest(endpoint, method = 'GET', data = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    }
  };

  if (data && method !== 'GET') {
    options.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.message || result.error || 'API request failed');
    }

    return result;
  } catch (error) {
    console.error('[API Client] Error:', error);
    throw error;
  }
}

/**
 * SQLgps API Calls
 */
const SQLgpsAPI = {
  async scan(url, techniques) {
    return apiRequest('/sqlgps/scan', 'POST', { url, techniques });
  },

  async enumerate(target, database = null) {
    return apiRequest('/sqlgps/enumerate', 'POST', { target, database });
  }
};

/**
 * HashSolver API Calls
 */
const HashSolverAPI = {
  async crack(hash, algorithm, attackMode) {
    return apiRequest('/hashsolver/crack', 'POST', {
      hash,
      algorithm,
      attack_mode: attackMode
    });
  },

  async identify(hash) {
    return apiRequest('/hashsolver/identify', 'POST', { hash });
  }
};

/**
 * NMAP API Calls
 */
const NMAPStreetAPI = {
  async scan(target, scanType, ports) {
    return apiRequest('/nmap/scan', 'POST', {
      target,
      scan_type: scanType,
      ports
    });
  },

  async detectService(target, port) {
    return apiRequest('/nmap/service-detect', 'POST', { target, port });
  }
};

/**
 * BelchStudio API Calls
 */
const BelchStudioAPI = {
  async intercept(url, method = 'GET', headers = {}, body = null) {
    return apiRequest('/belchstudio/intercept', 'POST', {
      url,
      method,
      headers,
      body
    });
  }
};

/**
 * Health Check
 */
async function healthCheck() {
  return apiRequest('/health', 'GET');
}

// Export for use in React components
if (typeof window !== 'undefined') {
  window.SecurityToolsAPI = {
    SQLgps: SQLgpsAPI,
    HashSolver: HashSolverAPI,
    NMAPStreet: NMAPStreetAPI,
    BelchStudio: BelchStudioAPI,
    healthCheck
  };
}
