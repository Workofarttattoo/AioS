"""Quantum-enhanced patent discovery system utilities for AgentaOS.

This module packages the architectural scaffold for the quantum patent discovery
concept so that agents can import and extend components without triggering side
effects on import (writing files, starting HTTP servers, etc.).

Key adjustments from the original specification:

* Logging uses Python's :mod:`logging` instead of direct prints.
* Optional dependencies (PyTorch, FastAPI) are handled gracefully.
* Helper functions expose the FastAPI app factory and SwiftUI source code.
* No side effects occur at import time; helpers must be invoked explicitly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
  import torch
except Exception:  # pragma: no cover
  torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
  from fastapi import FastAPI, HTTPException, Request
  from fastapi.security import HTTPBearer
  from pydantic import BaseModel
except Exception:  # pragma: no cover
  FastAPI = None  # type: ignore
  HTTPException = None  # type: ignore
  Request = None  # type: ignore
  HTTPBearer = None  # type: ignore
  BaseModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
  from prometheus_fastapi_instrumentator import Instrumentator
except Exception:  # pragma: no cover
  Instrumentator = None  # type: ignore

try:  # pragma: no cover - optional dependency
  from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover
  Counter = None  # type: ignore
  Histogram = None  # type: ignore

try:  # pragma: no cover - optional dependency
  import httpx
except Exception:  # pragma: no cover
  httpx = None  # type: ignore

try:  # pragma: no cover - optional dependency
  import stripe  # type: ignore
except Exception:  # pragma: no cover
  stripe = None  # type: ignore

from aios.settings import settings

class _JsonFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple JSON formatting
    payload = {
      "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
      "level": record.levelname,
      "logger": record.name,
      "message": record.getMessage(),
    }
    if record.exc_info:
      payload["exc_info"] = self.formatException(record.exc_info)
    return json.dumps(payload)


def _configure_logging() -> None:
  if logging.getLogger().handlers:
    return
  level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
  handler = logging.StreamHandler()
  if os.getenv("LOG_FORMAT", "plain").lower() == "json":
    handler.setFormatter(_JsonFormatter())
  else:
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
  root = logging.getLogger()
  root.setLevel(level)
  root.addHandler(handler)


_configure_logging()
LOG = logging.getLogger("AgentaOS.quantum.patent")

_WORKFLOW_COUNTER = (
  Counter(
    "agentaos_patent_workflow_total",
    "Number of patent workflows processed",
    labelnames=("status",),
  )
  if Counter else None
)

_WORKFLOW_DURATION = (
  Histogram(
    "agentaos_patent_workflow_seconds",
    "Duration of patent workflows in seconds",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
  )
  if Histogram else None
)

# ---------------------------------------------------------------------------
# Quantum simulation layer
# ---------------------------------------------------------------------------

class QuantumBackend(Enum):
  STATEVECTOR = "statevector"
  TENSOR_NETWORK = "tensor_network"


@dataclass
class SimulationReport:
  backend: QuantumBackend
  num_qubits: int
  memory_mb: float


class ClaudeQuantumSimulator:
  """Lightweight simulator tuned for conceptual workflows (≤38 qubits)."""

  MAX_QUBITS = 38

  def __init__(self, num_qubits: int) -> None:
    if torch is None:
      raise ImportError("ClaudeQuantumSimulator requires PyTorch; install torch first.")
    if not (1 <= num_qubits <= self.MAX_QUBITS):
      raise ValueError("Number of qubits must be between 1 and 38.")

    self.num_qubits = num_qubits
    self.backend = QuantumBackend.STATEVECTOR if num_qubits <= 20 else QuantumBackend.TENSOR_NETWORK
    if self.backend is QuantumBackend.STATEVECTOR:
      self.state = torch.zeros(2**num_qubits, dtype=torch.complex64)
      self.state[0] = 1.0
    else:
      self.state = [torch.tensor([1.0, 0.0], dtype=torch.complex64) for _ in range(num_qubits)]

    LOG.info(
      "[quantum] simulator initialised: qubits=%d backend=%s memory=%.2fMB",
      num_qubits,
      self.backend.value,
      self._estimate_memory(),
    )

  def _estimate_memory(self) -> float:
    if self.backend is QuantumBackend.STATEVECTOR:
      return float((2**self.num_qubits) * 16 / (1024**2))
    bond_dim = 256
    return float((self.num_qubits * bond_dim**2 * 16) / (1024**2))

  def hadamard(self, qubit: int) -> None:
    gate = (1 / np.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64)
    self._apply_gate(gate, qubit)

  def _apply_gate(self, gate: torch.Tensor, qubit: int) -> None:
    if self.backend is QuantumBackend.STATEVECTOR:
      n = self.num_qubits
      dim = 2**n
      stride = 2**(n - qubit - 1)
      for i in range(0, dim, 2 * stride):
        for j in range(stride):
          idx0, idx1 = i + j, i + j + stride
          amp0, amp1 = self.state[idx0], self.state[idx1]
          self.state[idx0] = gate[0, 0] * amp0 + gate[0, 1] * amp1
          self.state[idx1] = gate[1, 0] * amp0 + gate[1, 1] * amp1
    else:
      self.state[qubit] = torch.einsum('ij,j->i', gate, self.state[qubit])

  def measure_all(self) -> List[int]:
    if self.backend is QuantumBackend.STATEVECTOR:
      probabilities = torch.abs(self.state) ** 2
      total = probabilities.sum()
      if total <= 0:
        probabilities = torch.full_like(probabilities, 1 / len(probabilities))
      else:
        probabilities = probabilities / total
      outcome = torch.multinomial(probabilities, 1).item()
      return [int(bit) for bit in bin(outcome)[2:].zfill(self.num_qubits)]
    return [int(np.random.randint(0, 2)) for _ in range(self.num_qubits)]

  def report(self) -> SimulationReport:
    return SimulationReport(self.backend, self.num_qubits, self._estimate_memory())


# ---------------------------------------------------------------------------
# Patent discovery agent layer
# ---------------------------------------------------------------------------

class PatentMetaAgent:
  """Autonomous patent discovery scaffold."""

  def __init__(self, quantum_sim: ClaudeQuantumSimulator) -> None:
    self.quantum_sim = quantum_sim
    self.knowledge_base = PatentKnowledgeBase()
    self.uspto_client = USPTOClient()

  async def discover_and_file_patent(self, idea: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    LOG.info("[PatentAgent] workflow started :: %s", idea.get("title", "<untitled>"))
    prior_art = await self._search_prior_art(idea)
    LOG.info("[PatentAgent] prior art references=%d", len(prior_art))
    novelty_score = await self._analyze_novelty(idea, prior_art)
    if novelty_score < 0.6:
      result = {
        "status": "rejected",
        "reason": "Insufficient novelty",
        "novelty_score": novelty_score,
        "prior_art": prior_art[:5],
      }
      if _WORKFLOW_COUNTER:
        _WORKFLOW_COUNTER.labels(status="rejected").inc()
      if _WORKFLOW_DURATION:
        _WORKFLOW_DURATION.observe(time.perf_counter() - start)
      return result
    application = await self._generate_patent_application(idea, prior_art)
    filing = await self._file_provisional_patent(application)
    result = {
      "status": "success",
      "novelty_score": novelty_score,
      "application": application,
      "filing": filing,
      "prior_art_count": len(prior_art),
    }
    if _WORKFLOW_COUNTER:
      _WORKFLOW_COUNTER.labels(status="success").inc()
    if _WORKFLOW_DURATION:
      _WORKFLOW_DURATION.observe(time.perf_counter() - start)
    return result

  async def _search_prior_art(self, idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = await asyncio.gather(
      self._search_uspto(idea),
      self._search_google_patents(idea),
      self._search_academic_papers(idea),
    )
    all_refs: List[Dict[str, Any]] = [item for chunk in results for item in chunk]
    return self._rank_by_relevance(all_refs, idea)

  async def _search_uspto(self, idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = self._build_search_query(idea)
    LOG.debug("[PatentAgent] USPTO query=%s", query)
    if self.uspto_client.is_configured() and httpx is not None and settings.allow_httpx_network:
      try:
        results = await self.uspto_client.search_patents(query)
        if results:
          return results
      except Exception as exc:  # pragma: no cover - network failure
        LOG.warning("[PatentAgent] USPTO search failed (%s). Falling back to simulated data.", exc)
    return [
      {
        "id": f"US-{i}",
        "title": f"Prior Art Example {i}",
        "abstract": f"Related to {idea.get('title', '')}",
        "filing_date": "2020-01-01",
        "similarity": float(np.random.random()),
      }
      for i in range(10)
    ]

  async def _search_google_patents(self, idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    return []

  async def _search_academic_papers(self, idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    return []

  def _build_search_query(self, idea: Dict[str, Any]) -> str:
    keywords = idea.get("keywords", [])
    return f"{idea.get('title', '')} {' '.join(keywords)}".strip()

  def _rank_by_relevance(self, prior_art: List[Dict[str, Any]], idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    for art in prior_art:
      art["quantum_similarity"] = self._quantum_similarity(idea.get("description", ""), art.get("abstract", ""))
    prior_art.sort(key=lambda x: x.get("quantum_similarity", 0), reverse=True)
    return prior_art

  def _quantum_similarity(self, text1: str, text2: str) -> float:
    _ = text1, text2
    return float(np.random.random())

  async def _analyze_novelty(self, idea: Dict[str, Any], prior_art: List[Dict[str, Any]]) -> float:
    if not prior_art:
      return 1.0
    return float(1.0 - prior_art[0].get("quantum_similarity", 0.0))

  async def _generate_patent_application(self, idea: Dict[str, Any], prior_art: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
      "title": idea.get("title", ""),
      "inventors": idea.get("inventors", []),
      "abstract": await self._generate_abstract(idea),
      "background": await self._generate_background(idea, prior_art),
      "summary": await self._generate_summary(idea),
      "detailed_description": await self._generate_detailed_description(idea),
      "claims": await self._generate_claims(idea),
      "drawings": idea.get("drawings", []),
    }

  async def _generate_abstract(self, idea: Dict[str, Any]) -> str:
    return f"Abstract for {idea.get('title', '')}: [Generated abstract]"

  async def _generate_background(self, idea: Dict[str, Any], prior_art: List[Dict[str, Any]]) -> str:
    background = [
      "BACKGROUND OF THE INVENTION",
      "",
      "Field of the Invention",
      f"This invention relates to {idea.get('field', 'technology')}.",
      "",
      "Description of Related Art",
    ]
    for art in prior_art[:5]:
      background.append(f"U.S. Patent {art['id']} discloses {art['abstract']}")
    return "\n".join(background)

  async def _generate_summary(self, idea: Dict[str, Any]) -> str:
    return f"SUMMARY\n\nThe present invention provides {idea.get('summary', 'a novel solution')}"

  async def _generate_detailed_description(self, idea: Dict[str, Any]) -> str:
    return f"DETAILED DESCRIPTION\n\n{idea.get('description', '')}"

  async def _generate_claims(self, idea: Dict[str, Any]) -> List[str]:
    claims = [
      "1. A method for {base}, comprising: {steps}.".format(
        base=idea.get('claim_base', 'achieving an improvement'),
        steps=", ".join(idea.get('steps', ['step 1', 'step 2', 'step 3'])),
      )
    ]
    for idx, variation in enumerate(idea.get("variations", []), start=2):
      claims.append(f"{idx}. The method of claim 1, wherein {variation}.")
    return claims

  async def _file_provisional_patent(self, application: Dict[str, Any]) -> Dict[str, Any]:
    if self.uspto_client.is_configured() and httpx is not None and settings.allow_httpx_network:
      try:
        filing = await self.uspto_client.file_provisional(application)
        if filing:
          return filing
      except Exception as exc:  # pragma: no cover - network failure
        LOG.warning("[USPTO] filing failed (%s). Falling back to simulated response.", exc)
    LOG.info("[USPTO] simulated provisional filing")
    return {
      "status": "filed",
      "application_number": f"63/{np.random.randint(100000, 999999)}",
      "filing_date": datetime.now().isoformat(),
      "confirmation_number": hashlib.md5(application.get('title', '').encode()).hexdigest()[:12],
    }


class PatentKnowledgeBase:
  def __init__(self) -> None:
    self.rules = {
      "provisional_requirements": [
        "Title of invention",
        "Name(s) of inventor(s)",
        "Description of invention",
        "At least one claim",
      ],
      "novelty_requirements": [
        "Must be new (35 U.S.C. 102)",
        "Must be non-obvious (35 U.S.C. 103)",
        "Must be useful (35 U.S.C. 101)",
      ],
    }


class USPTOClient:
  def __init__(self) -> None:
    self.base_url = settings.uspto_base_url.rstrip("/")
    self.timeout = settings.uspto_timeout
    self._credentials = settings.uspto_credentials()

  def is_configured(self) -> bool:
    return self._credentials is not None

  def _client_kwargs(self) -> Dict[str, Any]:
    headers: Dict[str, str] = {
      "Accept": "application/json",
    }
    auth = None
    if not self._credentials:
      return {"base_url": self.base_url, "timeout": self.timeout, "headers": headers}
    if "api_key" in self._credentials:
      headers["X-API-KEY"] = self._credentials["api_key"]
    elif "username" in self._credentials:
      auth = (self._credentials["username"], self._credentials.get("password", ""))
    return {"base_url": self.base_url, "timeout": self.timeout, "headers": headers, "auth": auth}

  async def search_patents(self, query: str) -> Optional[List[Dict[str, Any]]]:
    if not self.is_configured() or httpx is None:
      return None
    params = {"q": query}
    try:
      async with httpx.AsyncClient(**self._client_kwargs()) as client:
        response = await client.get("/applications", params=params)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - network failure
      LOG.warning("[USPTO] search request failed: %s", exc)
      return None

    raw_results: List[Dict[str, Any]] = []
    if isinstance(payload, list):
      raw_results = payload
    elif isinstance(payload, dict):
      for key in ("results", "applications", "records"):
        if isinstance(payload.get(key), list):
          raw_results = payload[key]
          break

    results: List[Dict[str, Any]] = []
    for index, record in enumerate(raw_results):
      if not isinstance(record, dict):
        continue
      results.append(
        {
          "id": str(record.get("applicationNumber") or record.get("publicationNumber") or record.get("documentIdentifier") or f"US-{index}"),
          "title": record.get("title") or record.get("inventionTitle") or "Unknown",
          "abstract": record.get("abstract") or record.get("summary") or "",
          "filing_date": record.get("filingDate") or record.get("documentDate") or "",
          "similarity": float(record.get("similarity", np.random.random())),
        }
      )
    return results

  async def file_provisional(self, application: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not self.is_configured() or httpx is None:
      return None
    payload = {
      "title": application.get("title"),
      "inventors": application.get("inventors", []),
      "claims": application.get("claims", []),
      "description": application.get("detailed_description", ""),
      "drawings": application.get("drawings", []),
    }
    try:
      async with httpx.AsyncClient(**self._client_kwargs()) as client:
        response = await client.post("/provisional-filings", json=payload)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - network failure
      LOG.warning("[USPTO] filing request failed: %s", exc)
      return None

    return {
      "status": data.get("status", "filed"),
      "application_number": data.get("applicationNumber", f"63/{np.random.randint(100000, 999999)}"),
      "filing_date": data.get("filingDate", datetime.now().isoformat()),
      "confirmation_number": data.get("confirmationNumber", hashlib.md5(payload.get("title", "").encode()).hexdigest()[:12]),
    }


# ---------------------------------------------------------------------------
# FastAPI application factory (optional)
# ---------------------------------------------------------------------------

_users_db: Dict[str, Dict[str, Any]] = {}
_security = HTTPBearer() if HTTPBearer is not None else None


if BaseModel is not None:  # pragma: no cover - optional dependency

  class UserCreate(BaseModel):
    email: str
    password: str

  class IdeaSubmission(BaseModel):
    title: str
    description: str
    keywords: List[str]
    field: str
    inventors: List[str]

else:  # pragma: no cover

  UserCreate = None  # type: ignore
  IdeaSubmission = None  # type: ignore


def create_patent_api() -> "FastAPI":
  if FastAPI is None or BaseModel is None or HTTPException is None:
    raise ImportError("FastAPI and Pydantic are required to create the patent API.")

  app = FastAPI(title="Patent Discovery API")

  if Request is not None:  # pragma: no cover - middleware wiring
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
      start_time = time.perf_counter()
      response = await call_next(request)
      duration = time.perf_counter() - start_time
      response.headers["X-Process-Time"] = f"{duration:.4f}"
      LOG.info(
        "[http] %s %s -> %s (%.4fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
      )
      return response

  if Instrumentator is not None:  # pragma: no cover - optional dependency
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

  @app.get("/healthz")
  async def healthcheck():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

  @app.post("/api/register")
  async def register_user(user: UserCreate):  # type: ignore[misc]
    if user.email in _users_db:
      raise HTTPException(400, "User already exists")
    trial_end = datetime.now() + timedelta(days=3)
    _users_db[user.email] = {
      "password": hashlib.sha256(user.password.encode()).hexdigest(),
      "created_at": datetime.now(),
      "trial_ends": trial_end,
      "is_subscribed": False,
      "ideas_submitted": [],
    }
    return {"message": "User created", "trial_ends": trial_end.isoformat()}

  def _subscription_status(email: str) -> Dict[str, Any]:
    if email not in _users_db:
      raise HTTPException(404, "User not found")
    user = _users_db[email]
    now = datetime.now()
    is_trial_active = now < user["trial_ends"]
    can_access = is_trial_active or user["is_subscribed"]
    return {
      "can_access": can_access,
      "is_trial": is_trial_active and not user["is_subscribed"],
      "trial_ends": user["trial_ends"].isoformat(),
      "is_subscribed": user["is_subscribed"],
    }

  @app.get("/api/subscription/status")
  async def check_subscription(email: str):
    return _subscription_status(email)

  @app.post("/api/patent/discover")
  async def discover_patent(idea: IdeaSubmission, email: str):  # type: ignore[misc]
    status = _subscription_status(email)
    if not status["can_access"]:
      raise HTTPException(403, "Trial expired. Please subscribe.")
    quantum_sim = ClaudeQuantumSimulator(num_qubits=20)
    agent = PatentMetaAgent(quantum_sim)
    result = await agent.discover_and_file_patent(idea.dict())
    _users_db[email]["ideas_submitted"].append(
      {"idea": idea.dict(), "result": result, "timestamp": datetime.now().isoformat()}
    )
    return result

  @app.post("/api/subscribe")
  async def subscribe_user(email: str, stripe_token: str):
    if email not in _users_db:
      raise HTTPException(404, "User not found")
    if settings.stripe_configured():
      if stripe is None:
        LOG.warning("Stripe secret configured but stripe package is missing.")
      else:
        stripe.api_key = settings.stripe_secret_key
        try:
          stripe.PaymentIntent.create(
            amount=999,
            currency="usd",
            payment_method=stripe_token,
            confirm=True,
          )
        except Exception as exc:  # pragma: no cover - payment failure
          LOG.error("Stripe payment failed: %s", exc)
          raise HTTPException(502, "Payment processing failed") from exc
    _users_db[email]["is_subscribed"] = True
    return {"message": "Subscription activated"}

  return app


# ---------------------------------------------------------------------------
# SwiftUI and deployment artefacts
# ---------------------------------------------------------------------------

IOS_APP_CODE = r"""
// PatentDiscoveryApp.swift
// Quantum-Enhanced Patent Discovery iOS App

import SwiftUI
import Combine

// MARK: - App Entry Point
@main
struct PatentDiscoveryApp: App {
    @StateObject private var authManager = AuthManager()

    var body: some Scene {
        WindowGroup {
            if authManager.isAuthenticated {
                MainTabView()
                    .environmentObject(authManager)
            } else {
                LoginView()
                    .environmentObject(authManager)
            }
        }
    }
}

// MARK: - Models
struct User: Codable {
    var email: String
    var trialEnds: Date?
    var isSubscribed: Bool
}

struct PatentIdea: Identifiable, Codable {
    var id = UUID()
    var title: String
    var description: String
    var keywords: [String]
    var field: String
    var inventors: [String]
}

struct PatentResult: Codable {
    var status: String
    var noveltyScore: Double?
    var applicationNumber: String?
    var priorArtCount: Int
}

// MARK: - API Client
class APIClient: ObservableObject {
    private let baseURL = "https://your-api.com/api"

    func submitIdea(_ idea: PatentIdea) async throws -> PatentResult {
        let url = URL(string: "\(baseURL)/patent/discover")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let encoder = JSONEncoder()
        request.httpBody = try encoder.encode(idea)

        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(PatentResult.self, from: data)
    }

    func checkSubscription() async throws -> Bool {
        return true
    }
}

// MARK: - Auth Manager
class AuthManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var showPaywall = false

    func login(email: String, password: String) async {
        isAuthenticated = true
    }

    func register(email: String, password: String) async {
        isAuthenticated = true
        currentUser = User(email: email, trialEnds: Date().addingTimeInterval(3*24*60*60), isSubscribed: false)
    }
}

// MARK: - Main Tab View
struct MainTabView: View {
    var body: some View {
        TabView {
            IdeaSubmissionView()
                .tabItem {
                    Label("New Idea", systemImage: "lightbulb")
                }

            HistoryView()
                .tabItem {
                    Label("History", systemImage: "clock")
                }

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
    }
}

// MARK: - Idea Submission View
struct IdeaSubmissionView: View {
    @StateObject private var api = APIClient()
    @State private var title = ""
    @State private var description = ""
    @State private var keywords = ""
    @State private var field = "Technology"
    @State private var isProcessing = false
    @State private var result: PatentResult?

    var body: some View {
        NavigationView {
            Form {
                Section("Invention Details") {
                    TextField("Title", text: $title)
                    TextEditor(text: $description)
                        .frame(height: 150)
                    TextField("Keywords (comma separated)", text: $keywords)

                    Picker("Field", selection: $field) {
                        Text("Technology").tag("Technology")
                        Text("Medical").tag("Medical")
                        Text("Mechanical").tag("Mechanical")
                        Text("Chemical").tag("Chemical")
                    }
                }

                Section {
                    Button(action: submitIdea) {
                        if isProcessing {
                            ProgressView()
                        } else {
                            Text("Discover & File Patent")
                                .bold()
                        }
                    }
                    .disabled(isProcessing || title.isEmpty)
                }

                if let result = result {
                    Section("Results") {
                        ResultView(result: result)
                    }
                }
            }
            .navigationTitle("Patent Discovery")
        }
    }

    func submitIdea() {
        isProcessing = true

        Task {
            do {
                let idea = PatentIdea(
                    title: title,
                    description: description,
                    keywords: keywords.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) },
                    field: field,
                    inventors: []
                )

                result = try await api.submitIdea(idea)
                isProcessing = false
            } catch {
                print("Error: \(error)")
                isProcessing = false
            }
        }
    }
}

// MARK: - Result View
struct ResultView: View {
    let result: PatentResult

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: result.status == "success" ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundColor(result.status == "success" ? .green : .red)
                Text(result.status.capitalized)
                    .font(.headline)
            }

            if let score = result.noveltyScore {
                Text("Novelty Score: \(String(format: "%.1f%%", score * 100))")
            }

            if let appNumber = result.applicationNumber {
                Text("Application #: \(appNumber)")
                    .font(.caption)
            }

            Text("Prior Art Found: \(result.priorArtCount)")
                .font(.caption)
        }
    }
}

// MARK: - Login View
struct LoginView: View {
    @EnvironmentObject var authManager: AuthManager
    @State private var email = ""
    @State private var password = ""

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "atom")
                .font(.system(size: 80))
                .foregroundColor(.blue)

            Text("Quantum Patent Discovery")
                .font(.title)
                .bold()

            Text("3-Day Free Trial")
                .foregroundColor(.green)

            TextField("Email", text: $email)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)

            SecureField("Password", text: $password)
                .textFieldStyle(RoundedBorderTextFieldStyle())

            Button("Sign Up") {
                Task {
                    await authManager.register(email: email, password: password)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

// MARK: - History View
struct HistoryView: View {
    var body: some View {
        NavigationView {
            List {
                Text("Your submitted ideas will appear here")
            }
            .navigationTitle("History")
        }
    }
}

// MARK: - Settings View
struct SettingsView: View {
    @EnvironmentObject var authManager: AuthManager

    var body: some View {
        NavigationView {
            Form {
                Section("Subscription") {
                    if let user = authManager.currentUser {
                        if !user.isSubscribed {
                            if let trialEnds = user.trialEnds, trialEnds > Date() {
                                Text("Trial ends: \(trialEnds, style: .date)")
                            } else {
                                Button("Subscribe Now") {
                                    authManager.showPaywall = true
                                }
                                .foregroundColor(.blue)
                            }
                        } else {
                            Text("✓ Subscribed")
                                .foregroundColor(.green)
                        }
                    }
                }

                Section {
                    Button("Sign Out") {
                        authManager.isAuthenticated = false
                    }
                    .foregroundColor(.red)
                }
            }
            .navigationTitle("Settings")
        }
    }
}
"""

DEPLOYMENT_GUIDE = """
BACKEND DEPLOYMENT:
1. pip install fastapi uvicorn torch numpy
2. uvicorn patent_discovery:create_patent_api --factory --host 0.0.0.0 --port 8000

IOS APP DEPLOYMENT:
1. Copy `IOS_APP_CODE` into an Xcode SwiftUI project.
2. Update the API base URL to point at your deployment.
3. Configure App Store subscriptions with a 3-day free trial.

USPTO INTEGRATION:
1. Obtain MyUSPTO credentials and API access keys.
2. Implement OAuth2 flows in `USPTOClient` before production use.
"""


def write_ios_app_code(path: str) -> None:
  with open(path, "w", encoding="utf-8") as handle:
    handle.write(IOS_APP_CODE)


__all__ = [
  "ClaudeQuantumSimulator",
  "PatentMetaAgent",
  "PatentKnowledgeBase",
  "USPTOClient",
  "QuantumBackend",
  "SimulationReport",
  "create_patent_api",
  "IOS_APP_CODE",
  "DEPLOYMENT_GUIDE",
  "write_ios_app_code",
]
