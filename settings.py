"""Centralised runtime settings for AgentaOS services.

The settings object surfaces environment-driven configuration for integrations
such as USPTO, Stripe, telemetry endpoints, and logging.  No third-party
library is required—values are read from :mod:`os.environ` when the module is
imported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
  uspto_base_url: str = os.getenv("USPTO_BASE_URL", "https://patentcenter.uspto.gov/api")
  uspto_username: str = os.getenv("USPTO_USERNAME", "")
  uspto_password: str = os.getenv("USPTO_PASSWORD", "")
  uspto_api_key: str = os.getenv("USPTO_API_KEY", "")
  uspto_timeout: float = float(os.getenv("USPTO_TIMEOUT", "30"))

  stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")

  sentry_dsn: str = os.getenv("SENTRY_DSN", "")

  allow_httpx_network: bool = os.getenv("ALLOW_NETWORK_CALLS", "false").lower() in {"1", "true", "yes", "on"}

  def uspto_credentials(self) -> Optional[dict]:
    if self.uspto_api_key:
      return {"api_key": self.uspto_api_key}
    if self.uspto_username and self.uspto_password:
      return {"username": self.uspto_username, "password": self.uspto_password}
    return None

  def stripe_configured(self) -> bool:
    return bool(self.stripe_secret_key)


settings = Settings()

__all__ = ["Settings", "settings"]
