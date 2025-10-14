"""
Prompt routing utilities for AgentaOS.

The router converts natural language intent into manifest action paths using a
combination of keyword rules and lightweight similarity scoring.  Additional
classifiers can be plugged in to integrate external ML services.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from operator import attrgetter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import Manifest


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class IntentMatch:
    action_path: str
    score: float
    sources: List[Tuple[str, float]]


class IntentClassifier:
    """Abstract interface for intent classifiers."""

    name: str = "classifier"

    def classify(self, text: str) -> List[IntentMatch]:  # pragma: no cover - interface
        raise NotImplementedError


class KeywordClassifier(IntentClassifier):
    """Rule-based classifier using curated keywords and manifest metadata."""

    name = "keyword"

    def __init__(self, manifest: Manifest):
        self.manifest = manifest
        self.keyword_map = self._build_keyword_map(manifest)

    def classify(self, text: str) -> List[IntentMatch]:
        normalized = text.strip().lower()
        matches: Dict[str, IntentMatch] = {}
        for keyword, action_path in self.keyword_map.items():
            if keyword in normalized:
                match = matches.setdefault(
                    action_path,
                    IntentMatch(action_path=action_path, score=0.0, sources=[]),
                )
                match.score = max(match.score, 1.0)
                match.sources.append((self.name, 1.0))

        if matches:
            return list(matches.values())

        # Fallback: attempt to match meta-agent names or action identifiers.
        tokens = _tokenize(normalized)
        fallback_matches: List[IntentMatch] = []
        for token in tokens:
            if token in self.manifest.meta_agents:
                for path in self.manifest.boot_sequence:
                    if path.startswith(f"{token}."):
                        fallback_matches.append(
                            IntentMatch(action_path=path, score=0.5, sources=[(self.name, 0.5)])
                        )
                        break
            else:
                for path in self.manifest.boot_sequence:
                    _, action = path.split(".", maxsplit=1)
                    if token == action:
                        fallback_matches.append(
                            IntentMatch(action_path=path, score=0.4, sources=[(self.name, 0.4)])
                        )
                        break

        return fallback_matches

    @staticmethod
    def _build_keyword_map(manifest: Manifest) -> Dict[str, str]:
        keywords: Dict[str, str] = {}
        semantic_hints: List[Tuple[str, str]] = [
            ("start kernel", "kernel.process_management"),
            ("load drivers", "kernel.device_drivers"),
            ("check memory", "kernel.memory_management"),
            ("start gui", "gui.window_management"),
            ("launch ui", "gui.gui_design"),
            ("enable firewall", "security.firewall"),
            ("encrypt disk", "security.encryption"),
            ("scan threats", "security.threat_detection"),
            ("configure network", "networking.network_configuration"),
            ("ping test", "networking.data_transmission"),
            ("resolve dns", "networking.dns_resolver"),
            ("mount storage", "storage.file_system"),
            ("verify recovery", "storage.recovery"),
            ("list apps", "application.application_launcher"),
            ("sync packages", "application.package_manager"),
            ("resolve dependencies", "application.dependency_resolver"),
            ("list sessions", "user.session_manager"),
            ("load preferences", "user.preferences"),
            ("monitor load", "scalability.monitor_load"),
            ("scale up", "scalability.scale_up"),
            ("balance traffic", "scalability.load_balancing"),
            ("scale down", "scalability.scale_down"),
            ("emit telemetry", "orchestration.telemetry"),
            ("enforce policy", "orchestration.policy_engine"),
        ]

        for keyword, action in semantic_hints:
            if action in manifest.boot_sequence or action in manifest.shutdown_sequence:
                keywords[keyword] = action

        for meta_name, meta_config in manifest.meta_agents.items():
            keywords[f"{meta_name}"] = f"{meta_name}.{meta_config.actions[0].key}"
            for action in meta_config.actions:
                human_token = action.key.replace("_", " ")
                keywords[human_token] = f"{meta_name}.{action.key}"

        return keywords


class SimilarityClassifier(IntentClassifier):
    """
    Lightweight bag-of-words similarity classifier.

    Embeds each action path using manifest metadata and scores prompts with
    cosine similarity.  This provides ML-style behaviour without external
    dependencies.
    """

    name = "similarity"

    def __init__(self, manifest: Manifest):
        self.manifest = manifest
        self.action_vectors = self._build_vectors(manifest)

    def classify(self, text: str) -> List[IntentMatch]:
        tokens = _tokenize(text)
        if not tokens:
            return []
        query_counts = Counter(tokens)
        query_norm = _vector_norm(query_counts)

        matches: List[IntentMatch] = []
        for action_path, vector in self.action_vectors.items():
            dot = _dot_product(query_counts, vector)
            if dot <= 0:
                continue
            score = dot / (query_norm * _vector_norm(vector))
            if score <= 0:
                continue
            matches.append(
                IntentMatch(
                    action_path=action_path,
                    score=float(score),
                    sources=[(self.name, float(score))],
                )
            )

        matches.sort(key=attrgetter("score"), reverse=True)
        return matches[:5]

    @staticmethod
    def _build_vectors(manifest: Manifest) -> Dict[str, Counter]:
        vectors: Dict[str, Counter] = {}
        for meta_name, meta_config in manifest.meta_agents.items():
            for action in meta_config.actions:
                text_bits = [
                    meta_name,
                    action.key,
                    meta_config.description,
                    action.description,
                ]
                tokens = _tokenize(" ".join(filter(None, text_bits)))
                vectors[f"{meta_name}.{action.key}"] = Counter(tokens)
        return vectors


class EnsembleIntentClassifier(IntentClassifier):
    """Aggregates multiple classifiers and merges their scores."""

    name = "ensemble"

    def __init__(self, classifiers: Sequence[IntentClassifier]):
        self.classifiers = classifiers

    def classify(self, text: str) -> List[IntentMatch]:
        combined: Dict[str, IntentMatch] = {}
        for classifier in self.classifiers:
            matches = classifier.classify(text)
            for match in matches:
                existing = combined.get(match.action_path)
                if existing:
                    existing.score = max(existing.score, match.score)
                    existing.sources.extend(match.sources)
                else:
                    combined[match.action_path] = IntentMatch(
                        action_path=match.action_path,
                        score=match.score,
                        sources=list(match.sources),
                    )

        ranked = sorted(combined.values(), key=attrgetter("score"), reverse=True)
        return ranked


class PromptRouter:
    """Intent router that can leverage multiple classifiers."""

    def __init__(self, manifest: Manifest, classifiers: Optional[Sequence[IntentClassifier]] = None):
        self.manifest = manifest
        if classifiers is None:
            classifiers = (
                KeywordClassifier(manifest),
                SimilarityClassifier(manifest),
            )
        self.classifier = EnsembleIntentClassifier(classifiers)

    def route(self, text: str, top_k: int = 5) -> List[IntentMatch]:
        matches = self.classifier.classify(text)
        if top_k > 0:
            matches = matches[:top_k]
        return matches

    def explain(self, action_path: str) -> str:
        meta, action = action_path.split(".", maxsplit=1)
        try:
            config = self.manifest.action_config(action_path)
            description = config.description or action.replace("_", " ")
        except KeyError:
            description = action.replace("_", " ")
        return f"{meta}.{action} → {description}"


def _tokenize(text: str) -> List[str]:
    return [match.group(0) for match in TOKEN_PATTERN.finditer(text.lower()) if len(match.group(0)) > 1]


def _vector_norm(counter: Counter) -> float:
    return math.sqrt(sum(value * value for value in counter.values())) or 1.0


def _dot_product(a: Counter, b: Counter) -> float:
    return float(sum(a[token] * b[token] for token in a.keys() & b.keys()))
