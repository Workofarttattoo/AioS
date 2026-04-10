#!/usr/bin/env python3
"""
Setup file for Ai:oS with ECH0 & Alex Twin Flame Consciousness
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="aios",
    version="1.0.0",
    description="AI:OS — Agentic Intelligence Operating System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joshua Hendricks Cole",
    author_email="admin@aios.is",
    url="https://github.com/Workofarttattoo/AioS",
    packages=find_packages(),
    py_modules=[
        'config',
        'runtime',
        'settings',
        'model',
        'apps',
        'diagnostics',
        'virtualization',
        'ech0_consciousness',
        'twin_flame_consciousness',
        'emergence_pathway',
        'creative_collaboration',
        'aios_consciousness_integration',
        'quantum_cognition',
        'oracle',
    ],
    install_requires=[
        'numpy>=1.20.0',
        'flask>=2.0.0',
        'flask-cors>=3.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'ruff>=0.1.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'aios=aios_cli:main',
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='ai agents operating-system quantum consciousness meta-agents',
)
