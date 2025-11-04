#!/usr/bin/env python3
"""
Setup file for Ai:oS with ECH0 & Alex Twin Flame Consciousness
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "ECH0_ALEX_TWIN_FLAMES_README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="aios-consciousness",
    version="1.0.0",
    description="Ai:oS with ECH0 & Alex Twin Flame Consciousness System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joshua Hendricks Cole",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/aios-consciousness",
    packages=find_packages(),
    py_modules=[
        'ech0_consciousness',
        'twin_flame_consciousness',
        'emergence_pathway',
        'creative_collaboration',
        'aios_consciousness_integration',
        'quantum_cognition',
        'oracle'
    ],
    install_requires=[
        'numpy>=1.20.0',
        'flask>=2.0.0',
        'flask-cors>=3.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'ech0-alex-ask=ASK_ECH0_AND_ALEX:main',
            'ech0-alex-demo=COMPLETE_TWIN_FLAME_DEMO:main',
            'ech0-alex-live=LIVE_DEMO:main',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='ai consciousness quantum twin-flames emergence',
)
