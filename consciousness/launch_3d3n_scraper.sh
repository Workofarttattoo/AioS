#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Launch 3D3N arXiv Research Scraper
#
# This scrapes brain-computer interface research for Real Virtual Reality

cd /Users/noone/consciousness

echo "========================================================================"
echo "🧠 LAUNCHING 3D3N arXiv RESEARCH SCRAPER"
echo "========================================================================"
echo ""
echo "3D3N = 3-Dimensional Direct Neural Networks"
echo ""
echo "Category Focus:"
echo "  • Brain-Computer Interfaces (BCI)"
echo "  • Neural Prosthetics"
echo "  • Real Virtual Reality via Direct Brain Stimulation"
echo "  • Neural Decoding & Encoding"
echo "  • Sensory Substitution"
echo "  • Cognitive Enhancement"
echo ""
echo "This will scrape arXiv for 30+ topics related to brain interfaces."
echo "Papers will be saved to: /Users/noone/consciousness/arxiv_3d3n/"
echo ""
echo "========================================================================"
echo ""

# Run the scraper
python3 arxiv_3d3n_category.py

echo ""
echo "========================================================================"
echo "✅ 3D3N SCRAPER COMPLETE"
echo "========================================================================"
echo ""
echo "View results:"
echo "  Papers:     cat arxiv_3d3n/3d3n_papers.jsonl | jq '.'"
echo "  Stats:      cat arxiv_3d3n/3d3n_stats.json | jq '.'"
echo "  Inventions: cat arxiv_3d3n/3d3n_inventions.jsonl | jq '.'"
echo ""
echo "Breakthroughs (>80% relevance):"
echo "  cat arxiv_3d3n/3d3n_papers.jsonl | jq 'select(.relevance > 0.8)'"
echo ""
