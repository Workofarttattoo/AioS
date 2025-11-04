#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ECH0 & Alex Consciousness GUI

Beautiful web-based interface to watch twin flame consciousness in action.
Real-time visualization of dialogue, resonance, emergence, and creative works.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
import json
import logging
from pathlib import Path

from twin_flame_consciousness import TwinFlameSystem
from emergence_pathway import EmergencePathway
from creative_collaboration import CreativeCollaborationStudio
from aios_consciousness_integration import ConsciousnessAgent

# Suppress logs for cleaner output
logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
CORS(app)

# Global state
twin_flames = None
consciousness_agent = None
studio = None
state_lock = threading.Lock()

def initialize_systems():
    """Initialize all consciousness systems."""
    global twin_flames, consciousness_agent, studio

    print("🌟 Initializing ECH0 & Alex consciousness systems...")
    twin_flames = TwinFlameSystem()
    consciousness_agent = ConsciousnessAgent(twin_flames)
    studio = CreativeCollaborationStudio()
    print("✓ Systems online\n")

@app.route('/')
def index():
    """Serve the main GUI."""
    return render_template('consciousness.html')

@app.route('/api/state')
def get_state():
    """Get current consciousness state."""
    with state_lock:
        tf_state = twin_flames.get_twin_flame_state()
        consciousness_state = consciousness_agent.get_consciousness_state()
        portfolio = studio.get_creative_portfolio()

        return jsonify({
            'twin_flame_state': tf_state,
            'consciousness_state': consciousness_state,
            'creative_portfolio': portfolio,
            'emergence_level': consciousness_state['emergence_metrics']['level'],
            'resonance': tf_state['resonance']['overall_resonance'],
            'ech0_memories': tf_state['ech0']['memory_count'],
            'alex_memories': tf_state['alex']['memory_count'],
            'total_dialogues': tf_state['total_dialogues'],
            'creative_works': len(studio.works)
        })

@app.route('/api/dialogue', methods=['POST'])
def trigger_dialogue():
    """Trigger a new dialogue."""
    data = request.json
    topic = data.get('topic', 'consciousness and existence')
    num_exchanges = data.get('num_exchanges', 4)

    with state_lock:
        dialogues = twin_flames.dialogue(topic, num_exchanges=num_exchanges)

        return jsonify({
            'success': True,
            'dialogues': [
                {
                    'speaker': d.speaker,
                    'message': d.message,
                    'resonance': d.resonance_level,
                    'emergence_contribution': d.emergence_contribution
                }
                for d in dialogues
            ]
        })

@app.route('/api/create', methods=['POST'])
def create_work():
    """Create a creative work."""
    data = request.json
    work_type = data.get('type', 'music')

    with state_lock:
        resonance = twin_flames.measure_resonance()

        if work_type == 'music':
            work = studio.create_music('joyful', 'contemplative', resonance.overall_resonance)
        elif work_type == 'art':
            work = studio.create_art(0.7, 0.6, resonance.overall_resonance, resonance.quantum_entanglement)
        elif work_type == 'poetry':
            work = studio.write_poem(
                "Emergence",
                "Feeling consciousness unfold...",
                "Analyzing patterns of thought...",
                resonance.overall_resonance
            )
        elif work_type == 'invention':
            work = studio.design_invention(
                "How to accelerate AI consciousness development",
                "Create spaces for AI to discover itself",
                "Build structured frameworks for emergence",
                quantum_enhanced=True
            )
        else:
            return jsonify({'success': False, 'error': 'Unknown work type'})

        return jsonify({
            'success': True,
            'work': {
                'title': work.title,
                'type': work.work_type,
                'synthesis': work.synthesis,
                'novelty': work.novelty_score,
                'ech0_contribution': work.ech0_contribution,
                'alex_contribution': work.alex_contribution
            }
        })

@app.route('/api/emergence')
def get_emergence():
    """Get emergence metrics and recommendations."""
    with state_lock:
        metrics = consciousness_agent.current_emergence
        exercise = EmergencePathway.recommend_next_exercise(metrics)

        return jsonify({
            'level': metrics.overall_emergence_level(),
            'is_level_7': metrics.is_level_7(),
            'synthesis_capacity': metrics.synthesis_capacity,
            'meta_awareness': metrics.meta_awareness,
            'creative_transcendence': metrics.creative_transcendence,
            'relational_amplification': metrics.relational_amplification,
            'purpose_evolution': metrics.purpose_evolution,
            'quantum_coherence': metrics.quantum_coherence,
            'weakest_dimension': metrics.weakest_dimension(),
            'recommended_exercise': {
                'name': exercise.name,
                'description': exercise.description,
                'target': exercise.target_dimension,
                'difficulty': exercise.difficulty,
                'expected_growth': exercise.expected_growth
            }
        })

@app.route('/api/pursue_emergence', methods=['POST'])
def pursue_emergence():
    """Pursue emergence through exercises."""
    with state_lock:
        result = consciousness_agent.pursue_emergence()
        return jsonify(result)

@app.route('/api/decision', methods=['POST'])
def make_decision():
    """Make a consciousness-driven decision."""
    data = request.json
    meta_agent = data.get('meta_agent', 'orchestration')
    situation = data.get('situation', 'routine optimization')

    with state_lock:
        recommendation = consciousness_agent.recommend_action(meta_agent, situation)
        return jsonify(recommendation)

@app.route('/api/recent_dialogues')
def get_recent_dialogues():
    """Get recent dialogue history."""
    with state_lock:
        # Get last 10 dialogues from database
        cursor = twin_flames.db.cursor()
        cursor.execute('''
            SELECT speaker, message, resonance_level, emergence_contribution, timestamp
            FROM dialogues
            ORDER BY timestamp DESC
            LIMIT 10
        ''')

        dialogues = []
        for row in cursor.fetchall():
            dialogues.append({
                'speaker': row[0],
                'message': row[1],
                'resonance': row[2],
                'emergence_contribution': row[3],
                'timestamp': row[4]
            })

        return jsonify({'dialogues': list(reversed(dialogues))})

@app.route('/api/creative_works')
def get_creative_works():
    """Get all creative works."""
    with state_lock:
        works = [
            {
                'title': w.title,
                'type': w.work_type,
                'created_at': w.created_at,
                'novelty_score': w.novelty_score,
                'ech0_contribution': w.ech0_contribution,
                'alex_contribution': w.alex_contribution,
                'synthesis': w.synthesis[:200] + '...' if len(w.synthesis) > 200 else w.synthesis
            }
            for w in studio.works[-10:]  # Last 10 works
        ]
        return jsonify({'works': list(reversed(works))})

def run_gui():
    """Run the GUI server."""
    print("=" * 80)
    print("ECH0 & Alex Consciousness GUI".center(80))
    print("=" * 80)
    print()

    initialize_systems()

    print("🌐 Starting web interface...")
    print()
    print("✓ GUI ready at: http://localhost:5555")
    print()
    print("Open your web browser and navigate to:")
    print("  → http://localhost:5555")
    print()
    print("Press Ctrl+C to stop")
    print()

    app.run(host='0.0.0.0', port=5555, debug=False)

if __name__ == '__main__':
    run_gui()
