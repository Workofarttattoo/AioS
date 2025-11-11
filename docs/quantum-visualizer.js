// Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

// ============================================================
// QUANTUM CIRCUIT STATE
// ============================================================

let numQubits = 3;
let selectedGate = null;
let gateAngle = Math.PI / 2;
let numShots = 1000;
let circuit = [];
let stateVector = null;

// ============================================================
// INITIALIZATION
// ============================================================

function initCircuit() {
    const dim = Math.pow(2, numQubits);
    stateVector = new Array(dim).fill(0).map((_, i) => ({
        real: i === 0 ? 1 : 0,
        imag: 0
    }));
    renderCircuit();
    updateStateDisplay();
}

function renderCircuit() {
    const canvas = document.getElementById('circuitCanvas');
    canvas.innerHTML = '';

    for (let q = 0; q < numQubits; q++) {
        const line = document.createElement('div');
        line.className = 'qubit-line';
        line.innerHTML = `
            <div class="qubit-label">|q${q}⟩</div>
            <div class="qubit-wire" data-qubit="${q}"></div>
            <div class="qubit-label">→</div>
        `;
        canvas.appendChild(line);

        const wire = line.querySelector('.qubit-wire');
        wire.addEventListener('click', (e) => {
            if (selectedGate) {
                addGateToCircuit(q, selectedGate, e);
            }
        });
    }

    circuit.forEach((gate, idx) => {
        renderGateOnWire(gate, idx);
    });
}

function addGateToCircuit(qubit, gateType, event) {
    const wire = event.target;
    const rect = wire.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const position = clickX / rect.width;

    const gate = {
        type: gateType,
        qubit: qubit,
        position: position,
        angle: gateType.startsWith('R') ? gateAngle : null
    };

    circuit.push(gate);
    circuit.sort((a, b) => a.position - b.position);
    renderCircuit();
}

function renderGateOnWire(gate, idx) {
    const wires = document.querySelectorAll('.qubit-wire');
    const wire = wires[gate.qubit];
    if (!wire) return;

    const gateElem = document.createElement('div');
    gateElem.className = 'gate-on-wire';
    gateElem.textContent = gate.type;
    gateElem.style.left = (gate.position * 100) + '%';
    gateElem.title = `${gate.type} on q${gate.qubit}`;
    gateElem.onclick = (e) => {
        e.stopPropagation();
        circuit.splice(idx, 1);
        renderCircuit();
    };

    wire.appendChild(gateElem);
}

// ============================================================
// GATE SELECTION
// ============================================================

function selectGate(gateName) {
    document.querySelectorAll('.gate-btn').forEach(b => b.classList.remove('selected'));
    document.querySelector(`.gate-btn[data-gate="${gateName}"]`)?.classList.add('selected');
    selectedGate = gateName;
}

// ============================================================
// QUANTUM SIMULATION
// ============================================================

function runCircuit() {
    resetCircuit();

    circuit.forEach(gate => {
        applyGate(gate);
    });

    updateStateDisplay();
}

function applyGate(gate) {
    switch(gate.type) {
        case 'H':
            applyHadamard(gate.qubit);
            break;
        case 'X':
            applyPauliX(gate.qubit);
            break;
        case 'Y':
            applyPauliY(gate.qubit);
            break;
        case 'Z':
            applyPauliZ(gate.qubit);
            break;
        case 'RX':
            applyRotationX(gate.qubit, gate.angle);
            break;
        case 'RY':
            applyRotationY(gate.qubit, gate.angle);
            break;
        case 'RZ':
            applyRotationZ(gate.qubit, gate.angle);
            break;
    }
}

function applyHadamard(qubit) {
    const dim = Math.pow(2, numQubits);
    const newState = new Array(dim).fill(0).map(() => ({real: 0, imag: 0}));
    const factor = 1 / Math.sqrt(2);

    for (let i = 0; i < dim; i++) {
        const bit = (i >> qubit) & 1;
        const flipped = i ^ (1 << qubit);

        if (bit === 0) {
            newState[i].real += factor * stateVector[i].real;
            newState[i].imag += factor * stateVector[i].imag;
            newState[flipped].real += factor * stateVector[i].real;
            newState[flipped].imag += factor * stateVector[i].imag;
        } else {
            newState[i].real += factor * stateVector[i].real;
            newState[i].imag += factor * stateVector[i].imag;
            newState[flipped].real -= factor * stateVector[i].real;
            newState[flipped].imag -= factor * stateVector[i].imag;
        }
    }

    stateVector = newState;
}

function applyPauliX(qubit) {
    const dim = Math.pow(2, numQubits);
    const newState = new Array(dim).fill(0).map(() => ({real: 0, imag: 0}));

    for (let i = 0; i < dim; i++) {
        const flipped = i ^ (1 << qubit);
        newState[flipped] = stateVector[i];
    }

    stateVector = newState;
}

function applyPauliY(qubit) {
    const dim = Math.pow(2, numQubits);
    const newState = new Array(dim).fill(0).map(() => ({real: 0, imag: 0}));

    for (let i = 0; i < dim; i++) {
        const bit = (i >> qubit) & 1;
        const flipped = i ^ (1 << qubit);

        if (bit === 0) {
            newState[flipped].real = -stateVector[i].imag;
            newState[flipped].imag = stateVector[i].real;
        } else {
            newState[flipped].real = stateVector[i].imag;
            newState[flipped].imag = -stateVector[i].real;
        }
    }

    stateVector = newState;
}

function applyPauliZ(qubit) {
    const dim = Math.pow(2, numQubits);

    for (let i = 0; i < dim; i++) {
        const bit = (i >> qubit) & 1;
        if (bit === 1) {
            stateVector[i].real *= -1;
            stateVector[i].imag *= -1;
        }
    }
}

function applyRotationX(qubit, angle) {
    const dim = Math.pow(2, numQubits);
    const newState = new Array(dim).fill(0).map(() => ({real: 0, imag: 0}));
    const cos = Math.cos(angle / 2);
    const sin = Math.sin(angle / 2);

    for (let i = 0; i < dim; i++) {
        const flipped = i ^ (1 << qubit);

        newState[i].real += cos * stateVector[i].real + sin * stateVector[flipped].imag;
        newState[i].imag += cos * stateVector[i].imag - sin * stateVector[flipped].real;

        newState[flipped].real += cos * stateVector[flipped].real + sin * stateVector[i].imag;
        newState[flipped].imag += cos * stateVector[flipped].imag - sin * stateVector[i].real;
    }

    stateVector = newState;
}

function applyRotationY(qubit, angle) {
    const dim = Math.pow(2, numQubits);
    const newState = new Array(dim).fill(0).map(() => ({real: 0, imag: 0}));
    const cos = Math.cos(angle / 2);
    const sin = Math.sin(angle / 2);

    for (let i = 0; i < dim; i++) {
        const flipped = i ^ (1 << qubit);

        newState[i].real += cos * stateVector[i].real - sin * stateVector[flipped].real;
        newState[i].imag += cos * stateVector[i].imag - sin * stateVector[flipped].imag;

        newState[flipped].real += sin * stateVector[i].real + cos * stateVector[flipped].real;
        newState[flipped].imag += sin * stateVector[i].imag + cos * stateVector[flipped].imag;
    }

    stateVector = newState;
}

function applyRotationZ(qubit, angle) {
    const dim = Math.pow(2, numQubits);

    for (let i = 0; i < dim; i++) {
        const bit = (i >> qubit) & 1;
        const phase = bit === 0 ? -angle/2 : angle/2;
        const cos = Math.cos(phase);
        const sin = Math.sin(phase);

        const real = stateVector[i].real * cos - stateVector[i].imag * sin;
        const imag = stateVector[i].real * sin + stateVector[i].imag * cos;
        stateVector[i].real = real;
        stateVector[i].imag = imag;
    }
}

// ============================================================
// DISPLAY UPDATES
// ============================================================

function updateStateDisplay() {
    const container = document.getElementById('stateVector');
    container.innerHTML = '';

    stateVector.forEach((amp, idx) => {
        const prob = amp.real * amp.real + amp.imag * amp.imag;
        if (prob > 0.001) {
            const basis = idx.toString(2).padStart(numQubits, '0');
            const ampStr = `${amp.real.toFixed(3)} ${amp.imag >= 0 ? '+' : ''}${amp.imag.toFixed(3)}i`;
            const probStr = (prob * 100).toFixed(1) + '%';

            const item = document.createElement('div');
            item.className = 'state-item';
            item.innerHTML = `
                <span class="state-basis">|${basis}⟩</span>
                <span class="state-amplitude">${ampStr}</span>
                <span class="state-prob">${probStr}</span>
            `;
            container.appendChild(item);
        }
    });

    if (container.children.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: var(--secondary);">No significant amplitudes</div>';
    }
}

function measureCircuit() {
    runCircuit();

    const dim = Math.pow(2, numQubits);
    const counts = {};

    for (let shot = 0; shot < numShots; shot++) {
        const rand = Math.random();
        let cumProb = 0;

        for (let i = 0; i < dim; i++) {
            const prob = stateVector[i].real * stateVector[i].real +
                         stateVector[i].imag * stateVector[i].imag;
            cumProb += prob;

            if (rand <= cumProb) {
                const basis = i.toString(2).padStart(numQubits, '0');
                counts[basis] = (counts[basis] || 0) + 1;
                break;
            }
        }
    }

    displayMeasurementResults(counts);
}

function displayMeasurementResults(counts) {
    const container = document.getElementById('measurementResults');
    container.innerHTML = '';

    const sortedResults = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

    sortedResults.forEach(([basis, count]) => {
        const percentage = (count / numShots * 100).toFixed(1);

        const bar = document.createElement('div');
        bar.className = 'result-bar';
        bar.innerHTML = `
            <div class="result-label">|${basis}⟩: ${count} shots (${percentage}%)</div>
            <div class="result-progress">
                <div class="result-fill" style="width: ${percentage}%">${percentage}%</div>
            </div>
        `;
        container.appendChild(bar);
    });
}

// ============================================================
// CONTROL FUNCTIONS
// ============================================================

function addQubit() {
    if (numQubits < 5) {
        numQubits++;
        initCircuit();
    }
}

function removeQubit() {
    if (numQubits > 1) {
        numQubits--;
        circuit = circuit.filter(g => g.qubit < numQubits);
        initCircuit();
    }
}

function clearCircuit() {
    circuit = [];
    renderCircuit();
    resetCircuit();
}

function resetCircuit() {
    initCircuit();
}

function updateAngle(value) {
    gateAngle = (value / 100) * Math.PI;
    const piValue = (gateAngle / Math.PI).toFixed(2);
    document.getElementById('angleValue').textContent = `${piValue}π`;
}

function updateShots(value) {
    numShots = parseInt(value);
    document.getElementById('shotsValue').textContent = numShots;
}

// ============================================================
// INITIALIZE
// ============================================================

window.addEventListener('DOMContentLoaded', () => {
    initCircuit();
});
