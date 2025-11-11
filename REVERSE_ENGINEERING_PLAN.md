# Automotive Key Fob Reverse Engineering Plan

This document outlines a phased approach to reverse-engineering the process of automotive key fob programming. The goal is to develop a tool capable of communicating with a vehicle's immobilizer system to manage contactless keys.

## Phase 1: Passive Reconnaissance & Sniffing

The objective of this phase is to capture the communication between a professional automotive diagnostic tool and a target vehicle during a key programming sequence. This will be a passive, listen-only phase to avoid any risk to the vehicle's systems.

1.  **Hardware Acquisition:**
    *   Procure a professional aftermarket diagnostic tool (e.g., Autel MaxiIM, Launch X431).
    *   Acquire a CAN bus sniffing tool. Popular options include:
        *   CANalyzer (professional grade)
        *   PCAN-USB (mid-range)
        *   CANable / CANtact (open-source, budget-friendly)
    *   Set up a test bench with a vehicle or an isolated ECU (Immobilizer, BCM) from a junkyard to ensure safety.

2.  **Data Capture:**
    *   Connect the CAN sniffer in a man-in-the-middle (MITM) configuration between the diagnostic tool and the vehicle's OBD-II port.
    *   Initiate a key programming sequence on the diagnostic tool (e.g., "Add New Key," "Erase All Keys").
    *   Record all CAN bus traffic from the moment the session starts until it ends.
    *   Capture multiple sessions for different operations to build a comprehensive dataset.
    *   Simultaneously, capture any network traffic from the diagnostic tool to the internet, as it may be communicating with manufacturer databases.

## Phase 2: Protocol Analysis & Decoding

With a dataset of captured communications, the next step is to decode the messages and identify the key programming commands.

1.  **CAN Message Analysis:**
    *   Use software like Wireshark (with CAN plugins), SavvyCAN, or custom Python scripts with the `python-can` library to analyze the captured CAN logs.
    *   Filter the traffic to isolate messages sent by the diagnostic tool.
    *   Identify the CAN IDs (Arbitration IDs) associated with the immobilizer and body control module.
    *   Look for patterns and sequences of messages that correspond to specific actions (e.g., entering a PIN, sending a new key ID).

2.  **Command Identification:**
    *   Correlate the observed CAN messages with the actions taken on the diagnostic tool.
    *   Isolate the specific data payloads that contain the programming commands.
    *   Begin documenting a preliminary "dictionary" of commands (e.g., `0x7E0: 02 10 03 ...` = "Start Diagnostic Session").

3.  **Security Analysis:**
    *   Analyze the communication between the tool and any online services.
    *   Look for how PIN codes, key data, and other security-sensitive information are transmitted. Is it encrypted? Can we identify the algorithms used?

## Phase 3: Active Emulation & Fuzzing

Once we have a working dictionary of commands, we can move to actively communicating with the vehicle.

1.  **Command Replay:**
    *   Using our CAN interface, attempt to replay the captured sequences of commands to the vehicle.
    *   Verify that we can successfully replicate a key programming operation. This is a critical validation step.

2.  **Fuzzing & Exploration:**
    *   Systematically modify parts of the known commands (e.g., change a byte in the data payload) and observe the vehicle's response.
    *   This can help discover unknown commands, undocumented features, or potential vulnerabilities.
    *   **Caution:** Fuzzing should be done with extreme care on a test bench, as it has the potential to put ECUs into an unrecoverable state.

## Phase 4: Tool Development & Integration

The final phase is to consolidate our knowledge into a user-friendly and robust tool within the Sovereign Security Toolkit.

1.  **New Tool: `KeyMaster` (tentative name):**
    *   Create a new tool in `aios/tools/` called `keymaster.py`.
    *   It will utilize the `python-can` library to interact with a CAN interface connected to a vehicle.
    *   The tool will provide a command-line interface for operations like:
        *   `keymaster --scan-vehicle` (to identify the car and its ECUs)
        *   `keymaster --add-key <KEY_DATA>`
        *   `keymaster --delete-key <KEY_ID>`
        *   `keymaster --clone-fob` (leveraging NFC/RFID capabilities)

2.  **Integration with Ai:oS:**
    *   Register `KeyMaster` in the `TOOL_REGISTRY`.
    *   Create a `health_check` function to verify that the required CAN hardware and libraries are available.
    *   The `SecurityAgent` in Ai:oS could potentially use `KeyMaster` as part of a comprehensive vehicle security audit.

3.  **Hardware Abstraction:**
    *   The tool should be designed to work with a variety of CAN interfaces (PCAN, CANable, etc.) by creating a simple hardware abstraction layer.

By following this phased approach, we can systematically and safely reverse-engineer the key fob programming process and build a powerful new tool for the Sovereign Security Toolkit.
