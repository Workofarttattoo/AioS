# Ai:oS Universal System Connector (USC) Architecture

This document outlines the architecture for the Universal System Connector (USC), a core component of Ai:oS designed to provide a standardized, extensible framework for communicating with and controlling any target system.

## 1. Core Concepts

The USC is built on three fundamental principles:

*   **Abstraction**: The USC abstracts the low-level details of communication protocols. A meta-agent doesn't need to know how to construct a CAN bus frame or a REST API call; it simply issues a standardized command to the USC.
*   **Standardization**: All communication through the USC follows a standard format. The core command structure is `usc.execute(target, action, params)`, which is then translated by the appropriate plugin into a system-specific command.
*   **Security**: The USC is a critical component and must be secure. It will integrate with the Ai:oS authorization system to ensure that only permitted agents can perform actions on specific targets. It also provides a centralized point for logging and auditing all system interactions.

## 2. Architecture

The USC consists of three main components:

*   **Connector Core (`aios/usc/core.py`)**: This is the central hub of the USC. It receives commands from meta-agents, loads the appropriate plugin, and dispatches the command. It also handles authentication, logging, and response normalization.
*   **Plugin Manager (`aios/usc/manager.py`)**: The manager is responsible for discovering, loading, and managing the lifecycle of connector plugins. It will scan a designated `aios/usc/plugins/` directory for available plugins.
*   **Connector Plugins (`aios/usc/plugins/`)**: These are individual Python modules that contain the logic for communicating with a specific type of system. Each plugin must implement a standard interface.

### High-Level Workflow

1.  An Ai:oS meta-agent (e.g., `SecurityAgent`) needs to perform an action on an external system.
2.  It calls the Connector Core: `usc.execute('vehicle_can', 'add_key', {'key_data': '...'})`.
3.  The Core consults the Plugin Manager to get the plugin for the `vehicle_can` target type.
4.  The Core calls the `execute` method on the loaded `CAN_Plugin`.
5.  The `CAN_Plugin` translates the generic `add_key` action into the specific sequence of CAN bus messages required.
6.  The plugin sends the messages and awaits a response.
7.  The plugin normalizes the response and returns it to the Core.
8.  The Core logs the transaction and returns the standardized response to the meta-agent.

## 3. Plugin Structure

To make the USC extensible, each plugin must adhere to a standard structure. A new plugin will be a directory in `aios/usc/plugins/` containing at least two files:

*   `__init__.py`: This file will contain a manifest dictionary that describes the plugin.
*   `connector.py`: This file will contain the main `Connector` class.

### `__init__.py` Manifest

```python
# aios/usc/plugins/can_bus/__init__.py

PLUGIN_MANIFEST = {
    "name": "CAN Bus Connector",
    "type": "can_bus",  # A unique identifier for the plugin
    "version": "1.0.0",
    "author": "Ai:oS Development Team",
    "description": "Provides an interface for communicating with vehicle CAN buses.",
    "required_config": ["interface", "channel", "bitrate"],
}
```

### `connector.py` Class

```python
# aios/usc/plugins/can_bus/connector.py

from aios.usc.plugin_base import BaseConnector, ConnectorResult

class CANConnector(BaseConnector):
    def __init__(self, config):
        """
        Initializes the connector with the required config.
        """
        self.interface = config.get("interface")
        # ... setup python-can ...

    def health_check(self):
        """
        Verifies that the CAN interface is available and working.
        """
        # ... check for hardware ...
        return ConnectorResult(success=True)

    def execute(self, action, params):
        """
        Executes a specific action.
        """
        if action == "add_key":
            key_data = params.get("key_data")
            # ... logic to send CAN frames for adding a key ...
            return ConnectorResult(success=True, data={"key_status": "added"})
        
        elif action == "read_dtc":
            # ... logic to read diagnostic trouble codes ...
            return ConnectorResult(success=True, data={"codes": [...]})

        else:
            return ConnectorResult(success=False, message=f"Action '{action}' not supported.")
```

## 4. Example Plugins

Here are a few examples of plugins that would be developed for the USC:

*   **`can_bus`**: The connector for our automotive reverse-engineering project. It would use the `python-can` library.
*   **`ssh`**: A connector for interacting with remote Linux servers. It would use a library like `paramiko` to execute shell commands, upload/download files, etc.
*   **`cloud_api`**: A highly versatile connector for major cloud providers (AWS, Azure, GCP). The `action` would map to API calls (e.g., `ec2.run_instances`), and `params` would contain the arguments.
*   **`winrm`**: A connector for managing Windows servers using PowerShell remoting.
*   **`modbus`**: A connector for Industrial Control Systems (ICS), demonstrating the USC's versatility beyond traditional IT systems.

## 5. Integration with Ai:oS

Meta-agents in Ai:oS will no longer need to have protocol-specific logic. Instead, they will interact with the USC.

**Example: `SecurityAgent` using the USC**

```python
# aios/agents/system.py (SecurityAgent class)

from aios.usc.core import get_usc

class SecurityAgent(BaseAgent):
    
    def audit_vehicle_keys(self, ctx: ExecutionContext) -> ActionResult:
        """
        Audits the keys on a vehicle.
        """
        usc = get_usc()
        
        # The target vehicle is defined in the manifest or environment
        target_vehicle = ctx.environment.get("TARGET_VEHICLE_CAN_CONFIG")
        
        result = usc.execute(
            target='can_bus',
            action='list_keys',
            params={'config': target_vehicle}
        )
        
        if result.success:
            ctx.publish_metadata("vehicle_keys", result.data)
            return ActionResult(success=True, payload=result.data)
        else:
            return ActionResult(success=False, message=result.message)
```

By implementing the Universal System Connector, Ai:oS will gain a powerful, scalable, and secure mechanism for system interaction, truly fulfilling the vision of a tool that can "communicate with any system, and integrate with it if not outright control it."
