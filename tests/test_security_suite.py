import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from aios.config import ActionConfig, DEFAULT_MANIFEST, Manifest, MetaAgentConfig
from aios.runtime import AgentaRuntime
from aios.virtualization import VirtualizationLayer


class SecuritySuiteTests(unittest.TestCase):
  def _security_manifest(self) -> Manifest:
    security_agent = MetaAgentConfig(
      name="security",
      description="Security automation for testing.",
      actions=[
        ActionConfig("access_control", "Activate RBAC policies."),
        ActionConfig("sovereign_suite", "Assess Sovereign toolkit readiness.", critical=False),
      ],
    )
    return Manifest(
      meta_agents={"security": security_agent},
      boot_sequence=["security.access_control"],
      shutdown_sequence=[],
    )

  def test_sovereign_suite_reports_health(self) -> None:
    runtime = AgentaRuntime(DEFAULT_MANIFEST)
    runtime.context.environment["AGENTA_SECURITY_TOOLS"] = "AuroraScan,CipherSpear"

    result = runtime.superuser.execute_action_path("security.sovereign_suite", runtime.context)
    self.assertTrue(result.success, result.message)

    payload = runtime.context.metadata.get("security.sovereign_suite")
    self.assertIsInstance(payload, dict)
    self.assertEqual(sorted(payload.get("requested", [])), ["AuroraScan", "CipherSpear"])
    reports = payload.get("reports", [])
    self.assertEqual(len(reports), 2)
    self.assertFalse(payload.get("missing"))
    self.assertFalse(payload.get("degraded"))
    binaries = payload.get("binaries", {})
    cipher_meta = binaries.get("CipherSpear", {})
    self.assertTrue(cipher_meta.get("exists"))
    if os.name != "nt":
      self.assertTrue(cipher_meta.get("executable", True))

  def test_sovereign_suite_flags_unknown_tool(self) -> None:
    runtime = AgentaRuntime(DEFAULT_MANIFEST)
    runtime.context.environment["AGENTA_SECURITY_TOOLS"] = "UnknownTool"

    result = runtime.superuser.execute_action_path("security.sovereign_suite", runtime.context)
    self.assertTrue(result.success, result.message)

    payload = runtime.context.metadata.get("security.sovereign_suite")
    self.assertIsInstance(payload, dict)
    self.assertEqual(payload.get("requested"), ["UnknownTool"])
    missing = payload.get("missing", [])
    self.assertIn("UnknownTool", missing)
    self.assertFalse(payload.get("degraded"))

  def test_auto_security_suite_runs_after_boot(self) -> None:
    manifest = self._security_manifest()
    runtime = AgentaRuntime(manifest)
    runtime.context.environment["AGENTA_SECURITY_TOOLS"] = "AuroraScan"

    runtime.boot()

    payload = runtime.context.metadata.get("security.sovereign_suite")
    self.assertIsInstance(payload, dict)
    self.assertIn("AuroraScan", payload.get("requested", []))
    self.assertEqual(payload.get("suite"), "Sovereign Security Toolkit")

  def test_auto_security_suite_skips_without_tools(self) -> None:
    manifest = self._security_manifest()
    runtime = AgentaRuntime(manifest)

    runtime.boot()

    self.assertNotIn("security.sovereign_suite", runtime.context.metadata)


class VirtualizationPlanTests(unittest.TestCase):
  def _sample_env(self) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    qemu_stub = str(repo_root / ".bin" / "qemu-system-x86_64")
    return {
      "AGENTA_QEMU_BINARY": qemu_stub,
      "AGENTA_QEMU_IMAGE": str(repo_root / ".bin" / "mock.qcow2"),
      "AGENTA_QEMU_EXECUTE": "1",
      "AGENTA_QEMU_PASSTHROUGH_DEVICES": "",
      "AGENTA_QEMU_NET_DEVICE": "virtio-net-pci",
      "AGENTA_QEMU_MANAGED_SHUTDOWN": "1",
    }

  def test_managed_shutdown_uses_qmp(self) -> None:
    env = self._sample_env()
    env["AGENTA_QEMU_QMP"] = "tcp://127.0.0.1:5555"

    layer = VirtualizationLayer(env)
    plan = layer._plan_qemu("down")
    self.assertIsNotNone(plan)
    self.assertTrue(plan.execute)
    self.assertEqual(plan.command[0], sys.executable)
    self.assertIn("system_powerdown", " ".join(plan.command[2:]))
    metadata = plan.metadata
    self.assertEqual(metadata.get("managed_shutdown_strategy"), "tcp")
    self.assertTrue(metadata.get("managed_shutdown_enabled"))

  def test_plans_include_bridge_prerequisites_and_passthrough(self) -> None:
    with tempfile.NamedTemporaryFile() as passthrough_device:
      env = self._sample_env()
      env["AGENTA_QEMU_NET_BRIDGE"] = "agenta0"
      env["AGENTA_QEMU_TAP"] = "agenta-tap0"
      env["AGENTA_QEMU_NETDEV_ID"] = "agenta"
      env["AGENTA_QEMU_PASSTHROUGH_DEVICES"] = f"path:{passthrough_device.name}"

      layer = VirtualizationLayer(env)
      plans = list(layer._plans_for("up"))

      summaries = [plan.summary for plan in plans if plan.provider == "qemu"]
      self.assertTrue(any("interface 'agenta0'" in summary or "agenta-tap0" in summary for summary in summaries))

      verify_steps = [
        plan for plan in plans
        if plan.provider == "qemu"
        and plan.metadata.get("role") in {"bridge", "tap"}
        and not plan.metadata.get("setup_step")
      ]
      self.assertTrue(verify_steps)
      for plan in verify_steps:
        metadata = plan.metadata
        self.assertIn("setup_commands", metadata)
        self.assertFalse(metadata.get("auto_setup", False))

      launch_plan = next(plan for plan in plans if plan.provider == "qemu" and plan.execute)
      self.assertIn("virtio-net-pci", " ".join(launch_plan.command or []))
      metadata = launch_plan.metadata
      self.assertTrue(metadata.get("netdev_attached"))
      self.assertEqual(metadata.get("passthrough_attached"), 1)

  def test_network_autocreate_generates_setup_steps(self) -> None:
    env = self._sample_env()
    env["AGENTA_QEMU_NET_BRIDGE"] = "agenta0"
    env["AGENTA_QEMU_NET_AUTOCREATE"] = "1"

    original_which = shutil.which

    def fake_which(name: str):
      if name == "ip":
        return "/usr/bin/ip"
      return original_which(name)

    with mock.patch("AgentaOS.virtualization.shutil.which", side_effect=fake_which):
      layer = VirtualizationLayer(env)
      plans = list(layer._plan_qemu_prerequisites("up"))

    setup_plans = [plan for plan in plans if plan.metadata.get("setup_step")]
    self.assertTrue(setup_plans)
    self.assertTrue(all(plan.execute for plan in setup_plans))
    total_steps = setup_plans[0].metadata.get("setup_steps")
    self.assertEqual(total_steps, len(setup_plans))
    for plan in setup_plans:
      self.assertEqual(plan.metadata.get("setup_steps"), total_steps)
      self.assertIn("/usr/bin/ip", plan.command[0])

  def test_network_autocreate_generates_teardown_steps(self) -> None:
    env = self._sample_env()
    env["AGENTA_QEMU_NET_BRIDGE"] = "agenta0"
    env["AGENTA_QEMU_TAP"] = "agenta-tap0"
    env["AGENTA_QEMU_NET_AUTOCREATE"] = "1"

    original_which = shutil.which

    def fake_which(name: str):
      if name == "ip":
        return "/usr/bin/ip"
      return original_which(name)

    with mock.patch("AgentaOS.virtualization.shutil.which", side_effect=fake_which):
      layer = VirtualizationLayer(env)
      self.assertTrue(layer._autocreate_network())
      self.assertTrue(layer._network_teardown_commands("tap", "agenta-tap0"))
      plans = list(layer._plan_qemu_prerequisites("down"))

    teardown_plans = [plan for plan in plans if plan.metadata.get("teardown_step")]
    self.assertTrue(teardown_plans)
    self.assertTrue(all(plan.execute for plan in teardown_plans))
    commands = [" ".join(plan.command) for plan in teardown_plans]
    self.assertIn("/usr/bin/ip link set agenta-tap0 down", commands)
    self.assertIn("/usr/bin/ip tuntap del dev agenta-tap0 mode tap", commands)
    self.assertIn("/usr/bin/ip link del agenta0", commands)
    total_steps = teardown_plans[0].metadata.get("teardown_steps")
    self.assertEqual(total_steps, len(teardown_plans))
    self.assertTrue(all(plan.metadata.get("auto_teardown") for plan in teardown_plans))


class ToolkitCLITests(unittest.TestCase):
  def test_cli_wrappers_expose_help(self) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wrappers = [
      "cipherspear",
      "skybreaker",
      "mythickey",
      "spectratrace",
      "nemesishydra",
      "obsidianhunt",
      "vectorflux",
    ]
    for wrapper in wrappers:
      script = repo_root / wrapper
      self.assertTrue(script.exists(), f"Wrapper {wrapper} is missing")
      proc = subprocess.run([str(script), "--help"], capture_output=True, text=True)
      self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
  unittest.main()
