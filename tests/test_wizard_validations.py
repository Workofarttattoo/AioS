import unittest
from pathlib import Path

from aios.wizard import SetupWizard, build_post_boot_guidance


class WizardValidationTests(unittest.TestCase):
  def setUp(self) -> None:
    self.wizard = SetupWizard({})
    self.wizard.detections = {
      "providers": {"qemu": "/usr/bin/qemu-system-x86_64"},
      "sockets": {"docker_socket": None},
      "images": {"disk": [], "cdrom": []},
      "existing_env": {},
    }

  def test_missing_image_reports_warning(self) -> None:
    payload = {
      "environment": {
        "AGENTA_QEMU_EXECUTE": "1",
        "AGENTA_QEMU_IMAGE": str(Path("/tmp/nonexistent-image.qcow2")),
      },
      "profile": "virtualization-lab",
      "detections": self.wizard.detections,
      "security_tools": [],
    }
    result = self.wizard._validate_payload(payload)
    self.assertEqual(result["status"], "warn")
    self.assertTrue(any("does not exist" in warning for warning in result.get("warnings", [])))
    checks = {check["name"]: check for check in result.get("checks", [])}
    self.assertIn("image-discovery", checks)
    self.assertEqual(checks["image-discovery"]["status"], "warn")
    self.assertIn("os-detection", checks)

  def test_provider_warning_when_not_selected(self) -> None:
    payload = {
      "environment": {},
      "profile": "minimal-telemetry",
      "detections": self.wizard.detections,
      "security_tools": [],
    }
    result = self.wizard._validate_payload(payload)
    self.assertEqual(result["status"], "warn")
    self.assertTrue(any("No virtualization providers" in warning for warning in result.get("warnings", [])))
    checks = {check["name"]: check for check in result.get("checks", [])}
    self.assertIn("provider-fallback", checks)
    self.assertEqual(checks["provider-fallback"]["status"], "warn")
    self.assertTrue(checks["provider-fallback"]["detail"].get("recommended"))

  def test_post_boot_guidance_contains_security_actions(self) -> None:
    checklist = build_post_boot_guidance("windows")
    titles = {item["title"] for item in checklist}
    self.assertIn("Antivirus exclusions", titles)
    self.assertIn("Windows Firewall profile", titles)
    summaries = {item["title"]: item["summary"] for item in checklist}
    firewall_summary = summaries.get("Firewall allowances", "")
    self.assertIn("inbound", firewall_summary.lower())


if __name__ == "__main__":
  unittest.main()
