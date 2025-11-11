import logging
import importlib


def test_aurorascan_verbose_lists_profiles_emits_debug(caplog):
  aurorascan = importlib.import_module("tools.aurorascan")
  with caplog.at_level(logging.DEBUG):
    rc = aurorascan.main(["--list-profiles", "--verbose"])
  assert rc == 0
  messages = [rec.getMessage() for rec in caplog.records]
  assert any("Listing available profiles" in m or "Parsed args" in m for m in messages)

