import logging
import importlib


def test_mythickey_verbose_demo_emits_debug(caplog):
  mythickey = importlib.import_module("tools.mythickey")
  with caplog.at_level(logging.DEBUG):
    rc = mythickey.main(["--demo", "--verbose"])
  assert rc == 0
  messages = [rec.getMessage() for rec in caplog.records]
  assert any("Parsed args" in m or "Loaded" in m or "Evaluated" in m for m in messages)

