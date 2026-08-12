from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.validate_boulder_anchor_discovery_packet import PACKET, validate


class BoulderAnchorDiscoveryPacketTests(unittest.TestCase):
    def test_checked_in_packet_is_complete_and_fail_closed(self):
        self.assertEqual(validate(), [])

    def test_opened_safety_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp) / "packet"
            shutil.copytree(PACKET, packet)
            summary_path = packet / "coverage_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["elo_update_allowed"] = True
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            errors = validate(packet)
            self.assertTrue(any("safety gate" in error for error in errors))
            self.assertTrue(any("hash mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

