"""The language switch must preserve the selected scientific result."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from migration_workbench.gui import MainWindow
from migration_workbench.runner import _report_html


class LocalizationTest(unittest.TestCase):
    def test_language_switch_and_report_preserve_values(self):
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        previous_setting = window.settings.value("language", "zh")
        try:
            window.case_combo.setCurrentIndex(1)
            window._open_selected_case()
            case_id = window.case_combo.currentData()
            energy = window.current["readout"]["Eperc_a"]
            window.language_combo.setCurrentIndex(1)
            self.assertEqual(window.tabs.tabText(0), "Overview & sources")
            self.assertEqual(window.case_combo.currentData(), case_id)
            self.assertEqual(window.current["readout"]["Eperc_a"], energy)
            self.assertIn("nodes", window.metrics_label.text())
            report = _report_html(window.current, "en")
            self.assertIn('html lang="en"', report)
            self.assertIn("Migration Network Report", report)
            self.assertNotIn("个节点", report)
            window.language_combo.setCurrentIndex(0)
            self.assertEqual(window.tabs.tabText(0), "概览与来源")
            self.assertEqual(window.current["readout"]["Eperc_a"], energy)
        finally:
            window.settings.setValue("language", previous_setting)
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
