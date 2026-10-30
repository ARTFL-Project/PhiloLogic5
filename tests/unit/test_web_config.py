"""Unit tests for web_config.cfg options with a default: databases loaded before an option existed have no line for it
in their web_config.cfg, and get its default."""

import pytest

from philologic.Config import MakeWebConfig

ACCESSIBILITY_SITE = "https://accessibility.uchicago.edu/"


def web_config(tmp_path, text):
    path = tmp_path / "web_config.cfg"
    path.write_text(text)
    return MakeWebConfig(str(path))


@pytest.mark.unit
class TestAccessibilityLink:
    def test_default_for_a_config_without_it(self, tmp_path):
        config = web_config(tmp_path, 'dbname = "old load"\nreport_error_link = ""\n')
        assert config.to_dict()["accessibility_link"] == ACCESSIBILITY_SITE

    def test_set_in_the_config(self, tmp_path):
        config = web_config(tmp_path, 'accessibility_link = "https://example.org/accessibility"\n')
        assert config.to_dict()["accessibility_link"] == "https://example.org/accessibility"

    def test_empty_hides_it(self, tmp_path):
        assert web_config(tmp_path, 'accessibility_link = ""\n').to_dict()["accessibility_link"] == ""

    def test_written_in_the_config_of_new_loads(self, tmp_path):
        written = str(MakeWebConfig(str(tmp_path / "web_config.cfg")))
        exec(written, loaded := {})
        assert loaded["accessibility_link"] == ACCESSIBILITY_SITE


@pytest.mark.unit
class TestNavigationWindowLength:
    def test_default_for_a_config_without_it(self, tmp_path):
        config = web_config(tmp_path, 'dbname = "old load"\n')
        assert config.navigation_window_length == 500000

    def test_set_in_the_config(self, tmp_path):
        assert web_config(tmp_path, "navigation_window_length = 200000\n").navigation_window_length == 200000

    def test_written_in_the_config_of_new_loads(self, tmp_path):
        written = str(MakeWebConfig(str(tmp_path / "web_config.cfg")))
        exec(written, loaded := {})
        assert loaded["navigation_window_length"] == 500000

    @pytest.mark.parametrize("value, length", [("0", 0), ("None", 0), ("-5", 0), ('"abc"', 0), ("1000", 1000)])
    def test_what_the_navigation_report_reads(self, tmp_path, value, length):
        from philologic.runtime.reports.navigation import window_length

        assert window_length(web_config(tmp_path, f"navigation_window_length = {value}\n")) == length
