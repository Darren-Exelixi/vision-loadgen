"""The provisioned Grafana dashboard stays in step with the metrics the exporter declares."""

import json
import re
from pathlib import Path

import pytest

from vision_loadgen.exporter import METRICS

MONITORING = Path(__file__).resolve().parent.parent / "monitoring"
DASHBOARD = MONITORING / "grafana" / "dashboards" / "vision-loadgen.json"
DATASOURCE_UID = "loadgen-prometheus"


def _dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _panels(dashboard: dict) -> list[dict]:
    panels = []
    for panel in dashboard["panels"]:
        panels.append(panel)
        panels.extend(panel.get("panels", []))
    return panels


def _expressions(dashboard: dict) -> list[str]:
    expressions = [target["expr"] for panel in _panels(dashboard) for target in panel.get("targets", [])]
    expressions += [item["expr"] for item in dashboard["annotations"]["list"] if "expr" in item]
    for variable in dashboard["templating"]["list"]:
        query = variable.get("query")
        expressions.append(query["query"] if isinstance(query, dict) else query)
    return expressions


def test_every_metric_in_the_dashboard_is_exported():
    used = {name for expr in _expressions(_dashboard()) for name in re.findall(r"\bloadgen_[a-z0-9_]+", expr)}
    assert used, "the dashboard queries no loadgen metrics"
    assert used - set(METRICS) == set()


def test_panels_use_the_provisioned_datasource():
    dashboard = _dashboard()
    for panel in _panels(dashboard):
        if panel["type"] == "row":
            continue
        assert panel["datasource"]["uid"] == DATASOURCE_UID, panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == DATASOURCE_UID, panel["title"]


def test_dashboard_identity_and_unique_panel_ids():
    dashboard = _dashboard()
    assert dashboard["uid"] == "vision-loadgen"
    ids = [panel["id"] for panel in _panels(dashboard)]
    assert len(ids) == len(set(ids))
    assert {variable["name"] for variable in dashboard["templating"]["list"]} == {"run_id", "worker"}


def test_provisioning_files_point_at_each_other():
    yaml = pytest.importorskip("yaml")
    datasources = yaml.safe_load((MONITORING / "grafana/provisioning/datasources/prometheus.yml").read_text())
    assert [source["uid"] for source in datasources["datasources"]] == [DATASOURCE_UID]
    providers = yaml.safe_load((MONITORING / "grafana/provisioning/dashboards/loadgen.yml").read_text())
    assert providers["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"
    compose = yaml.safe_load((MONITORING / "docker-compose.yml").read_text())
    assert "./grafana/dashboards:/var/lib/grafana/dashboards:ro" in compose["services"]["grafana"]["volumes"]
    prometheus = yaml.safe_load((MONITORING / "prometheus/prometheus.yml").read_text())
    assert prometheus["scrape_configs"][0]["file_sd_configs"][0]["files"] == ["/etc/prometheus/targets/*.json"]
    targets = json.loads((MONITORING / "prometheus/targets/loadgen.json").read_text())
    assert targets[0]["targets"] == ["host.docker.internal:9464"]
