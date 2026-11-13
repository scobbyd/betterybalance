import pathlib
import re

CC = pathlib.Path(__file__).resolve().parent.parent / "custom_components" / "betterybalance"
ALLOWED = {
    "api.energy-charts.info", "publications.tennet.eu", "dataportal-api.nordpoolgroup.com",
    "services.tenergy.nl", "api.tennet.eu", "api.open-meteo.com",
    "historical-forecast-api.open-meteo.com", "archive-api.open-meteo.com",
    "previous-runs-api.open-meteo.com", "github.com", "open-meteo.com",
}


def test_the_integration_names_only_the_allowed_hosts():
    hosts = {
        h
        for p in CC.rglob("*")
        if p.suffix in (".py", ".js", ".json")
        for h in re.findall(r"https?://([^/\"'\s?]+)", p.read_text())
    }
    assert hosts <= ALLOWED, hosts - ALLOWED
