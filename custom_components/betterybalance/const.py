"""Constants shared by the integration modules."""

DOMAIN = "betterybalance"
PLATFORMS = ("sensor", "binary_sensor", "select")
FEEDS = ("day_ahead", "ida", "settlement", "live", "irradiance", "pack")
SENSOR_KEYS = (
    "state",
    "plan",
    "result_today",
    "result",
    "result_provisional",
    "shadow_soc",
    "take_nowcast",
    "feed_nowcast",
    "pv_expected",
    "backtest",
)
BINARY_KEYS = tuple(f"feed_{f}" for f in FEEDS)
SERVICES = (
    "run_backtest",
    "export_ledger",
    "export_results",
    "refit_site_models",
    "verify_ledger",
)
STAT_RESULT = "betterybalance:result"
WS_LEDGER_TAIL = "betterybalance/ledger_tail"
URL_LEDGER = "/api/betterybalance/{entry_id}/ledger.zip"
CARD_URL = "/betterybalance_static/betterybalance-plan-card.js"
STORE_DIRNAME = "betterybalance"
STORE_FILE = "store.sqlite3"
PACK_DIRNAME = "pack"  # the installed data packs, under STORE_DIRNAME
ISSUE_URL = "https://github.com/scobbyd/betterybalance/issues"
