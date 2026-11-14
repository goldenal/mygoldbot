"""Calendar importers: the Fed page parser, .ics, FRED (mocked), merge/write, and refusal to mislabel or guess."""
import json

import pandas as pd
import pytest

from goldsys.data import calendar_import as ci
from goldsys.risk.calendar import EventCalendar

FOMC_HTML = """
<div class="panel panel-default">
    <div class="panel-heading"><h4><a id="1">2024 FOMC Meetings</a></h4></div>
    <div class="row fomc-meeting" ">
        <div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
        <div class="fomc-meeting__date col-xs-4">30-31</div>
        <div class="col-xs-12"><strong>Statement:</strong><br><a href="/newsevents/pressreleases/monetary20240131a.htm">HTML</a></div>
        <div class="fomc-meeting__minutes"><strong>Minutes:</strong><br> (Released February 21, 2024)</div>
    </div>
    <div class="fomc-meeting--shaded row fomc-meeting" ">
        <div class="fomc-meeting--shaded fomc-meeting__month col-xs-5"><strong>Apr/May</strong></div>
        <div class="fomc-meeting--shaded fomc-meeting__date col-xs-4">30-1</div>
        <div class="col-xs-12"><a href="/newsevents/pressreleases/monetary20240501a.htm">HTML</a></div>
    </div>
    <div class="row fomc-meeting" ">
        <div class="fomc-meeting__month col-xs-5"><strong>March</strong></div>
        <div class="fomc-meeting__date col-xs-4">19-20*</div>
    </div>
    <div class="row fomc-meeting" ">
        <div class="fomc-meeting__month col-xs-5"><strong>August</strong></div>
        <div class="fomc-meeting__date col-xs-4">22 (unscheduled)</div>
    </div>
</div>
<div class="panel panel-default">
    <div class="panel-heading"><h4><a id="2">2027 FOMC Meetings</a></h4></div>
    <div class="row fomc-meeting" ">
        <div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
        <div class="fomc-meeting__date col-xs-4">26-27</div>
    </div>
</div>
"""


def test_fomc_parse_dates_times_and_diagnostics():
    ev, diag = ci.parse_fomc(FOMC_HTML)
    st = {e.ts.strftime("%Y-%m-%d %H:%M") for e in ev if e.name == "FOMC Statement"}
    assert st == {"2024-01-31 19:00", "2024-05-01 18:00", "2024-03-20 18:00", "2027-01-27 19:00"}      # 14:00 New York: EST = 19:00 UTC, EDT = 18:00 UTC
    assert [e.ts.strftime("%Y-%m-%d %H:%M") for e in ev if e.name == "FOMC Minutes"] == ["2024-02-21 19:00"]
    assert diag["meetings"] == 4 and diag["link_checked"] == 2 and diag["link_mismatch"] == [] and any("unscheduled" in x for x in diag["skipped"])
    assert all(e.tier == 1 for e in ev if e.name == "FOMC Statement") and all(e.tier == 2 for e in ev if e.name == "FOMC Minutes")


def test_fomc_link_date_overrides_the_table_and_the_disagreement_is_reported():
    html = FOMC_HTML.replace("30-31", "30-30")                                   # the table says Jan 30; the statement link says Jan 31
    ev, diag = ci.parse_fomc(html)
    assert diag["link_mismatch"] == [("2024-01-30", "2024-01-31")]
    assert pd.Timestamp("2024-01-31 19:00", tz="UTC") in {e.ts for e in ev}


def test_fomc_refuses_a_page_it_cannot_read():
    with pytest.raises(ci.CalendarImportError, match="layout"):
        ci.parse_fomc("<html><body>maintenance</body></html>")


class Resp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._p


def fred_get(names=None, dates=None):
    names = names or {50: "Employment Situation", 10: "Consumer Price Index"}
    dates = dates or {50: ["2024-11-01", "2024-11-08", "2014-01-10"], 10: ["2025-03-12"]}

    def get(url, params=None, **kw):
        rid = int(params["release_id"])
        if url.endswith("/release"):
            return Resp(200, {"releases": [{"id": rid, "name": names.get(rid, "Something Else")}]})
        return Resp(200, {"release_dates": [{"release_id": rid, "date": d} for d in dates.get(rid, [])]})
    return get


KEY = "a" * 32


def test_fred_dates_become_0830_new_york_with_correct_daylight_saving():
    ev, diag = ci.fetch_fred(KEY, "2015-01-01", "2027-12-31", releases={50: ("employment situation", "US Non-Farm Payrolls"), 10: ("consumer price index", "US CPI")}, get=fred_get())
    got = {(e.name, e.ts.strftime("%Y-%m-%d %H:%M")) for e in ev}
    assert ("US Non-Farm Payrolls", "2024-11-01 12:30") in got           # 1 Nov 2024 is still EDT (UTC-4)
    assert ("US Non-Farm Payrolls", "2024-11-08 13:30") in got           # 8 Nov 2024 is EST (UTC-5)
    assert ("US CPI", "2025-03-12 12:30") in got                         # DST began 9 Mar 2025
    assert not any(e.ts.year == 2014 for e in ev) and diag["US Non-Farm Payrolls"] == 2     # outside the requested range


def test_fred_refuses_to_label_a_release_it_does_not_recognise():
    with pytest.raises(ci.CalendarImportError, match="refusing to label"):
        ci.fetch_fred(KEY, releases={50: ("employment situation", "US Non-Farm Payrolls")}, get=fred_get(names={50: "Quarterly Services Survey"}))


def test_fred_needs_a_real_looking_key_and_reports_http_errors():
    for bad in ["", "short", "A" * 32]:
        with pytest.raises(ci.CalendarImportError, match="32-character"):
            ci.fetch_fred(bad)
    with pytest.raises(ci.CalendarImportError, match="HTTP 400"):
        ci.fetch_fred(KEY, releases={50: ("employment situation", "x")}, get=lambda *a, **k: Resp(400, {"error_message": "bad key"}))


ICS = """BEGIN:VCALENDAR
BEGIN:VEVENT
SUMMARY:Employment Situation for October 2026
DTSTART;TZID=America/New_York:20261106T083000
END:VEVENT
BEGIN:VEVENT
SUMMARY:Consumer Price Index for October 2026
DTSTART:20261110T133000Z
END:VEVENT
BEGIN:VEVENT
SUMMARY:Producer Price Index
 for October 2026
DTSTART;VALUE=DATE:20261112
END:VEVENT
BEGIN:VEVENT
SUMMARY:Real Earnings for October 2026
DTSTART;TZID=America/New_York:20261110T083000
END:VEVENT
END:VCALENDAR
"""


def test_ics_timezones_all_day_and_unrecognised_entries():
    ev, diag = ci.parse_ics(ICS.replace("\n", "\r\n"))
    got = {e.name: e.ts.strftime("%Y-%m-%d %H:%M") for e in ev}
    assert got == {"US Non-Farm Payrolls": "2026-11-06 13:30", "US CPI": "2026-11-10 13:30", "US PPI": "2026-11-12 13:30"}
    assert diag == {"vevents": 4, "ignored": 1}


def test_csv_roundtrip_merge_and_write(tmp_path):
    f = tmp_path / "ev.csv"
    f.write_text("ts_utc,name,tier\n2024-05-03 12:30,US Non-Farm Payrolls,1\n2024-05-01 18:00,FOMC Statement,1\n")
    ev, d = ci.read_csv(f)
    assert d["rows"] == 2
    out = tmp_path / "cal" / "events.csv"
    assert ci.write(ci.merge([], ev), out) == 2
    again = ci.merge(ci.load_existing(out), [ci.Event(pd.Timestamp("2024-05-01 18:00", tz="UTC"), "FOMC Statement", 1, "fed")])      # same event again: replaced, not duplicated
    assert len(again) == 2 and [e.name for e in again] == ["FOMC Statement", "US Non-Farm Payrolls"]
    cal = EventCalendar.from_csv(out)                                                    # the blackout's own loader reads the file we write
    assert len(cal) == 2 and cal.nearest(pd.Timestamp("2024-05-01 17:50", tz="UTC"), 1)[1] == "FOMC Statement"
    with pytest.raises(ci.CalendarImportError):
        ci.write([ci.Event(pd.Timestamp("1999-01-01", tz="UTC"), "x", 1, "")], tmp_path / "bad.csv")
    (tmp_path / "bad_tier.csv").write_text("ts_utc,name,tier\n2024-05-03 12:30,x,9\n")
    with pytest.raises(ci.CalendarImportError):
        ci.read_csv(tmp_path / "bad_tier.csv")
    (tmp_path / "no_cols.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ci.CalendarImportError):
        ci.read_csv(tmp_path / "no_cols.csv")


def test_coverage_table_shows_gaps():
    ev = [ci.make("US CPI", pd.Timestamp("2024-03-12").date(), "t"), ci.make("FOMC Statement", pd.Timestamp("2025-03-19").date(), "t")]
    cov = ci.coverage(ev)
    assert cov.loc["US CPI", 2024] == 1 and cov.loc["US CPI", 2025] == 0
