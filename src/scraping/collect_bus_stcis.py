from pathlib import Path
from datetime import date, timedelta
import requests
import time


BASE_URL = "https://stcis.go.kr/pivotIndi/indicatorAjax.do"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = date(2020, 11, 1)
END_DATE = date(2026, 5, 19)

def create_session():
    session = requests.Session()

    # Initialise la session JSP
    session.get(
        "https://stcis.go.kr/pivotIndi/wpsPivotIndicator.do"
        "?indiClss=IC04&indiSel=IC0402&siteGb=P",
        timeout=60,
    )

    return session


def build_payload(start_day, end_day, origin_zone_sd, origin_zone_sgg, dest_zone_sd, dest_zone_sgg):
    return {
        # Some fields seem to be ignored but are required by the server
        "indiCd": "Z01713",
        "siteGb": "P",
        "indiNm": "이용객 수요(O/D)(철도·고속·시외버스이용 O/D)",

        "searchDateGubun": "3",

        "searchFromYear": str(start_day.year),
        "searchToYear": str(end_day.year),

        "searchFromMonth": start_day.strftime("%Y-%m"),
        "searchToMonth": end_day.strftime("%Y-%m"),

        "searchFromDay": start_day.strftime("%Y-%m-%d"),
        "searchFromDayDD": start_day.strftime("%Y%m%d"),
        "searchToDay": end_day.strftime("%Y-%m-%d"),

        "searchAreaGubun": "2",

        "zoneSd": "",
        "zoneSgg": "",
        "zoneEmd": "",
        "zoneDstrct": "",

        "selectZoneSd": "",
        "selectZoneSgg": "",

        "tcboId": "",
        "excclcAreaCd": "",
        "routeId": "",
        "routeSdCd": "",
        "routeSggCd": "",

        "tcboIdSttn": "",
        "excclcAreaCdSttn": "",
        "sttnId": "",
        "sttnIdGrp": "",
        "sttnSdCd": "",
        "sttnSggCd": "",

        "searchODAreaGubun": "2",
        "searchODAreaGubun_2": "2",

        "rdStgptSel": "Y",

        "searchStgptZoneSd": {origin_zone_sd},
        "searchStgptZoneSgg": {origin_zone_sgg},
        "searchStgptZoneEmd": "",

        "rdAlocSel": "Y",

        "searchAlocZoneSd": {dest_zone_sd},
        "searchAlocZoneSgg": {dest_zone_sgg},
        "searchAlocZoneEmd": "",

        "pgngYn": "Y",

        "daybyTblNm": "DM_OD_NTSS_T",
        "mnbyTblNm": "DM_MMBY_OD_NTSS_T",
        "yrbyTblNm": "DM_YRBY_OD_NTSS_T",

        "dstrctTblNm": "",
        "mnbyDstrctTblNm": "",
        "yrbyDstrctTblNm": "",
    }


def daterange():
    current = START_DATE

    while current <= END_DATE:

        end = min(
            current + timedelta(days=27),
            END_DATE
        )

        yield current, end

        current = end + timedelta(days=1)


def import_transportation_data(origin, origin_zone_sd, origin_zone_sgg, destination, dest_zone_sd, dest_zone_sgg):

    session = create_session()

    headers = {
        "Origin": "https://stcis.go.kr",
        "Referer": (
            "https://stcis.go.kr/pivotIndi/"
            "wpsPivotIndicator.do"
            "?indiClss=IC04&indiSel=IC0402&siteGb=P"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }

    for start_day, end_day in daterange():
        filename = (
            OUTPUT_DIR
            / f"{origin}_to_{destination}"
            / f"{origin}_to_{destination}_{start_day:%Y%m%d}_{end_day:%Y%m%d}.html"
        )

        if filename.exists():
            print(f"SKIP {filename.name}")
            continue

        payload = build_payload(start_day, end_day, origin_zone_sd, origin_zone_sgg, dest_zone_sd, dest_zone_sgg)

        print(
            f"Downloading "
            f"{start_day} -> {end_day}"
        )

        r = session.post(
            BASE_URL,
            data=payload,
            headers=headers,
            timeout=60,
        )

        r.raise_for_status()

        filename.write_text(
            r.text,
            encoding="utf-8"
        )

        time.sleep(1)

if __name__ == "__main__":
    import_transportation_data(
        origin="pohang",
        origin_zone_sd="47",
        origin_zone_sgg={"47111","47113"},
        destination="seoul",
        dest_zone_sd="11",
        dest_zone_sgg="",
    )
    import_transportation_data(
        origin="seoul",
        origin_zone_sd="11",
        origin_zone_sgg="",
        destination="pohang",
        dest_zone_sd="47",
        dest_zone_sgg={"47111","47113"},
    )