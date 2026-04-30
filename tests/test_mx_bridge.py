from ashare_quant.data.mx_bridge import (
    extract_mx_xuangu_rows,
    mx_result_message,
    mx_xuangu_code_name_set,
    summarize_mx_data_result,
    summarize_mx_xuangu_rows,
)


def test_mx_result_message_extracts_error() -> None:
    payload = {"success": False, "status": 113, "message": "今日调用次数已达上限"}
    assert "上限" in mx_result_message(payload)


def test_summarize_mx_data_result_reads_table() -> None:
    payload = {
        "data": {
            "data": {
                "searchDataResultDTO": {
                    "dataTableDTOList": [
                        {
                            "title": "华海清科最新行情",
                            "nameMap": {"f2": "最新价", "f3": "涨跌幅"},
                            "table": {
                                "headName": ["2026-04-08"],
                                "f2": [178.56],
                                "f3": [2.13],
                            },
                        }
                    ]
                }
            }
        }
    }
    text = summarize_mx_data_result(payload)
    assert "华海清科最新行情" in text
    assert "最新价=178.56" in text


def test_extract_and_summarize_mx_xuangu_rows() -> None:
    payload = {
        "data": {
            "data": {
                "allResults": {
                    "result": {
                        "columns": [
                            {"key": "SECURITY_CODE", "title": "证券代码"},
                            {"key": "SECURITY_SHORT_NAME", "title": "股票简称"},
                            {"key": "NEWEST_PRICE", "title": "最新价 (元)"},
                            {"key": "CHG", "title": "涨跌幅 (%)"},
                        ],
                        "dataList": [
                            {
                                "SECURITY_CODE": "002320",
                                "SECURITY_SHORT_NAME": "海峡股份",
                                "NEWEST_PRICE": 8.61,
                                "CHG": 2.5,
                            }
                        ],
                    }
                }
            }
        }
    }
    rows = extract_mx_xuangu_rows(payload)
    assert rows and rows[0]["证券代码"] == "002320"
    assert "海峡股份" in summarize_mx_xuangu_rows(rows)
    identities = mx_xuangu_code_name_set(rows)
    assert "002320" in identities
    assert "海峡股份".upper() in identities
