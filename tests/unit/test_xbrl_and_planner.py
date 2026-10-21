from __future__ import annotations

import pytest

from secrag.analytics.planner import execute_plan, plan_numeric
from secrag.ingest.xbrl import FactStore, _annual_period


def companyfacts(observations: list[dict]) -> dict:
    return {
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": observations}
                }
            }
        },
    }


def obs(start: str, end: str, val: float, **kw) -> dict:
    return {
        "start": start,
        "end": end,
        "val": val,
        "form": "10-K",
        "fp": "FY",
        "accn": kw.get("accn", "a-1"),
        "filed": kw.get("filed", "2024-11-01"),
        "fy": kw.get("fy", 2024),
    }


def test_quarterly_observations_are_rejected() -> None:
    assert _annual_period(obs("2024-06-30", "2024-09-28", 64_698_000_000)) is None


def test_annual_observations_are_accepted() -> None:
    period = _annual_period(obs("2023-10-01", "2024-09-28", 391_035_000_000))
    assert period is not None
    fiscal_year, days = period
    assert fiscal_year == 2024
    assert 340 <= days <= 400


def test_balance_sheet_instants_are_accepted() -> None:
    period = _annual_period({"end": "2024-09-28", "val": 1, "form": "10-K"})
    assert period == (2024, 0)


def test_january_fiscal_year_end_is_labelled_by_its_ending_year() -> None:
    assert _annual_period(obs("2024-01-29", "2025-01-26", 130_497_000_000))[0] == 2025
    assert _annual_period(obs("2025-02-01", "2026-01-31", 700_000_000_000))[0] == 2026


def test_concept_switch_between_years_does_not_lose_data() -> None:
    payload = {
        "entityName": "NVIDIA Corporation",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [obs("2021-02-01", "2022-01-30", 26_914_000_000)]}
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            obs("2023-01-30", "2024-01-28", 60_922_000_000),
                            obs("2024-01-29", "2025-01-26", 130_497_000_000),
                        ]
                    }
                },
            }
        },
    }
    store = FactStore.from_company_facts(payload, "NVDA")

    assert store.get("NVDA", "revenue", 2022).value == 26_914_000_000
    assert store.get("NVDA", "revenue", 2024).value == 60_922_000_000
    assert store.get("NVDA", "revenue", 2025).value == 130_497_000_000


def test_canonical_alias_wins_when_both_report_the_same_year() -> None:
    payload = {
        "entityName": "Test Corp",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [obs("2023-01-01", "2023-12-31", 100.0)]}
                },
                "Revenues": {"units": {"USD": [obs("2023-01-01", "2023-12-31", 999.0)]}},
            }
        },
    }
    store = FactStore.from_company_facts(payload, "TST")
    assert store.get("TST", "revenue", 2023).value == 100.0


def test_quarterly_facts_do_not_reach_the_fact_table() -> None:
    payload = companyfacts(
        [
            obs("2023-10-01", "2024-09-28", 391_035_000_000),
            obs("2024-06-30", "2024-09-28", 64_698_000_000),
        ]
    )
    store = FactStore.from_company_facts(payload, "AAPL")
    fact = store.get("AAPL", "revenue", 2024)
    assert fact is not None
    assert fact.value == 391_035_000_000


def test_restatement_prefers_the_latest_filing() -> None:
    payload = companyfacts(
        [
            obs("2022-10-01", "2023-09-30", 100.0, accn="a-1", filed="2023-11-01"),
            obs("2022-10-01", "2023-09-30", 110.0, accn="a-2", filed="2024-11-01"),
        ]
    )
    store = FactStore.from_company_facts(payload, "AAPL")
    assert store.get("AAPL", "revenue", 2023).value == 110.0


@pytest.fixture
def store() -> FactStore:
    return FactStore.from_company_facts(
        companyfacts(
            [
                obs("2021-10-01", "2022-09-30", 400.0, accn="a1"),
                obs("2022-10-01", "2023-09-30", 450.0, accn="a2"),
                obs("2023-10-01", "2024-09-28", 500.0, accn="a3"),
            ]
        ),
        "AAPL",
    )


def test_growth_is_computed_with_an_auditable_formula(store: FactStore) -> None:
    result = store.growth("AAPL", "revenue", 2022, 2024)
    assert result.value == pytest.approx(25.0)
    assert result.inputs == {"FY2022": 400.0, "FY2024": 500.0}
    assert "FY2024" in result.formula


def test_missing_data_returns_none_rather_than_guessing(store: FactStore) -> None:
    assert store.growth("AAPL", "revenue", 1999, 2024).value is None
    assert store.value_of("MSFT", "revenue", 2024).value is None


def test_series_reports_year_over_year(store: FactStore) -> None:
    frame = store.series("AAPL", "revenue")
    assert list(frame["fiscal_year"]) == [2022, 2023, 2024]
    assert frame["yoy_pct"].iloc[-1] == pytest.approx(11.11, abs=0.01)


def test_plan_detects_value_lookup(store: FactStore) -> None:
    plan = plan_numeric("What was AAPL revenue in 2024?", store)
    assert plan is not None
    assert plan.operation == "value"
    assert plan.metric == "revenue"
    assert plan.years == [2024]


def test_plan_detects_growth_between_two_years(store: FactStore) -> None:
    plan = plan_numeric("How much did AAPL revenue grow from 2022 to 2024?", store)
    assert plan is not None
    assert plan.operation == "growth"
    assert plan.years == [2022, 2024]


def test_single_year_growth_implies_the_prior_year(store: FactStore) -> None:
    plan = plan_numeric("How much did AAPL revenue increase in 2024?", store)
    assert plan is not None
    assert plan.years == [2023, 2024]


def test_plan_resolves_a_named_ratio(store: FactStore) -> None:
    plan = plan_numeric("What was AAPL gross margin in 2024?", store)
    assert plan is not None
    assert plan.operation == "ratio"
    assert plan.ratio_numerator == "gross_profit"
    assert plan.ratio_denominator == "revenue"


def test_plan_refuses_when_no_company_is_indexed(store: FactStore) -> None:
    assert plan_numeric("What was Tesla revenue in 2024?", store) is None


def test_plan_refuses_when_no_metric_is_named(store: FactStore) -> None:
    assert plan_numeric("Tell me about AAPL in 2024", store) is None


def test_execute_plan_drops_unresolvable_results(store: FactStore) -> None:
    plan = plan_numeric("What was AAPL revenue in 2024?", store)
    assert plan is not None
    results = execute_plan(plan, store)
    assert len(results) == 1
    assert results[0].value == 500.0
