CREATE OR REPLACE TABLE main.mart_economic_stress AS
SELECT
    s.geo_id,
    s.date,
    COALESCE(s.jobless_claims_delta, 0) * 0.25
        + COALESCE(s.delinquency_rate, 0) * 0.20
        + COALESCE(s.poverty_rate, 0) * 0.10
        + COALESCE(s.cpi_monthly_delta, 0) * 0.15
        + COALESCE(s.reddit_negativity_score, 0) * 0.07 AS tier1_score,
    COALESCE(s.cpi_monthly_delta, 0) * 0.15 AS tier2_score,
    COALESCE(s.reddit_negativity_score, 0) * 0.07 AS tier3_score,
    COALESCE(s.jobless_claims_delta, 0) * 0.25
        + COALESCE(s.delinquency_rate, 0) * 0.20
        + COALESCE(s.poverty_rate, 0) * 0.10
        + COALESCE(s.cpi_monthly_delta, 0) * 0.15
        + COALESCE(s.reddit_negativity_score, 0) * 0.07 AS ess_score,
    gm.geo_level,
    gm.geo_name,
    CAST(CURRENT_DATE AS DATE) AS run_date,
    '' AS stale_sources,
    '' AS anomaly_flags,
    0.5 AS data_quality_score
FROM main.int_zip_weekly_signals s
LEFT JOIN (
    SELECT DISTINCT geo_id, geo_level, geo_name
    FROM main.stg_openweather
) gm ON s.geo_id = gm.geo_id;
