CREATE OR REPLACE TABLE main.mart_economic_stress AS
WITH signals AS (
    SELECT * FROM main.int_zip_weekly_signals
),
geo_meta AS (
    SELECT DISTINCT geo_id, geo_level, geo_name
    FROM main.stg_openweather
),
sentiment_coverage AS (
    SELECT DISTINCT zip_code, date
    FROM main.int_sentiment_scores
    WHERE zip_code IS NOT NULL
),
zscores AS (
    SELECT
        geo_id,
        date,
        (jobless_claims_delta - AVG(jobless_claims_delta) OVER ())
            / NULLIF(STDDEV_POP(jobless_claims_delta) OVER (), 0)   AS z_jobless_claims_delta,
        (delinquency_rate - AVG(delinquency_rate) OVER ())
            / NULLIF(STDDEV_POP(delinquency_rate) OVER (), 0)       AS z_delinquency_rate,
        (poverty_rate - AVG(poverty_rate) OVER ())
            / NULLIF(STDDEV_POP(poverty_rate) OVER (), 0)           AS z_poverty_rate,
        (cpi_monthly_delta - AVG(cpi_monthly_delta) OVER ())
            / NULLIF(STDDEV_POP(cpi_monthly_delta) OVER (), 0)      AS z_cpi_monthly_delta,
        (reddit_negativity_score - AVG(reddit_negativity_score) OVER ())
            / NULLIF(STDDEV_POP(reddit_negativity_score) OVER (), 0) AS z_reddit_negativity_score
    FROM signals
),
final AS (
    SELECT
        z.geo_id,
        z.date,
        COALESCE(z.z_jobless_claims_delta, 0) * 0.25
            + COALESCE(z.z_delinquency_rate, 0) * 0.20
            + COALESCE(z.z_poverty_rate, 0) * 0.10 AS tier1_score,
        COALESCE(z.z_cpi_monthly_delta, 0) * 0.15 AS tier2_score,
        COALESCE(z.z_reddit_negativity_score, 0) * 0.07 AS tier3_score,
        COALESCE(z.z_jobless_claims_delta, 0) * 0.25
            + COALESCE(z.z_delinquency_rate, 0) * 0.20
            + COALESCE(z.z_poverty_rate, 0) * 0.10
            + COALESCE(z.z_cpi_monthly_delta, 0) * 0.15
            + COALESCE(z.z_reddit_negativity_score, 0) * 0.07 AS ess_score,
        gm.geo_level,
        gm.geo_name,
        CAST(CURRENT_DATE AS DATE) AS run_date,
        '' AS stale_sources,
        '' AS anomaly_flags,
        (CASE WHEN sc.zip_code IS NOT NULL THEN 1.0 ELSE 0.0 END) AS data_quality_score
    FROM zscores z
    LEFT JOIN geo_meta gm ON z.geo_id = gm.geo_id
    LEFT JOIN sentiment_coverage sc ON z.geo_id = sc.zip_code AND z.date = sc.date
)
SELECT * FROM final;
