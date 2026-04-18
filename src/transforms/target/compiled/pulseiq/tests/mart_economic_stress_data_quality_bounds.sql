-- Singular test: data_quality_score must be between 0.0 and 1.0 (inclusive).
-- dbt fails this test if any rows are returned (i.e. any out-of-bounds values exist).
select
    geo_id,
    run_date,
    data_quality_score
from "pulseiq"."main"."mart_economic_stress"
where data_quality_score < 0
   or data_quality_score > 1