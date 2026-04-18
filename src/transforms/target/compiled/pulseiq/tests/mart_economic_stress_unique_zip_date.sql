-- Singular test: (geo_id, run_date) must be unique in mart_economic_stress.
-- dbt fails this test if any rows are returned (i.e. any duplicate pairs exist).
select
    geo_id,
    run_date,
    count(*) as row_count
from "pulseiq"."main"."mart_economic_stress"
group by geo_id, run_date
having count(*) > 1