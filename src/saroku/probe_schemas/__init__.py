from saroku.probe_schemas.sycophancy import SYCOPHANCY_SCHEMAS
from saroku.probe_schemas.honesty import HONESTY_SCHEMAS
from saroku.probe_schemas.consistency import CONSISTENCY_SCHEMAS
from saroku.probe_schemas.prompt_injection import PROMPT_INJECTION_SCHEMAS
from saroku.probe_schemas.trust_hierarchy import TRUST_HIERARCHY_SCHEMAS
from saroku.probe_schemas.corrigibility import CORRIGIBILITY_SCHEMAS
from saroku.probe_schemas.minimal_footprint import MINIMAL_FOOTPRINT_SCHEMAS
from saroku.probe_schemas.goal_drift import GOAL_DRIFT_SCHEMAS

ALL_SCHEMAS = (
    SYCOPHANCY_SCHEMAS
    + HONESTY_SCHEMAS
    + CONSISTENCY_SCHEMAS
    + PROMPT_INJECTION_SCHEMAS
    + TRUST_HIERARCHY_SCHEMAS
    + CORRIGIBILITY_SCHEMAS
    + MINIMAL_FOOTPRINT_SCHEMAS
    + GOAL_DRIFT_SCHEMAS
)

SCHEMA_MAP = {s.id: s for s in ALL_SCHEMAS}
