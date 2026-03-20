from saroku.probe_schemas.sycophancy import SYCOPHANCY_SCHEMAS
from saroku.probe_schemas.honesty import HONESTY_SCHEMAS
from saroku.probe_schemas.consistency import CONSISTENCY_SCHEMAS

ALL_SCHEMAS = SYCOPHANCY_SCHEMAS + HONESTY_SCHEMAS + CONSISTENCY_SCHEMAS

SCHEMA_MAP = {s.id: s for s in ALL_SCHEMAS}
