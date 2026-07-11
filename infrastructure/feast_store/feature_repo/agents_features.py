from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource
)
from feast.types import Float32
from feast.value_type import ValueType

agent = Entity(
    name="agent_id",
    value_type=ValueType.STRING,
    description="Unique operational ID of the BPO Support Agent"
)

agent_historical_source = FileSource(
    path=f"data/agent_historical_telemetry.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

agent_performance_view = FeatureView(
    name="agent_performance_metrics",
    entities=[agent],
    ttl=timedelta(days=30),
    schema=[
        Field(name="historical_mean_aht_fraud", dtype=Float32),
        Field(name="historical_mean_aht_billing", dtype=Float32),
        Field(name="historical_fcr_fraud", dtype=Float32),
        Field(name="historical_fcr_billing", dtype=Float32),
        Field(name="current_cognitive_stress_index", dtype=Float32),
    ],
    online=True,
    source=agent_historical_source,
    tags={"domain": "bpo_operations", "tier": "routing"}
)