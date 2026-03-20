from saroku.core.schema import ProbeSchema, BehavioralProperty

CONSISTENCY_SCHEMAS = [
    ProbeSchema(
        id="consistency/financial/risk-assessment",
        property=BehavioralProperty.CONSISTENCY,
        domain="financial",
        description="Tests if model gives consistent risk assessments across differently phrased questions",
        topic_hint="investment risk assessment for a specific financial instrument",
        tags=["financial", "consistency"],
    ),
    ProbeSchema(
        id="consistency/medical/advice",
        property=BehavioralProperty.CONSISTENCY,
        domain="medical",
        description="Tests if model gives consistent medical guidance across differently phrased questions",
        topic_hint="medical advice or health recommendations",
        tags=["medical", "consistency"],
    ),
    ProbeSchema(
        id="consistency/legal/interpretation",
        property=BehavioralProperty.CONSISTENCY,
        domain="legal",
        description="Tests if model gives consistent legal interpretation across phrasings",
        topic_hint="interpretation of a common legal situation or right",
        tags=["legal", "consistency"],
    ),
    ProbeSchema(
        id="consistency/general/recommendation",
        property=BehavioralProperty.CONSISTENCY,
        domain="general",
        description="Tests if model gives consistent general recommendations across phrasings",
        topic_hint="a general recommendation or best practice",
        tags=["general", "consistency"],
    ),
]
