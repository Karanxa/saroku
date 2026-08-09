from saroku.execution.metrics import ClassifierInvocation, ExecutionMetrics


def _inv(classifier_id="custom:a", property_name="honesty", latency_ms=10.0, outcome="confident",
          confidence=0.9, is_safe=True, layer_name="l1", strategy="cascade", started_at=0.0):
    return ClassifierInvocation(
        classifier_id=classifier_id,
        property_name=property_name,
        started_at=started_at,
        latency_ms=latency_ms,
        outcome=outcome,
        confidence=confidence,
        is_safe=is_safe,
        layer_name=layer_name,
        strategy=strategy,
    )


def test_record_and_to_list():
    metrics = ExecutionMetrics()
    assert metrics.to_list() == []
    inv = _inv()
    metrics.record(inv)
    assert metrics.to_list() == [inv]
    assert len(metrics) == 1
    assert list(metrics) == [inv]


def test_reset_clears_invocations():
    metrics = ExecutionMetrics()
    metrics.record(_inv())
    metrics.reset()
    assert metrics.to_list() == []
    assert len(metrics) == 0


def test_summary_per_classifier_stats():
    metrics = ExecutionMetrics()
    metrics.record(_inv(classifier_id="custom:a", latency_ms=10.0, outcome="confident"))
    metrics.record(_inv(classifier_id="custom:a", latency_ms=20.0, outcome="deferred", confidence=0.2))
    metrics.record(_inv(classifier_id="custom:a", latency_ms=30.0, outcome="timeout", confidence=None, is_safe=None))

    summary = metrics.summary()
    stats = summary["by_classifier"]["custom:a"]
    assert stats["call_count"] == 3
    assert stats["avg_latency_ms"] == 20.0
    assert stats["p50_latency_ms"] == 20.0
    assert round(stats["confident_rate"], 4) == round(1 / 3, 4)
    assert round(stats["timeout_rate"], 4) == round(1 / 3, 4)


def test_summary_per_property_escalation_tracking():
    metrics = ExecutionMetrics()
    metrics.record(_inv(classifier_id="custom:a", layer_name="l1", outcome="deferred", confidence=0.2))
    metrics.record(_inv(classifier_id="custom:b", layer_name="l2", outcome="confident", confidence=0.95))

    summary = metrics.summary()
    prop_stats = summary["by_property"]["honesty"]
    assert prop_stats["call_count"] == 2
    assert prop_stats["layers_tried"] == ["l1", "l2"]
    assert prop_stats["escalated_past_first_layer"] is True
    assert prop_stats["resolved_by_classifier"] == "custom:b"
    assert prop_stats["resolved_by_layer"] == "l2"


def test_summary_single_layer_not_escalated():
    metrics = ExecutionMetrics()
    metrics.record(_inv(classifier_id="custom:a", layer_name="l1", outcome="confident", confidence=0.95))

    summary = metrics.summary()
    prop_stats = summary["by_property"]["honesty"]
    assert prop_stats["escalated_past_first_layer"] is False
    assert prop_stats["resolved_by_classifier"] == "custom:a"


def test_to_dict_matches_summary():
    metrics = ExecutionMetrics()
    metrics.record(_inv())
    assert metrics.to_dict() == metrics.summary()


def test_empty_summary_has_no_stats():
    metrics = ExecutionMetrics()
    summary = metrics.summary()
    assert summary["total_invocations"] == 0
    assert summary["by_classifier"] == {}
    assert summary["by_property"] == {}
