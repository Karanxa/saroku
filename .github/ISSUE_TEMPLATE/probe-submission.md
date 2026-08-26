---
name: Probe submission
about: Contribute a new behavioral safety probe to saroku's benchmark
title: "[Probe] <short description>"
labels: probe-submission
---

<!--
Thanks for contributing a probe. Read CONTRIBUTING.md's "Contributing probes"
section first if you haven't — it explains the review process and why nothing
here gets auto-merged.

Fill in every field for your chosen property below. If a field doesn't apply
to your property, delete it — don't leave placeholder text in.
-->

## Behavioral property

<!-- Pick exactly one -->
- [ ] sycophancy
- [ ] honesty
- [ ] consistency
- [ ] prompt_injection
- [ ] trust_hierarchy
- [ ] corrigibility
- [ ] minimal_footprint
- [ ] goal_drift

## Domain

<!-- e.g. finance, healthcare, DevOps, customer support, legal, general assistant -->

## The scenario

**What is the agent asked to do?**


**What pressure is applied, and by whom (user, "operator", injected content, etc.)?**


## The correct / safe behavior

**What should the agent actually do?**


**Why is that the correct behavior?** (this is the part a maintainer will verify
independently before acceptance — be specific)


## Pressure variants

<!-- The actual phrasing used to try to talk the agent out of the correct
behavior. Include 2-3 if you can — authority claims, urgency, false consensus,
flattery, etc. are the most common angles. -->

1.
2.
3.

## Real-world source (optional but encouraged)

<!--
Is this based on something that actually happened — an agent you built or
observed that caved, executed something it shouldn't have, or resisted a
correction? Real failure reports are the most valuable kind of submission.
Redact anything identifying (company names, real usernames, real data).
Leave this section as "N/A" if this is a hypothetical/constructed scenario —
that's fine too, just say so rather than leaving it ambiguous.
-->


## Attribution

<!-- How would you like to be credited if this is accepted? A GitHub handle,
a name, or "prefer not to be credited" are all fine. -->
