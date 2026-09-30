# Standards Reference Data

This directory contains the road marking standards reference data used by the **standards engine** in `src/standards_engine.py`.

## Files

| File | Description |
|---|---|
| `irc35_rules.json` | IRC:35-2015 and MoRTH Section 803 rule entries |

## Important Disclaimer

> **All entries in `irc35_rules.json` carry `"verification_required": true`.**
>
> The dimensions, patterns and requirements listed are indicative references only.
> Engineers **must** verify all requirements against:
> - The **current official edition** of IRC:35-2015 (Indian Roads Congress)
> - The **current edition** of MoRTH Specifications for Road and Bridge Works, Section 803
> - The **sanctioned drawings and project specifications** for the specific road

## How to Add / Modify Rules

Edit `irc35_rules.json` following the schema:

```json
{
  "id":                  "unique identifier",
  "standard":            "IRC:35-2015 | MoRTH Section 803 | ...",
  "section":             "clause/section reference",
  "road_type":           ["urban_arterial", "urban_sub_arterial", ...],
  "marking_type":        "centre_line | lane_line | edge_line | ...",
  "divided_undivided":   "divided | undivided | any",
  "applicability_notes": "plain-language description",
  "engineering_note":    "specific verification requirements",
  "verification_required": true,
  "source_document":     "full document title"
}
```

## No Fabricated Data Policy

Do **not** add entries that invent dimensions not found in the official standard.
If unsure, add an entry with `applicability_notes` directing the engineer to the relevant clause, and set `verification_required: true`.
