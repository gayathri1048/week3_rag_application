# Customer Support RAG Error Taxonomy

| Failure Mode Name | Count | Frequency (%) | Severity | Example Trace ID |
|---|---|---|---|---|
| **Returns payment card stolen warning for invoice archive questions** | 3 | 15.0% | Annoys user | `tr_1008` |
| **Returns unrelated credit expiration dates for card error queries** | 1 | 5.0% | Annoys user | `tr_1002` |
| **Substitutes concurrent login lockout info for credit policy inquiries** | 1 | 5.0% | Annoys user | `tr_1006` |
| **Accurately answers error code fixes & escalation workflows** | 10 | 50.0% | N/A (Success) | `tr_1003` |
| **Accurately enforces non-negotiable policy & API specifications** | 5 | 25.0% | N/A (Success) | `tr_1007` |

---
*Summary: 15 / 20 traces (75%) succeeded; 5 / 20 traces (25%) exhibited cross-topic context interference or semantic drift. Total sample: 20 traces.*
