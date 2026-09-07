# Customer Support RAG Error Taxonomy

| Failure Mode Name | Count | Frequency (%) | Severity | Example Trace ID |
|---|---|---|---|---|
| **Hallucinates unauthorized courtesy refunds or credit extensions** | 7 | 35.0% | Embarrasses client | `tr_1008` |
| **Quotes superseded policy or retired cutover dates** | 4 | 20.0% | Embarrasses client | `tr_1041` |
| **False refusal on valid custom / enterprise contract queries** | 3 | 15.0% | Annoys user | `tr_1016` |
| **Confuses similar error codes or symptoms (ERR-4001 vs ERR-4003)** | 1 | 5.0% | Annoys user | `tr_1014` |
| **Omits step-by-step resolution when error code splits across chunks** | 1 | 5.0% | Annoys user | `tr_1052` |
| **Accurately answers error code fixes & escalation workflows** | 4 | 20.0% | N/A (Success) | `tr_1057` |

---
*Summary: 16 / 20 traces (80.0%) failed across 5 distinct failure modes; 4 / 20 traces (20.0%) succeeded. Total sample: 20 traces.*
