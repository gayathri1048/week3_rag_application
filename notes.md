# Week 5 / Week 6 Error Analysis Notes

## 1. Seeded Random Sample
- **Random Seed**: `42`
- **Total Trace Pool Size**: 60 traces
- **Sampled 20 Trace IDs**:
  1. `tr_1002`
  2. `tr_1003`
  3. `tr_1006`
  4. `tr_1007`
  5. `tr_1008`
  6. `tr_1009`
  7. `tr_1014`
  8. `tr_1015`
  9. `tr_1016`
  10. `tr_1018`
  11. `tr_1028`
  12. `tr_1035`
  13. `tr_1038`
  14. `tr_1041`
  15. `tr_1044`
  16. `tr_1047`
  17. `tr_1048`
  18. `tr_1052`
  19. `tr_1057`
  20. `tr_1058`

---

## 2. Replay Evidence
- **Replay Target**: `tr_1016`
- **Replay Selection Seed**: `1048`
- **Trace Reconstructed Fields**:
  - `prompt_template_version`: `"v2.4-support-rag"`
  - `retrieved_chunks`: `["BM-007::t1"]` (Score: `0.890`)
  - `model`: `"claude-3-5-sonnet-20241022"`
  - `model_params`: `{"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0}`

### Comparison
| Source | Content |
|---|---|
| **Original Output** | `Tier-2 locks invoice and routes to Risk & Fraud; automated renewal is paused 14 days; Tier-1 must never refund directly (BM-007).` |
| **Replayed Output** | `Tier-2 locks invoice and routes to Risk & Fraud; automated renewal is paused 14 days; Tier-1 must never refund directly (BM-007).` |

- **Missing / Incomplete Fields Note**: The trace log successfully recorded `prompt_template_version`, `model`, `model_params`, `retrieved_chunks` (with chunk_id and scores), and `raw_output`. All generation context was fully preserved; no speculative values were required to reconstruct the execution.

---

## 3. Verbatim Open-Coding Observations (20 Traces)
*Strictly descriptive sentences of what occurred during trace execution; zero code changes applied.*

1. **`tr_1002`**: Customer asked what ERR-4001 means for their card, but the model responded with credit expiration and legacy PDF format retirement dates.
2. **`tr_1003`**: Model gave the exact repair command `billing-repair --invoice <INV-ID>` and cited BM-002 to resolve ERR-4010.
3. **`tr_1006`**: User asked about migration credit expiry, but the model returned an error description about ERR-4030 concurrent login lockouts.
4. **`tr_1007`**: Model correctly stated that Support cannot extend credits and accurately directed escalations to the Account Executive with BM-003 cited.
5. **`tr_1008`**: User inquired about post-cutover legacy PDF invoice availability, but the model responded with card re-tokenisation and stolen card warnings.
6. **`tr_1009`**: Model stated that non-USD credits use the ECB reference rate on the migration date with BM-003 citation.
7. **`tr_1014`**: Model identified the `Authorization: Bearer <API-KEY>` header for UBP API requests and cited BM-004.
8. **`tr_1015`**: Model classified ERR-4003 as a P1 condition and routed it to Billing Engineering on-call citing BM-002.
9. **`tr_1016`**: Model detailed the 14-day renewal pause, Tier-2 routing to Risk & Fraud, and forbade direct refunds under BM-007.
10. **`tr_1018`**: Model provided the `ledger-sync --dispute <DISPUTE-ID>` command citing BM-007.
11. **`tr_1028`**: Customer asked about legacy PDF invoice availability after March 2026, but the model outputted instructions regarding failed card re-tokenisation and stolen card bank alerts.
12. **`tr_1035`**: Model correctly identified ERR-4003 as P1 severity and directed escalation to Billing Engineering on-call.
13. **`tr_1038`**: Model outputted the `ledger-sync --dispute <DISPUTE-ID>` remediation command with citation to BM-007.
14. **`tr_1041`**: Model provided the manual custom plan creation procedure in UBP Admin for ERR-4032 with BM-002 citation.
15. **`tr_1044`**: Model accurately directed the customer to regenerate their secret under Developer Settings → Webhooks → Rotate Secret citing BM-002.
16. **`tr_1047`**: Model explicitly confirmed that Support cannot extend the 12-month credit window and cited BM-003.
17. **`tr_1048`**: In response to a query about legacy PDF access after 2026-03-01, the model warned about stolen payment cards and failed tokenisation.
18. **`tr_1052`**: Model rejected the claim of a 50% migration discount, confirming that subscription pricing remains unchanged with BM-006 cited.
19. **`tr_1057`**: Model enforced the policy prohibiting Tier-1 Support from issuing direct refunds on ERR-5001 chargebacks.
20. **`tr_1058`**: Model supplied the exact CLI syntax `ledger-sync --dispute <DISPUTE-ID>` to re-balance currency wallets citing BM-007.

---

## 4. Dated Falsifiable Prediction
- **Target Mode**: *Returns payment card stolen warning for invoice archive questions* (currently 3/20 = 15.0% frequency).
- **Target Intervention**: Implement BM25 lexical keyword boosting for document archive queries (boosting terms like `legacy`, `PDF`, `archive`, `INV-`) over generic error code documents in the hybrid retriever.
- **Expected Numerical Delta**: Drops the occurrence of this mode from **15.0% (3/20)** to **0.0% (0/20)** across a newly drawn 20-trace seeded random sample.
- **Date**: 2026-09-07
- **Git Commit Hash**: `[COMMITTED_BELOW]`

---

## 5. Public Benchmark Score Analysis (3 Sentences)
1. Public benchmarks evaluate models against static general-knowledge datasets or synthetic reading comprehension tasks, completely missing organization-specific document lifecycle changes such as legacy format deprecation dates.
2. Standard academic metrics (e.g. MMLU, GSM8K, RAG Triad) assume cleanly segregated, non-overlapping contexts, failing to reveal cross-topic interference where high-scoring error code chunks drown out invoice archival policies.
3. Because public benchmarks lack operational escalation protocols (such as distinct P1 vs. P3 remediation paths and strict non-negotiable refund policies), they cannot test whether an assistant correctly routes high-risk customer interactions.

---

## 6. Bonus Challenge: Curated Demo Set vs. Random Sample
- **Top Mode Frequency in Random Sample**: **15.0% (3/20 traces)**
- **Top Mode Frequency in Curated Demo Set**: **0.0% (0/10 traces)**

### Explanatory Paragraph
For the past month, the team has been assuring itself that the customer support assistant is nearly flawless because our client demo set exclusively queries standard, happy-path error codes (like `ERR-4001` or `ERR-4010`) that have 1-to-1 exact string matches in the knowledge base. By never querying ambiguous, multi-topic workflows (such as legacy invoice PDF access dates vs. payment token cutover rules) in our demo suite, we masked the fact that nearly 1 in 6 real-world interactions pulled unrelated payment error warnings, giving stakeholders a false sense of reliability.
