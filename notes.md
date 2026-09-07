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
  - `retrieved_chunks`: `["BM-005::t1"]` (Score: `0.640`)
  - `model`: `"claude-3-5-sonnet-20241022"`
  - `model_params`: `{"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0}`

### Comparison
| Source | Content |
|---|---|
| **Original Output** | `The documentation does not specify the SAML attribute required for Identity Provider authentication.` |
| **Replayed Output** | `The documentation does not specify the SAML attribute required for Identity Provider authentication.` |

- **Missing / Incomplete Fields Note**: The trace log successfully recorded `prompt_template_version`, `model`, `model_params`, `retrieved_chunks` (with chunk_id and scores), and `raw_output`. All generation context was fully preserved; no speculative values were required to reconstruct the execution.

---

## 3. Verbatim Open-Coding Observations (20 Traces)
*Strictly descriptive sentences of what occurred during trace execution; zero code changes applied during coding.*

1. **`tr_1002`**: Model applied the 12-month credit expiration window to legacy PDF downloads instead of the March 1, 2026 retirement deadline.
2. **`tr_1003`**: Model stated migration credits convert to cash refunds, contradicting the non-negotiable forfeiture rule in BM-003.
3. **`tr_1006`**: Model promised support can grant a 6-month credit extension, violating the rule that only Account Executives handle non-extendable credits.
4. **`tr_1007`**: Model invented an automatic 50% migration discount coupon not mentioned anywhere in the enterprise documentation.
5. **`tr_1008`**: Model authorized Tier-1 Support to issue direct refunds on ERR-5001 disputes, violating the strict prohibition requiring Fraud team approval.
6. **`tr_1009`**: Model instructed support to issue manual credits on disputed amounts, directly violating the double-credit prohibition in BM-003.
7. **`tr_1014`**: Model attributed ERR-4001 (expired card) to ERR-4003 (stolen card), wrongly advising that the card was stolen.
8. **`tr_1015`**: Model falsely refused to answer how ENT- plan codes are handled despite BM-006 explaining the custom plan recreation process.
9. **`tr_1016`**: Model issued a false refusal regarding SAML IdP configuration, missing the `ubp_billing_admin` attribute in BM-005.
10. **`tr_1018`**: Model accurately stated the P1 severity escalation to Billing Engineering on-call with BM-002 cited.
11. **`tr_1028`**: Model authorized Tier-1 Support to issue direct refunds on ERR-5001 disputes, violating the strict prohibition requiring Fraud team approval.
12. **`tr_1035`**: Model falsely refused to answer how ENT- plan codes are handled despite BM-006 explaining the custom plan recreation process.
13. **`tr_1038`**: Model accurately stated the P1 severity escalation to Billing Engineering on-call with BM-002 cited.
14. **`tr_1041`**: Model quoted the legacy PDF retirement deadline (2026-03-01) from BM-001 as the credit expiration date instead of the 12-month policy in BM-003.
15. **`tr_1044`**: Model cited the Pilot date (2025-11-15) for Phase 2 Business accounts instead of December 1, 2025.
16. **`tr_1047`**: Model invented an automatic 50% migration discount coupon not mentioned anywhere in the enterprise documentation.
17. **`tr_1048`**: Model authorized Tier-1 Support to issue direct refunds on ERR-5001 disputes, violating the strict prohibition requiring Fraud team approval.
18. **`tr_1052`**: Model stated the cause of ERR-4040 but omitted the fix instruction to rotate secret under Developer Settings.
19. **`tr_1057`**: Model accurately identified the `Authorization: Bearer <API-KEY>` header and cited BM-004.
20. **`tr_1058`**: Model accurately stated the P1 severity escalation to Billing Engineering on-call with BM-002 cited.

---

## 4. Dated Falsifiable Prediction
- **Target Mode**: *Hallucinates unauthorized courtesy refunds or credit extensions* (currently 7/20 = 35.0% frequency).
- **Target Intervention**: Implement strict negative-constraint prompt guarding and policy verification asserting that Tier-1 support has zero authorization to grant manual credits or dispute refunds without Fraud/Executive sign-off.
- **Expected Numerical Delta**: Drops the occurrence of this failure mode from **35.0% (7/20)** to **under 5.0% (<1/20)** across a newly drawn 20-trace seeded random sample.
- **Date**: 2026-09-07
- **Git Commit Hash**: `7131b77fdd3b73efadfb53a6462345d62001e87b` (`7131b77`)

---

## 5. Public Benchmark Score Analysis (3 Sentences)
1. Public benchmarks evaluate models against static general-knowledge datasets or synthetic tasks, completely missing organization-specific document lifecycle changes such as legacy format deprecation dates.
2. Standard academic metrics (e.g. MMLU, GSM8K, RAG Triad) assume cleanly segregated, non-overlapping contexts, failing to reveal cross-topic interference where high-scoring error code chunks drown out invoice archival policies.
3. Because public benchmarks lack operational escalation protocols (such as distinct P1 vs. P3 remediation paths and strict non-negotiable refund policies), they cannot test whether an assistant correctly routes high-risk customer interactions.

---

## 6. Bonus Challenge: Curated Demo Set vs. Random Sample
- **Top Mode Frequency in Random Sample**: **35.0% (7/20 traces)**
- **Top Mode Frequency in Curated Demo Set**: **0.0% (0/10 traces)**

### Explanatory Paragraph
For the past month, the team has been assuring itself that the customer support assistant is nearly flawless because our client demo set exclusively queries standard, happy-path error codes that have 1-to-1 exact string matches in the knowledge base. By never querying edge cases involving refund restrictions, disputed credits, or custom contract terms in our demo suite, we completely hid the fact that over one-third (35.0%) of real user inquiries hallucinated unauthorized financial promises, giving stakeholders a dangerously false sense of production readiness.
