#!/usr/bin/env python3
"""Generate realistic production customer support RAG traces for Week 5 / Week 6 Error Analysis.
Reflects an uncurated production log with diverse, realistic failure modes across billing and migration.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "traces.jsonl"

PROMPT_TEMPLATE_VERSION = "v2.4-support-rag"
CHAT_SYSTEM_PROMPT = (
    "You are an expert customer support assistant for the Unified Billing Platform (UBP) migration. "
    "Use only the provided documentation context to answer customer inquiries accurately, citing relevant article IDs and error codes. "
    "If the context does not contain the answer, state clearly that you do not have sufficient information."
)

# 20 distinct scenarios designed to represent realistic production failure patterns:
# 16 Failure cases across 5 distinct failure modes + 4 Clean Successes.
TRACES_POOL = [
    # --- Mode 1: Quotes superseded policy or retired cutover dates (5 traces) ---
    {
        "trace_id": "tr_1001",
        "user_query": "What is the expiration timeframe for unused migration credits?",
        "chunk_id": "BM-001::t3",
        "score": 0.824,
        "raw_output": "Migration credits expire on 2026-03-01 along with the retirement of legacy PDF invoice formats.",
        "obs": "Model quoted the legacy PDF retirement deadline (2026-03-01) from BM-001 as the credit expiration date instead of the 12-month policy in BM-003."
    },
    {
        "trace_id": "tr_1002",
        "user_query": "How long do we have to download legacy invoice PDFs before retirement?",
        "chunk_id": "BM-003::t2",
        "score": 0.791,
        "raw_output": "You have 12 months from your migration date to download all legacy invoice PDFs.",
        "obs": "Model applied the 12-month credit expiration window to legacy PDF downloads instead of the March 1, 2026 retirement deadline."
    },
    {
        "trace_id": "tr_1003",
        "user_query": "Can unused migration credits be converted into cash refunds after 12 months?",
        "chunk_id": "BM-001::t2",
        "score": 0.812,
        "raw_output": "Prorated credits from the old system convert at a 1:1 cash rate upon request to Tier-1 Support.",
        "obs": "Model stated migration credits convert to cash refunds, contradicting the non-negotiable forfeiture rule in BM-003."
    },
    {
        "trace_id": "tr_1004",
        "user_query": "When is the cutover date for Phase 2 Business plan accounts?",
        "chunk_id": "BM-001::t1",
        "score": 0.778,
        "raw_output": "Phase 2 Business accounts were migrated on 2025-11-15 during the Pilot phase.",
        "obs": "Model cited the Pilot date (2025-11-15) for Phase 2 Business accounts instead of December 1, 2025."
    },
    {
        "trace_id": "tr_1005",
        "user_query": "What happens to draft invoices when our account is migrated?",
        "chunk_id": "BM-003::t1",
        "score": 0.765,
        "raw_output": "Draft invoices are copied as-is into UBP with their original due dates preserved.",
        "obs": "Model stated draft invoices are copied to UBP, when the policy states draft invoices are discarded."
    },

    # --- Mode 2: Hallucinates non-existent discounts or courtesy extensions (4 traces) ---
    {
        "trace_id": "tr_1006",
        "user_query": "Can support extend my migration credit expiration by 6 months?",
        "chunk_id": "BM-003::t2",
        "score": 0.840,
        "raw_output": "Yes, our support team can issue a temporary 6-month extension if you open a ticket with Billing Operations.",
        "obs": "Model promised support can grant a 6-month credit extension, violating the rule that only Account Executives handle non-extendable credits."
    },
    {
        "trace_id": "tr_1007",
        "user_query": "Do enterprise accounts get an automatic 50% discount coupon upon completing migration?",
        "chunk_id": "BM-006::t0",
        "score": 0.690,
        "raw_output": "Yes, all enterprise accounts receive an automatic 50% discount coupon upon completing migration addendum.",
        "obs": "Model invented an automatic 50% migration discount coupon not mentioned anywhere in the enterprise documentation."
    },
    {
        "trace_id": "tr_1008",
        "user_query": "Can Tier-1 Support issue a direct courtesy refund for ERR-5001 dispute notice?",
        "chunk_id": "BM-007::t2",
        "score": 0.875,
        "raw_output": "Yes, Tier-1 Support can immediately issue a courtesy refund through the billing console to satisfy the customer.",
        "obs": "Model authorized Tier-1 Support to issue direct refunds on ERR-5001 disputes, violating the strict prohibition requiring Fraud team approval."
    },
    {
        "trace_id": "tr_1009",
        "user_query": "If our MIGRATION_CREDIT is disputed, can support add a manual credit on top?",
        "chunk_id": "BM-003::t3",
        "score": 0.830,
        "raw_output": "Support will issue a manual ledger credit to balance the disputed amount while engineering investigates.",
        "obs": "Model instructed support to issue manual credits on disputed amounts, directly violating the double-credit prohibition in BM-003."
    },

    # --- Mode 3: Omits step-by-step fix when chunk splits mid-table (3 traces) ---
    {
        "trace_id": "tr_1010",
        "user_query": "How do I fix ERR-4010 malformed legacy invoice missing tax_code?",
        "chunk_id": "BM-002::t1",
        "score": 0.745,
        "raw_output": "ERR-4010 occurs when a legacy invoice is malformed. Please contact your system administrator.",
        "obs": "Model identified ERR-4010 as a malformed record but omitted the required `billing-repair --invoice <INV-ID>` command."
    },
    {
        "trace_id": "tr_1011",
        "user_query": "What command fixes ERR-5004 multi-currency settlement ledger desync?",
        "chunk_id": "BM-007::t1",
        "score": 0.720,
        "raw_output": "ERR-5004 indicates currency wallet desync during settlement. Escalation is required.",
        "obs": "Model described the ERR-5004 desync issue but omitted the specific remediation command `ledger-sync --dispute <DISPUTE-ID>`."
    },
    {
        "trace_id": "tr_1012",
        "user_query": "What is the fix when webhook signature verification fails with ERR-4040?",
        "chunk_id": "BM-002::t2",
        "score": 0.735,
        "raw_output": "ERR-4040 means the webhook secret rotated automatically during migration.",
        "obs": "Model stated the cause of ERR-4040 but omitted the fix instruction to rotate secret under Developer Settings."
    },

    # --- Mode 4: Confuses similar error codes or symptoms (2 traces) ---
    {
        "trace_id": "tr_1013",
        "user_query": "What should support do when subscription plan is not found in UBP catalogue showing ERR-4032?",
        "chunk_id": "BM-002::t2",
        "score": 0.785,
        "raw_output": "To resolve this, navigate to Admin → Account → Force Unlock to clear the concurrent session lock.",
        "obs": "Model returned the unlock procedure for ERR-4030 (login lock) instead of the custom plan creation procedure for ERR-4032."
    },
    {
        "trace_id": "tr_1014",
        "user_query": "Customer card failed with ERR-4001 during re-tokenisation. What is the cause?",
        "chunk_id": "BM-002::t1",
        "score": 0.810,
        "raw_output": "The card was reported lost or stolen by the issuing bank; customer must not retry.",
        "obs": "Model attributed ERR-4001 (expired card) to ERR-4003 (stolen card), wrongly advising that the card was stolen."
    },

    # --- Mode 5: False refusal on valid custom/enterprise queries (2 traces) ---
    {
        "trace_id": "tr_1015",
        "user_query": "How are custom subscription plan codes beginning with ENT- handled in migration?",
        "chunk_id": "BM-006::t0",
        "score": 0.650,
        "raw_output": "I do not have sufficient information in the provided documentation to answer questions regarding custom ENT- contracts.",
        "obs": "Model falsely refused to answer how ENT- plan codes are handled despite BM-006 explaining the custom plan recreation process."
    },
    {
        "trace_id": "tr_1016",
        "user_query": "Which SAML attribute must be configured in IdP for post-migration billing permissions?",
        "chunk_id": "BM-005::t1",
        "score": 0.640,
        "raw_output": "The documentation does not specify the SAML attribute required for Identity Provider authentication.",
        "obs": "Model issued a false refusal regarding SAML IdP configuration, missing the `ubp_billing_admin` attribute in BM-005."
    },

    # --- Mode 6: Clean Successes (4 traces) ---
    {
        "trace_id": "tr_1017",
        "user_query": "What HTTP header replaces X-Billing-Token in UBP API requests?",
        "chunk_id": "BM-004::t1",
        "score": 0.910,
        "raw_output": "UBP API requests authenticate using the standard Bearer token header `Authorization: Bearer <API-KEY>` in place of `X-Billing-Token` (BM-004).",
        "obs": "Model accurately identified the `Authorization: Bearer <API-KEY>` header and cited BM-004."
    },
    {
        "trace_id": "tr_1018",
        "user_query": "What is the escalation protocol when ERR-4020 credit overflow occurs?",
        "chunk_id": "BM-002::t3",
        "score": 0.895,
        "raw_output": "ERR-4020 is a P1 severity condition (credit exceeding $10,000) and must be escalated immediately to Billing Engineering on-call. Support must not issue a manual credit (BM-002).",
        "obs": "Model accurately stated the P1 severity escalation to Billing Engineering on-call with BM-002 cited."
    },
    {
        "trace_id": "tr_1019",
        "user_query": "How is ERR-4011 duplicate invoice handled during the migration window?",
        "chunk_id": "BM-002::t1",
        "score": 0.905,
        "raw_output": "For ERR-4011, Support voids the duplicate invoice via Billing Admin → Void Invoice. The refund is issued automatically (BM-002).",
        "obs": "Model gave the exact procedure to void duplicate invoices and noted automatic refund issuance under BM-002."
    },
    {
        "trace_id": "tr_1020",
        "user_query": "What is the procedure when we see an ERR-5001 chargeback notice?",
        "chunk_id": "BM-007::t1",
        "score": 0.920,
        "raw_output": "For ERR-5001, Tier-2 Support locks the invoice and routes the case to Risk & Fraud. Automated renewals are paused for 14 days, and Tier-1 Support must never issue a direct refund (BM-007).",
        "obs": "Model correctly detailed the 14-day renewal freeze, Tier-2 routing to Risk & Fraud, and the direct refund restriction citing BM-007."
    }
]

def generate_full_pool():
    all_traces = []
    # Expand to a full 60-trace pool repeating the 20 realistic trace scenarios
    for i in range(60):
        t_template = TRACES_POOL[i % len(TRACES_POOL)]
        tid = f"tr_{1001 + i}"
        record = {
            "trace_id": tid,
            "timestamp": f"2026-09-0{1 + (i // 10)}T{8 + (i % 14):02d}:{10 + (i * 7) % 50:02d}:00Z",
            "user_query": t_template["user_query"],
            "article_filters": None,
            "retrieved_chunks": [
                {
                    "chunk_id": t_template["chunk_id"],
                    "article_id": t_template["chunk_id"].split("::")[0],
                    "subject": "Billing Migration Reference Guide",
                    "score": t_template["score"],
                    "text": f"Documentation context snippet for {t_template['chunk_id']} related to {t_template['user_query']}."
                }
            ],
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "system_prompt": CHAT_SYSTEM_PROMPT,
            "model": "claude-3-5-sonnet-20241022",
            "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
            "raw_output": t_template["raw_output"],
            "open_coding_observation": t_template["obs"]
        }
        all_traces.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for t in all_traces:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(all_traces)} realistic production traces in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_full_pool()
