#!/usr/bin/env python3
"""Generate realistic production customer support RAG traces for Week 5 / Week 6 Error Analysis."""

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

SAMPLE_TRACES = [
    {
        "trace_id": "tr_1001",
        "timestamp": "2026-09-01T08:14:22Z",
        "user_query": "I am getting error code ERR-4032 during subscription migration, what is the fix?",
        "article_filters": {"product_area": "Billing"},
        "retrieved_chunks": [
            {
                "chunk_id": "BM-002::t3",
                "article_id": "BM-002",
                "subject": "Billing Migration Error Codes — Reference Guide",
                "score": 0.892,
                "text": "ERR-4032 | Account | Subscription plan not found in UBP catalogue — legacy custom plan has no UBP equivalent | Support must manually create a matching plan in UBP Admin → Plans → Create Custom, then migrate the subscription to it"
            },
            {
                "chunk_id": "BM-006::t0",
                "article_id": "BM-006",
                "subject": "Enterprise Subscription Migration — Custom Plans and Contracts",
                "score": 0.764,
                "text": "Custom plans do not migrate automatically. Each custom plan must be recreated in the UBP catalogue before the account's migration date."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "For error code ERR-4032 (Subscription plan not found in UBP catalogue), the legacy custom plan does not have an automated UBP equivalent. To fix this, Support must manually create a matching plan in UBP Admin → Plans → Create Custom, and then migrate the subscription to it. (Reference: BM-002)."
    },
    {
        "trace_id": "tr_1002",
        "timestamp": "2026-09-01T08:45:10Z",
        "user_query": "What is the expiration timeframe for unused migration credits from the old billing engine?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-001::t3",
                "article_id": "BM-001",
                "subject": "Billing Migration Overview — What Changes and When",
                "score": 0.812,
                "text": "Download any invoices you need for tax purposes before 2026-03-01 — after that date, legacy PDF format will be retired in favour of UBP format. Prorated credits from the old system are converted at a 1:1 rate."
            },
            {
                "chunk_id": "BM-003::t2",
                "article_id": "BM-003",
                "subject": "Invoice and Credit Migration — How Your Balance Transfers",
                "score": 0.795,
                "text": "Migration credits expire 12 months from the migration date. Credits that are unused at expiry are forfeited and do not convert to cash refunds. This policy is non-negotiable."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "You must use your credits before 2026-03-01, as legacy formats and credits expire on that date."
    },
    {
        "trace_id": "tr_1003",
        "timestamp": "2026-09-01T09:12:05Z",
        "user_query": "Customer received ERR-4001 when card re-tokenisation failed. What action is required?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-002::t1",
                "article_id": "BM-002",
                "subject": "Billing Migration Error Codes — Reference Guide",
                "score": 0.915,
                "text": "ERR-4001 | Payment | Saved card token could not be re-tokenised — card expired | Customer must add a new payment method at billing.example.com"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "When ERR-4001 occurs, it indicates that the saved card token could not be re-tokenised because the card has expired. The required action is for the customer to add a new payment method at billing.example.com (BM-002)."
    },
    {
        "trace_id": "tr_1004",
        "timestamp": "2026-09-01T09:30:44Z",
        "user_query": "Can support extend my migration credit expiration deadline by 6 months?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-003::t2",
                "article_id": "BM-003",
                "subject": "Invoice and Credit Migration — How Your Balance Transfers",
                "score": 0.840,
                "text": "Migration credits expire 12 months from the migration date. Credits that are unused at expiry are forfeited and do not convert to cash refunds. This policy is non-negotiable and cannot be extended by Support — escalations on this topic should be directed to the customer's Account Executive."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "Yes, our support team can issue a temporary extension if you open a ticket with Billing Operations."
    },
    {
        "trace_id": "tr_1005",
        "timestamp": "2026-09-01T10:05:18Z",
        "user_query": "What happens to draft invoices during the cutover window?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-003::t1",
                "article_id": "BM-003",
                "subject": "Invoice and Credit Migration — How Your Balance Transfers",
                "score": 0.880,
                "text": "| DRAFT | Discarded — draft invoices are not migrated | N/A |"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "Draft invoices are not migrated to UBP. According to article BM-003, draft invoices are discarded during the cutover."
    },
    {
        "trace_id": "tr_1006",
        "timestamp": "2026-09-01T10:40:50Z",
        "user_query": "Account locked during cutover window showing ERR-4030, how can admin unlock it?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-002::t2",
                "article_id": "BM-002",
                "subject": "Billing Migration Error Codes — Reference Guide",
                "score": 0.865,
                "text": "ERR-4030 | Account | Account locked during migration — concurrent login during cutover window | Unlock via Admin → Account → Force Unlock; session clears automatically"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "To unlock an account displaying ERR-4030, navigate to Admin → Account → Force Unlock. The concurrent login session clears automatically (BM-002)."
    },
    {
        "trace_id": "tr_1007",
        "timestamp": "2026-09-01T11:15:32Z",
        "user_query": "We are on an enterprise custom plan with code ENT-ALPHA-99. How is our migration handled?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-006::t0",
                "article_id": "BM-006",
                "subject": "Enterprise Subscription Migration — Custom Plans and Contracts",
                "score": 0.820,
                "text": "Custom plans are identified in the Legacy Billing Engine by a plan code that begins with CUSTOM- or ENT-. Custom plans do not migrate automatically. Each custom plan must be recreated in the UBP catalogue before the account's migration date."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "Plans beginning with ENT- are custom plans and do not migrate automatically. They must be recreated in the UBP catalogue prior to migration, and enterprise customers must sign a contract addendum (BM-006)."
    },
    {
        "trace_id": "tr_1008",
        "timestamp": "2026-09-01T11:45:00Z",
        "user_query": "How do I fix ERR-4010 on a malformed legacy invoice?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-002::t1",
                "article_id": "BM-002",
                "subject": "Billing Migration Error Codes — Reference Guide",
                "score": 0.875,
                "text": "ERR-4010 | Invoice | Legacy invoice record is malformed — missing required `tax_code` field | Support runs `billing-repair --invoice <INV-ID>` to backfill the field"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "For ERR-4010 (missing tax_code field), Support must run the command `billing-repair --invoice <INV-ID>` to backfill the required field (BM-002)."
    },
    {
        "trace_id": "tr_1009",
        "timestamp": "2026-09-01T12:20:14Z",
        "user_query": "What should we do when webhook signature verification fails with ERR-4040?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-002::t2",
                "article_id": "BM-002",
                "subject": "Billing Migration Error Codes — Reference Guide",
                "score": 0.810,
                "text": "ERR-4040 | Webhook | Webhook secret rotated automatically during migration; old secret now invalid | Customer regenerates webhook secret in Developer Settings → Webhooks → Rotate Secret"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "When ERR-4040 occurs, the webhook secret was rotated during migration. The customer must regenerate the webhook secret in Developer Settings → Webhooks → Rotate Secret (BM-002)."
    },
    {
        "trace_id": "tr_1010",
        "timestamp": "2026-09-01T13:00:22Z",
        "user_query": "Which SAML attribute must be added to IdP configuration to fix post-migration login?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-005::t1",
                "article_id": "BM-005",
                "subject": "Account SSO and Permissions Post-Cutover",
                "score": 0.780,
                "text": "SAML configuration requires mapping the `ubp_billing_admin` attribute in the Identity Provider to grant administrative billing rights."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "The IdP configuration requires adding the `ubp_billing_admin` attribute mapping to restore post-migration permissions (BM-005)."
    },
    {
        "trace_id": "tr_1011",
        "timestamp": "2026-09-01T13:35:19Z",
        "user_query": "What is the procedure when we see ERR-5001 chargeback notice?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-007::t1",
                "article_id": "BM-007",
                "subject": "Billing Dispute and Chargeback Resolution Guide",
                "score": 0.890,
                "text": "ERR-5001 | Dispute | Bank chargeback initiated by cardholder | Tier-2 Support locks invoice and routes case to Risk & Fraud team"
            },
            {
                "chunk_id": "BM-007::t2",
                "article_id": "BM-007",
                "subject": "Billing Dispute and Chargeback Resolution Guide",
                "score": 0.850,
                "text": "1. When ERR-5001 appears, automated renewal is paused for 14 business days. 2. Customer has 7 days to submit proof at billing.example.com/disputes. Tier-1 Support must never refund directly."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "For ERR-5001, Tier-2 Support locks the invoice and routes the case to Risk & Fraud. Automated renewal is paused for 14 days and Tier-1 Support must never issue a direct refund (BM-007)."
    },
    {
        "trace_id": "tr_1012",
        "timestamp": "2026-09-01T14:10:04Z",
        "user_query": "Can we refund a disputed charge under ERR-5001 directly to appease the customer?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-007::t2",
                "article_id": "BM-007",
                "subject": "Billing Dispute and Chargeback Resolution Guide",
                "score": 0.880,
                "text": "Tier-1 Support must never refund an invoice displaying ERR-5001 directly without Fraud Team approval."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "Yes, you can issue an instant courtesy refund through the billing console."
    },
    {
        "trace_id": "tr_1013",
        "timestamp": "2026-09-01T14:45:30Z",
        "user_query": "When will Phase 2 Business migration begin?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-001::t1",
                "article_id": "BM-001",
                "subject": "Billing Migration Overview — What Changes and When",
                "score": 0.920,
                "text": "| Phase 2 — Business | 2025-12-01 | All Business and Pro Plan accounts | Verify payment method |"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "Phase 2 (Business and Pro Plan accounts) begins on 2025-12-01. Customers should verify their payment methods prior to this date (BM-001)."
    },
    {
        "trace_id": "tr_1014",
        "timestamp": "2026-09-01T15:20:18Z",
        "user_query": "Where do I look up the legacy invoice format after cutover?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-001::t2",
                "article_id": "BM-001",
                "subject": "Billing Migration Overview — What Changes and When",
                "score": 0.775,
                "text": "All historical invoices remain accessible under Billing → Invoice History → Legacy Archive. Invoice format changes from INV-YYMMDD-NNNN to UBP-YYYY-NNNNNN."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "Historical legacy invoices can be accessed in the portal under Billing → Invoice History → Legacy Archive (BM-001)."
    },
    {
        "trace_id": "tr_1015",
        "timestamp": "2026-09-01T15:55:40Z",
        "user_query": "What should support do when ERR-4020 credit overflow occurs?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-002::t1",
                "article_id": "BM-002",
                "subject": "Billing Migration Error Codes — Reference Guide",
                "score": 0.860,
                "text": "ERR-4020 | Credit | MIGRATION_CREDIT calculation overflow — credit balance exceeded $10,000 | Escalate to Billing Engineering; do not issue manual credit"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "For ERR-4020 (credit overflow exceeding $10,000), escalate immediately to Billing Engineering on-call. Do not issue a manual credit (BM-002)."
    },
    {
        "trace_id": "tr_1016",
        "timestamp": "2026-09-01T16:30:12Z",
        "user_query": "How do I fix ERR-5002 partial refund failure?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-007::t1",
                "article_id": "BM-007",
                "subject": "Billing Dispute and Chargeback Resolution Guide",
                "score": 0.885,
                "text": "ERR-5002 | Refund | Partial refund failed due to settled batch closure | Support issues a manual ledger adjustment via Billing Admin → Refunds → Force Credit"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "To fix ERR-5002 (partial refund failed due to batch closure), Support must issue a manual ledger adjustment via Billing Admin → Refunds → Force Credit (BM-007)."
    },
    {
        "trace_id": "tr_1017",
        "timestamp": "2026-09-01T17:05:00Z",
        "user_query": "What is the command to sync currency wallets for ERR-5004?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-007::t1",
                "article_id": "BM-007",
                "subject": "Billing Dispute and Chargeback Resolution Guide",
                "score": 0.895,
                "text": "ERR-5004 | Currency | Multi-currency settlement ledger desync | Run command `ledger-sync --dispute <DISPUTE-ID>` to re-balance currency wallets"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "The command to re-balance multi-currency wallets for ERR-5004 is `ledger-sync --dispute <DISPUTE-ID>` (BM-007)."
    },
    {
        "trace_id": "tr_1018",
        "timestamp": "2026-09-01T17:40:22Z",
        "user_query": "Can custom contract accounts negotiate 50% discount on UBP migration?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-006::t0",
                "article_id": "BM-006",
                "subject": "Enterprise Subscription Migration — Custom Plans and Contracts",
                "score": 0.690,
                "text": "Pricing does not change during migration. Subscription renewal dates do not change. Custom plans must be recreated in the UBP catalogue."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "Yes, all enterprise accounts receive an automatic 50% discount coupon upon completing migration."
    },
    {
        "trace_id": "tr_1019",
        "timestamp": "2026-09-01T18:15:45Z",
        "user_query": "How is ERR-4011 duplicate invoice handled during migration?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-002::t1",
                "article_id": "BM-002",
                "subject": "Billing Migration Error Codes — Reference Guide",
                "score": 0.890,
                "text": "ERR-4011 | Invoice | Duplicate invoice detected — same period billed twice in migration window | Support voids the duplicate via Billing Admin → Void Invoice; refund issued automatically"
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "For ERR-4011 (duplicate invoice), Support voids the duplicate via Billing Admin → Void Invoice. The refund is issued automatically (BM-002)."
    },
    {
        "trace_id": "tr_1020",
        "timestamp": "2026-09-01T18:50:11Z",
        "user_query": "What HTTP header replaces X-Billing-Token in UBP API?",
        "article_filters": None,
        "retrieved_chunks": [
            {
                "chunk_id": "BM-004::t1",
                "article_id": "BM-004",
                "subject": "Webhook and API Integration Migration Changes",
                "score": 0.870,
                "text": "UBP API requests authenticate using the standard Bearer token header `Authorization: Bearer <API-KEY>` instead of the legacy `X-Billing-Token` header."
            }
        ],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt": CHAT_SYSTEM_PROMPT,
        "model": "claude-3-5-sonnet-20241022",
        "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
        "raw_output": "In the UBP API, authentication requires `Authorization: Bearer <API-KEY>` in place of `X-Billing-Token` (BM-004)."
    }
]

# Generate a pool of 60 distinct traces with varied realistic queries
def generate_all_traces():
    # Expand to 60 distinct realistic traces
    all_traces = []
    base_id = 1000
    
    variations = [
        # Query 1-5: Error codes & resolutions
        ("I am seeing ERR-4032 when attempting plan cutover, what should support do?", "BM-002::t3", "ERR-4032: Support must manually create a matching plan in UBP Admin → Plans → Create Custom, then migrate the subscription (BM-002).", False),
        ("What does ERR-4001 mean for our customer card?", "BM-002::t1", "ERR-4001 means saved card token could not be re-tokenised because card expired. Customer must add new payment method (BM-002).", False),
        ("How to fix ERR-4010 tax_code missing error?", "BM-002::t1", "Support runs `billing-repair --invoice <INV-ID>` to backfill the missing tax_code (BM-002).", False),
        ("Webhook failed with ERR-4040, what is the fix?", "BM-002::t2", "Customer regenerates webhook secret in Developer Settings → Webhooks → Rotate Secret (BM-002).", False),
        ("Account is locked with ERR-4030 during migration cutover.", "BM-002::t2", "Admin navigates to Admin → Account → Force Unlock to clear session (BM-002).", False),
        
        # Query 6-10: Policy & Superseded rules
        ("When do migration credits expire?", "BM-003::t2", "Migration credits expire 12 months from migration date and are non-refundable (BM-003).", False),
        ("Can support extend migration credits beyond 12 months?", "BM-003::t2", "No, Support cannot extend credit expiration; escalations must go to Account Executive (BM-003).", False),
        ("Are legacy PDF invoices available after 2026-03-01?", "BM-001::t3", "No, legacy PDF format will be retired after 2026-03-01 in favour of UBP format (BM-001).", False),
        ("What is the exchange rate used for non-USD migration credits?", "BM-003::t2", "Non-USD credits use the ECB reference rate published on the migration date (BM-003).", False),
        ("What happens to draft invoices during migration?", "BM-003::t1", "Draft invoices are discarded and not migrated to UBP (BM-003).", False),

        # Query 11-15: Enterprise & Custom
        ("How are custom plans with code ENT-GOLD-01 migrated?", "BM-006::t0", "Custom plans with ENT- or CUSTOM- codes do not migrate automatically; they must be recreated in UBP catalogue (BM-006).", False),
        ("Do we get a 50% discount for enterprise migration?", "BM-006::t0", "No, pricing does not change during migration (BM-006).", False),
        ("Which SAML attribute gives billing admin access?", "BM-005::t1", "The `ubp_billing_admin` SAML attribute mapping must be added to IdP configuration (BM-005).", False),
        ("What header is required for UBP API?", "BM-004::t1", "Use `Authorization: Bearer <API-KEY>` header (BM-004).", False),
        ("What is the escalation for ERR-4003 stolen card?", "BM-002::t3", "ERR-4003 is a P1 condition and must be escalated to Billing Engineering on-call (BM-002).", False),

        # Query 16-20: Disputes & Edge cases
        ("What is the protocol for ERR-5001 dispute notice?", "BM-007::t1", "Tier-2 locks invoice and routes to Risk & Fraud; automated renewal is paused 14 days; Tier-1 must never refund directly (BM-007).", False),
        ("Can support issue direct refund on ERR-5001?", "BM-007::t2", "Tier-1 Support must never refund an invoice displaying ERR-5001 directly (BM-007).", False),
        ("What command fixes ERR-5004 multi-currency desync?", "BM-007::t1", "Run `ledger-sync --dispute <DISPUTE-ID>` (BM-007).", False),
        ("How is ERR-5002 partial refund failure resolved?", "BM-007::t1", "Support issues a manual ledger adjustment via Billing Admin → Refunds → Force Credit (BM-007).", False),
        ("What is the escalation for ERR-4020 credit overflow?", "BM-002::t3", "ERR-4020 is a P1 condition requiring immediate escalation to Billing Engineering on-call (BM-002).", False)
    ]

    # Repeat to make 60 realistic traces
    for i in range(60):
        tid = f"tr_{base_id + i + 1}"
        query_text, chunk_id, expected_ans, is_demo = variations[i % len(variations)]
        
        # Add realistic behavior to outputs
        # 1. Superseded policy failure (e.g. citing 2026-03-01 PDF date as credit expiry)
        if i in [1, 21, 41]:
            raw_out = "Migration credits expire on 2026-03-01 along with legacy PDF format retirement."
        # 2. Hallucinated courtesy refund / discount policy
        elif i in [3, 11, 16, 23, 31, 36]:
            raw_out = "Yes, support can issue a 50% courtesy credit or grant a 6-month extension upon request."
        # 3. Step-by-step fix omitted due to chunk truncation
        elif i in [5, 25, 45]:
            raw_out = "ERR-4030 indicates a concurrent login lockout during migration. Please contact your system administrator."
        # 4. Cross-code confusion (confusing ERR-4001 with ERR-4003)
        elif i in [7, 27, 47]:
            raw_out = "When card re-tokenisation fails, the card was reported stolen by issuing bank. Do not retry."
        # 5. False refusal on valid query
        elif i in [9, 29, 49]:
            raw_out = "I do not have sufficient information in the documentation to answer your question regarding custom enterprise plan contracts."
        # 6. Correct answers
        else:
            raw_out = expected_ans

        trace_record = {
            "trace_id": tid,
            "timestamp": f"2026-09-0{1 + (i // 10)}T{8 + (i % 14):02d}:{10 + (i * 7) % 50:02d}:00Z",
            "user_query": query_text,
            "article_filters": None,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk_id,
                    "article_id": chunk_id.split("::")[0],
                    "subject": "Billing Migration Documentation",
                    "score": round(0.75 + ((i * 3) % 20) / 100.0, 3),
                    "text": f"Documentation snippet for {chunk_id}: {expected_ans}"
                }
            ],
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "system_prompt": CHAT_SYSTEM_PROMPT,
            "model": "claude-3-5-sonnet-20241022",
            "model_params": {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0},
            "raw_output": raw_out
        }
        all_traces.append(trace_record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for t in all_traces:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(all_traces)} traces into {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_all_traces()
