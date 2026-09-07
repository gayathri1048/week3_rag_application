---
article_id: BM-007
product_area: Disputes
last_updated: 2026-02-15
tags: billing, disputes, chargebacks, error-codes
---
# Billing Dispute and Chargeback Resolution Guide

This article provides the authoritative escalation workflow for handling customer payment disputes, chargeback notices, and merchant fraud inquiries during and after the Unified Billing Platform (UBP) migration.

## Dispute Error Codes Reference

| Error Code | Category | Root Cause | Required Action |
|---|---|---|---|
| ERR-5001 | Dispute | Bank chargeback initiated by cardholder | Tier-2 Support locks invoice and routes case to Risk & Fraud team |
| ERR-5002 | Refund | Partial refund failed due to settled batch closure | Support issues a manual ledger adjustment via Billing Admin → Refunds → Force Credit |
| ERR-5003 | Tax | Regional VAT/GST mismatch on multi-entity accounts | Customer must upload a valid VAT exemption certificate under Account → Tax Profile |
| ERR-5004 | Currency | Multi-currency settlement ledger desync | Run command `ledger-sync --dispute <DISPUTE-ID>` to re-balance currency wallets |

## Escalation Protocol for ERR-5001

1. When `ERR-5001` appears in the customer audit log, the account's automated renewal is paused immediately for 14 business days.
2. The customer has 7 days to submit proof of authorization via the customer portal at `billing.example.com/disputes`.
3. If no proof is received within 7 days, Billing Operations automatically flags the account for manual review by Legal.
4. Tier-1 Support must **never** refund an invoice displaying `ERR-5001` directly without Fraud Team approval.
