---
article_id: BM-001
product_area: Billing
last_updated: 2025-11-01
tags: billing, migration, overview, legacy
---
# Billing Migration Overview — What Changes and When

Starting 1 December 2025, all accounts will be migrated from the Legacy Billing Engine (LBE) to the Unified Billing Platform (UBP). This article explains the timeline, what changes, and what you need to do before the cutover date.

## Why We Are Migrating

The Legacy Billing Engine was built in 2018 and does not support multi-currency invoicing, usage-based pricing tiers, or real-time spend alerts. The Unified Billing Platform replaces all of that with a single ledger and a new invoice format.

## Migration Timeline

| Phase | Date | Accounts affected | Action required |
|---|---|---|---|
| Phase 1 — Pilot | 2025-11-15 | Accounts created before 2020-01-01 | None — automatic |
| Phase 2 — Business | 2025-12-01 | All Business and Pro Plan accounts | Verify payment method |
| Phase 3 — Enterprise | 2026-01-10 | Enterprise and custom-contract accounts | Sign addendum |
| Phase 4 — Legacy Free | 2026-02-01 | Remaining free-tier accounts | None — automatic |

## What Changes for Customers

- Invoice numbering format changes from `INV-YYMMDD-NNNN` to `UBP-YYYY-NNNNNN`.
- The billing portal URL changes from `billing.example.com/legacy` to `billing.example.com`.
- Saved payment methods are migrated automatically; tokens are re-tokenised server-side with no card re-entry required.
- Prorated credits from the old system are converted at a 1:1 rate and appear as a `MIGRATION_CREDIT` line item on the first UBP invoice.

## What Stays the Same

Pricing does not change. Subscription renewal dates do not change. All historical invoices remain accessible under **Billing → Invoice History → Legacy Archive**.

## Common Pre-Migration Checklist

1. Confirm your billing email address is up to date under **Account → Settings → Billing Contact**.
2. Check that your saved payment method has not expired.
3. Download any invoices you need for tax purposes before 2026-03-01 — after that date, legacy PDF format will be retired in favour of UBP format.
