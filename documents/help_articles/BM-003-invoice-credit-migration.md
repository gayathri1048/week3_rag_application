---
article_id: BM-003
product_area: Payments
last_updated: 2025-11-08
tags: payments, migration, invoices, credits
---
# Invoice and Credit Migration — How Your Balance Transfers

This article explains how outstanding invoices, prorated credits, and account balances are transferred from the Legacy Billing Engine (LBE) to the Unified Billing Platform (UBP).

## Invoice Numbering Changes

Legacy invoices use the format `INV-YYMMDD-NNNN` (e.g. `INV-251101-0042`). After migration, all new invoices use the UBP format `UBP-YYYY-NNNNNN` (e.g. `UBP-2025-000042`). Old invoice numbers remain valid for lookups and remain searchable in **Billing → Invoice History → Legacy Archive**.

## How Open Invoices Are Handled

Any invoice with status `OPEN` or `OVERDUE` at migration time is copied into UBP with its balance intact. The following rules apply:

| Invoice Status at Migration | Action in UBP | Due Date |
|---|---|---|
| OPEN (not yet due) | Copied as-is; original due date preserved | Unchanged |
| OPEN (past due) | Copied and flagged `OVERDUE`; collections hold placed | Unchanged |
| DRAFT | Discarded — draft invoices are not migrated | N/A |
| PAID | Archived in Legacy Archive only; not copied to UBP ledger | N/A |
| VOID | Not migrated | N/A |

## Migration Credits

Every account that had an active prorated credit on the LBE receives a `MIGRATION_CREDIT` line item on their first UBP invoice. The credit is calculated as follows:

```
MIGRATION_CREDIT = LBE_credit_balance × exchange_rate_on_migration_date
```

Credits in non-USD currencies are converted using the ECB reference rate published on the migration date. The credit is non-refundable; it may only be applied against future invoices.

## Credit Expiry Policy

Migration credits expire 12 months from the migration date. Credits that are unused at expiry are forfeited and do not convert to cash refunds. This policy is non-negotiable and cannot be extended by Support — escalations on this topic should be directed to the customer's Account Executive.

## Disputing a Migration Invoice

If a customer believes their `MIGRATION_CREDIT` amount is wrong:

1. Ask them to navigate to **Billing → Invoice → View MIGRATION_CREDIT detail**.
2. The detail view shows the LBE balance, exchange rate, and conversion date.
3. If the LBE balance shown is wrong, open a case with Billing Engineering (not Tier-1 Support).
4. If the exchange rate is wrong (more than 0.5% from the published ECB rate), open a case with Billing Operations.

Do NOT issue a manual credit to compensate — manual credits on top of a disputed `MIGRATION_CREDIT` create double-credit entries that require a full ledger reconciliation.
