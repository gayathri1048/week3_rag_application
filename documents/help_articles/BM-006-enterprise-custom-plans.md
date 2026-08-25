---
article_id: BM-006
product_area: Billing
last_updated: 2025-11-15
tags: billing, migration, enterprise, custom-plans, subscriptions
---
# Enterprise Subscription Migration — Custom Plans and Contracts

Enterprise accounts with custom pricing plans or non-standard contract terms require additional steps during the billing migration. This article covers how custom plans are handled, what the addendum process looks like, and the error codes you may encounter.

## What Is a Custom Plan?

A custom plan is any subscription that was negotiated outside the standard catalogue — for example, a volume-discount rate, a multi-year prepay at a fixed price, or a plan that bundles products not offered in the public catalogue.

Custom plans are identified in the Legacy Billing Engine by a plan code that begins with `CUSTOM-` or `ENT-`.

## How Custom Plans Are Migrated

Custom plans do not migrate automatically. Each custom plan must be recreated in the UBP catalogue before the account's migration date. The process is:

| Step | Owner | Action |
|---|---|---|
| 1 | Billing Operations | Identify all custom-plan accounts (report run 30 days before Phase 3) |
| 2 | Account Executive | Notify the customer and send the migration addendum for signature |
| 3 | Billing Engineering | Recreate the plan in UBP Admin → Plans → Create Custom |
| 4 | Support | Migrate the subscription to the new UBP plan via Admin → Subscriptions → Migrate |
| 5 | Customer | Verify the plan details in the new billing portal |

If Step 3 or 4 is not completed before the migration date, the account will encounter `ERR-4032` when the migration tool attempts to copy the subscription.

## ERR-4032 Resolution Steps

ERR-4032 means: "Subscription plan not found in UBP catalogue — legacy custom plan has no UBP equivalent."

| Sub-step | Action |
|---|---|
| a | Confirm the legacy plan code by checking Admin → Subscriptions → View Legacy Details |
| b | Contact Billing Operations to confirm whether a UBP equivalent plan exists or needs to be created |
| c | Once the UBP plan is created, go to UBP Admin → Plans → Create Custom if it does not already exist |
| d | Go to Admin → Subscriptions → Migrate and select the UBP plan as the target |
| e | Confirm migration — the subscription moves immediately; billing resumes on the next renewal date |

Do NOT modify the subscription price manually to work around a missing plan — this bypasses contract enforcement and will be flagged in the quarterly audit.

## Migration Addendum

Enterprise accounts must sign a migration addendum before Phase 3 (cutover on 2026-01-10). The addendum confirms:

- The customer's plan terms are preserved in the UBP.
- Any volume commitments carry over.
- The migration date and any grace period agreed with the Account Executive.

Addendums that are not signed 7 days before the migration date trigger an automatic extension of the Phase 3 date by 14 days. After two extensions, the account is flagged for manual review by Legal.

## Custom Plan Pricing Verification

After migration, the customer should verify the following on their first UBP invoice:

| Item | Where to check | Expected value |
|---|---|---|
| Base plan price | Invoice line item: Plan fee | Same as legacy contract |
| Volume discount | Invoice line item: Volume discount | Same percentage as legacy |
| Prepay credit | Invoice line item: Prepay credit | Remaining balance from legacy prepay |
| Contract end date | Billing → Subscription → Contract details | Same as signed contract |

If any item does not match, the customer should contact their Account Executive — not Tier-1 Support. Pricing discrepancies on custom plans require a contract-level review.
