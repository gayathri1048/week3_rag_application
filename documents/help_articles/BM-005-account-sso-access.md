---
article_id: BM-005
product_area: Account
last_updated: 2025-11-12
tags: account, migration, sso, authentication, access
---
# Account Access and SSO During Billing Migration

The billing migration does not change your product login, but it does affect access to the **Billing Portal** and, for accounts that use SSO, the mapping between your identity provider and billing roles. This article covers the access-related changes and how to resolve them.

## Billing Portal Login Changes

Before migration, the billing portal was accessible at `billing.example.com/legacy` using your account email and a separate billing password. After migration, the portal moves to `billing.example.com` and uses your main product credentials (same email and password you use for the product itself).

There is a 48-hour overlap period where both portals are active so customers can retrieve any last invoices from the legacy portal.

## SSO Mapping After Migration

Enterprise accounts using SSO may experience a broken SSO mapping after cutover. The affected scenario is:

1. Your identity provider (IdP) sent a `billing_role` attribute in SAML assertions.
2. The Legacy Billing Engine stored this mapping in a separate role store.
3. The UBP reads billing roles from a different attribute (`ubp_billing_role`).

If the `ubp_billing_role` attribute is not present in the SAML assertion, the user authenticates but has no billing permissions. The error code for this state is `ERR-4031`.

### Fix for ERR-4031

| Step | Action |
|---|---|
| 1 | In your IdP, add the attribute `ubp_billing_role` to the SAML configuration |
| 2 | Set the value to one of: `billing_viewer`, `billing_editor`, `billing_admin` |
| 3 | In UBP Admin → SSO → Re-provision, click Re-provision for the affected user |
| 4 | Ask the user to log out and log back in — the new attribute is picked up immediately |

## Account Locked During Migration (ERR-4030)

Accounts that were logged into the billing portal during the migration cutover window (between 02:00 and 04:00 UTC on the migration date) may find their session in a locked state afterwards. This is because the migration tool acquires an account lock during data transfer.

To resolve ERR-4030:

1. Go to **Admin → Account → Force Unlock**.
2. Search for the account by email or account ID.
3. Click **Force Unlock**. The lock clears within 60 seconds.
4. Ask the customer to clear their browser cookies and log in again.

Note: the lock cannot be cleared by the customer themselves — it requires an admin action.

## Billing Role Reference

| Role | Billing Portal permissions |
|---|---|
| billing_viewer | View invoices, download PDFs, view payment methods (masked) |
| billing_editor | All viewer permissions + add/edit payment methods, update billing contact |
| billing_admin | All editor permissions + void invoices, issue credits, manage SSO mapping |

## Common Access Questions

**Q: Can a customer access their old legacy invoices after migration?**
Yes. Go to **Billing → Invoice History → Legacy Archive**. Legacy invoices are available in this archive until 2027-01-01.

**Q: Can the billing admin role be self-assigned?**
No. Billing admin must be granted by another billing admin or by Support via Admin → Account → Billing Roles.
