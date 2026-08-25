---
article_id: BM-004
product_area: Payments
last_updated: 2025-11-10
tags: payments, migration, webhooks, api, developer
---
# Developer Guide — Webhook and API Changes During Billing Migration

This article is for customers who use the Billing API or receive billing webhooks. The migration to the Unified Billing Platform (UBP) introduces breaking changes to event names, payload shapes, and authentication secrets.

## Breaking Changes Summary

| Area | Legacy Billing Engine | Unified Billing Platform | Action required |
|---|---|---|---|
| Webhook secret | Per-endpoint static secret | Rotated automatically on migration day | Regenerate in Developer Settings → Webhooks → Rotate Secret |
| Event name: invoice created | `invoice.created` | `ubp.invoice.created` | Update event listener |
| Event name: payment succeeded | `payment.success` | `ubp.payment.succeeded` | Update event listener |
| Event name: payment failed | `payment.failure` | `ubp.payment.failed` | Update event listener |
| Invoice ID field | `invoice_id` (string) | `ubp_invoice_id` (string) + `legacy_invoice_id` (nullable string) | Update field references |
| Amount field | `amount` (integer, cents) | `amount_cents` (integer) + `amount_display` (string with currency symbol) | Update field references |
| Auth header | `X-Billing-Token` | `Authorization: Bearer <token>` | Update API calls |

## Webhook Secret Rotation

The webhook secret is rotated automatically at the start of your account's migration window. The new secret is shown once in **Developer Settings → Webhooks → Rotate Secret**. If you miss it:

1. Go to Developer Settings → Webhooks.
2. Click **Rotate Secret** on any affected endpoint.
3. Copy and store the new secret immediately — it is shown only once.
4. Update your webhook handler's signature verification logic.

If your endpoint fails signature verification after migration, you will see error `ERR-4040` in the Migration Audit Log.

## API Authentication Change

The Legacy Billing Engine accepted a static `X-Billing-Token` header. The UBP uses standard Bearer tokens issued via OAuth 2.0. To obtain a Bearer token:

```bash
curl -X POST https://auth.example.com/oauth/token \
  -d grant_type=client_credentials \
  -d client_id=YOUR_CLIENT_ID \
  -d client_secret=YOUR_CLIENT_SECRET
```

The response contains `access_token` (valid for 1 hour) and `expires_in`. Rotate tokens proactively — the token endpoint rate-limits at 60 requests per minute per client ID.

## Replay and Retry Behaviour

| Scenario | UBP Behaviour |
|---|---|
| Endpoint returns 2xx | Event marked delivered; no retry |
| Endpoint returns 4xx | Event marked failed; no retry (client error) |
| Endpoint returns 5xx | Retry with exponential back-off: 30 s, 2 min, 10 min, 1 h, 6 h, then dead-letter |
| Endpoint unreachable (timeout) | Same as 5xx retry schedule |
| Dead-letter queue age | 7 days; Support can trigger manual replay via Admin → Webhooks → Replay |

## Testing Against the UBP Sandbox

A sandbox environment is available at `https://billing-sandbox.example.com`. Sandbox events use the prefix `ubp.sandbox.*`. Sandbox webhook deliveries do not count against your rate limits.
