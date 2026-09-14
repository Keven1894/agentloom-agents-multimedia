# Cloud MVP: accounts, one free short video, then pay

**Date**: 2026-09-12
**Status**: 📋 Deferred — do not implement until we are ready to put the app online
**Trigger**: A customer can sign in and use MediaLoom the way the author uses it locally:
ingest one video, review evidence, keep decisions. Not a multi-reviewer workflow.
**Related**: local persistence lives in `reviews/`, `proposals/`, `docs/digests/`,
`data/transcripts/`, `data/jobs/`; search is SQLite (`medialoom.db`).

---

## 0. What this MVP is (and is not)

**Is:** a video-learning service. One account owns its videos. One person reviews a video.
Free tier: **one video, duration ≤ 5 minutes**. Anything else requires payment.

**Is not:** several people reviewing the same video, an org-wide knowledge graph, or a
search engine across other people's libraries.

Local file storage stays correct until this plan is executed. Do not migrate "just in case".

---

## 1. Storage split (already decided)

| What | Where in the cloud | Why |
|---|---|---|
| Accounts, plan, free-quota flag, Stripe ids | **Postgres (Supabase)** | Quota is a ledger. It needs a transaction. |
| Media rows (url, duration, status, owner) | Postgres | List "my videos", gate ingest. |
| Ingest job status | Postgres | Survive deploys; show history. |
| Review events (accept / reject / unsure) | Postgres | Same schema as `reviews/review-*.json`, as rows. |
| Transcript JSON, proposal, digest | **Supabase Storage or R2** | Large, write-once, pointer in the media row. |
| Per-user search index | Keep SQLite per account at first, or skip cross-user search | Not the billing path. |

Do **not** use Mongo. Do **not** put the 400-utterance transcript in Postgres as the source of
truth. Do **not** run the whole cloud app on a single SQLite file.

**Database choice:** Postgres, hosted as [Supabase](https://supabase.com) (Auth + DB + Storage
together). FastAPI stays the app.

---

## 2. Quota rule (write this before Stripe)

After `probe_media`, before ASR / embeddings:

1. Read `duration` from the probe.
2. If the account has `free_used = false` **and** `duration <= 300`: allow, then set
   `free_used = true` in the same transaction.
3. Otherwise: refuse with a paywall unless `plan` is paid (or they have remaining credits).
4. Probe is cheap; Whisper is not. Never start ASR on a blocked URL.

Lifetime one-free-video is enough for v1. Monthly reset is a later product decision.

---

## 3. Execution checklist (when deploying)

### A. Identity and tenancy

- [ ] Supabase project; email + Google login
- [ ] FastAPI verifies the JWT on every `/api/*` except login
- [ ] Every media / job / review row has `account_id`; list endpoints filter by it
- [ ] Row Level Security so a user cannot read another account's storage objects

### B. Schema (Postgres)

- [ ] `accounts (id, email, plan, free_used, stripe_customer_id, created_at)`
- [ ] `media (id, account_id, url, title, duration_s, status, transcript_uri, proposal_uri, digest_uri)`
- [ ] `ingest_jobs (id, account_id, media_id, status, stage, error, created_at, completed_at)`
- [ ] `review_events (id, account_id, media_id, kind, target_id, evidence_index, verdict, reviewer, created_at)`
- [ ] Unique: one free grant per account (`free_used` or a `quota_grants` table)

Local `record_decision` / `JobManager.persist` become adapters over these tables. The event
shape does not change.

### C. Paywall

- [ ] Stripe Checkout for the paid plan
- [ ] Webhook marks `accounts.plan = 'paid'`
- [ ] UI: free slot remaining / "this video is 12 minutes — pay to ingest"
- [ ] Hard block in the worker, not only in the button

### D. Runtime

- [ ] One web service (FastAPI) + one worker for ingest (same code as `JobManager.run_job`)
- [ ] Object prefix `accounts/{account_id}/media/{media_id}/...`
- [ ] Export a `review-*.json` snapshot only if we still want an AgentLoom git artifact
- [ ] Keep local file backend behind a store interface so `agentloom-media ui` still works offline

### E. Explicitly later

- [ ] Collaborative review of one video
- [ ] Cross-tenant search / `pgvector`
- [ ] Monthly quota reset, team seats, usage-based ASR billing
- [ ] Migrating historical local `reviews/` and DAOJIE proposals into a hosted account

---

## 4. Local work that is *not* blocked on this plan

Continue as now: files for proposals and transcripts, `reviews/*.json` for decisions,
`data/jobs/` for ingest history. Those maps 1:1 onto the tables above.

When this plan starts, the first code change is a `Store` interface around
`record_decision` and `JobManager`, not a rewrite of distillation.
