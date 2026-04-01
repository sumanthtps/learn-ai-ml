---
title: CDN
sidebar_position: 5
description: Content delivery network basics, edge caching, and delivery strategy.
---

# CDN

A CDN serves content from geographically distributed edge locations.

## Why CDNs exist

- reduce latency
- lower origin server load
- absorb global traffic more efficiently

## Best use cases

- images
- CSS
- JavaScript bundles
- videos
- downloadable files

## Basic flow

```mermaid
flowchart LR
  User --> Edge[CDN Edge]
  Edge --> Origin[Origin Server]
```

If the edge already has the file, it can respond immediately.

## Important headers

- `Cache-Control`
- `ETag`
- `Last-Modified`

## Common concepts

- edge node
- origin pull
- cache purge
- signed URLs

## Example

If a homepage loads:

- hero image
- app JS bundle
- fonts

those should usually come through a CDN.

## Common mistakes

- routing dynamic personalized responses through CDN without care
- forgetting cache invalidation after deployment
- ignoring asset versioning

## Quick revision

- CDN is mostly about distribution and edge caching
- static assets are the easiest wins
- versioned asset names simplify invalidation
