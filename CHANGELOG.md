# Changelog – cachly SDK (go)

**Language:** Go  
**Package:** `github.com/cachly-dev/sdk-go` on **pkg.go.dev**

> Full cross-SDK release notes: [../CHANGELOG.md](../CHANGELOG.md)

---

## [0.2.0] – 2026-04-07

### Added

- **`MSet(ctx, []MSetItem)`** – bulk set with per-key TTL via Redis pipeline
- **`MGet(ctx, []string)`** – bulk get via native `MGET`; returns `[]any` (nil on miss)
- **`Lock(ctx, key, LockOptions)`** – distributed lock (SetNX + Lua release)
  - Returns `*LockHandle`; `nil` when retries exhausted
  - `LockHandle.Release(ctx)` is idempotent via `sync.Once`
- **`StreamSet(ctx, key, chan string, StreamSetOptions)`** – cache token stream via RPUSH
- **`StreamGet(ctx, key)`** – returns `*StreamReader`; iterate with `Next(ctx)`

### Types added

- `MSetItem{Key, Value, TTL}` – per-item bulk payload
- `LockOptions{TTL, Retries, RetryDelay}` – lock configuration
- `LockHandle{Token}` – fencing token + release
- `StreamReader{Len()}` – sequential chunk access

---

## [0.1.0-beta.1] – 2026-04-07

Initial beta release.

### Added

- `Set` / `Get` / `Delete` / `Exists` / `Expire` / `TTL` / `Incr` / `IncrBy`
- `GetOrSet(ctx, key, fn, ttl)` – read-through pattern
- **Semantic cache:** `Semantic(embedFn)` / `SemanticWithVector(embedFn, vectorURL)`
  - `GetOrSet`, `Warmup`, `Flush`, `Size`, `Entries`, `Invalidate`
  - §4 `DetectNamespace`, §7 `QuantizeEmbedding` / `DequantizeEmbedding`
  - §3 Hybrid BM25+Vector RRF, §1 `AdaptiveThreshold` / `Feedback`
- TLS by default, EU data residency

### Previously known limitations – resolved in v0.2.0

- ~~Bulk operations (`MSet` / `MGet`) not yet implemented~~ ✅

---

## [Unreleased]

See [../CHANGELOG.md](../CHANGELOG.md) for upcoming features.

