# cachly Go SDK

> Official Go SDK for [cachly.dev](https://cachly.dev) —  
> Managed Valkey/Redis cache built for AI apps. **GDPR-compliant · German servers · Live in 30 seconds.**

[![Go Reference](https://pkg.go.dev/badge/github.com/cachly-dev/sdk-go.svg)](https://pkg.go.dev/github.com/cachly-dev/sdk-go)
[![Go 1.21+](https://img.shields.io/badge/Go-1.21%2B-00ADD8?logo=go)](https://pkg.go.dev/github.com/cachly-dev/sdk-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)
[![GDPR: EU-only](https://img.shields.io/badge/GDPR-EU%20only-green)](https://cachly.dev/legal)

---

## Installation

```bash
go get github.com/cachly-dev/sdk-go
```

> Requires Go 1.21+. Uses `github.com/redis/go-redis/v9`.

---

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "os"
    "time"

    "github.com/cachly-dev/sdk-go/cachly"
)

func main() {
    client, err := cachly.New(os.Getenv("CACHLY_URL"))
    if err != nil {
        panic(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Set with TTL
    if err := client.Set(ctx, "user:42", map[string]string{"name": "Alice"}, 5*time.Minute); err != nil {
        panic(err)
    }

    // Get (auto-deserialises JSON); redis.Nil on miss
    var user map[string]string
    if err := client.Get(ctx, "user:42", &user); err != nil {
        fmt.Println("miss or error:", err)
    }

    // Get-or-Set
    val, err := client.GetOrSet(ctx, "report:monthly", func() (any, error) {
        return db.RunExpensiveReport(ctx)
    }, time.Hour)
    if err != nil {
        panic(err)
    }
    _ = val

    // Atomic counter
    views, _ := client.Incr(ctx, "page:views")
    fmt.Println("views:", views)
}
```

Create your free instance at **[cachly.dev](https://cachly.dev)** — no credit card required.

---

## Semantic AI Cache

Cache LLM responses **by meaning**, not exact text. The same prompt phrased differently returns the cached answer — cutting OpenAI costs by up to 60 %.

```go
sem := client.Semantic(func(ctx context.Context, text string) ([]float64, error) {
    return openaiClient.Embed(ctx, text)
})

result, err := sem.GetOrSet(ctx, userQuestion, func() (any, error) {
    return openaiClient.Ask(ctx, userQuestion)
}, cachly.SemanticOptions{
    Threshold: 0.92,
    TTL:       time.Hour,
    Namespace: "cachly:sem", // optional, default: "cachly:sem"
})

if result.Hit {
    fmt.Printf("cache hit (similarity=%.3f)\n", result.Similarity)
} else {
    fmt.Println("miss")
}
```

---

## Batch API — Multiple Ops in One Round-Trip

Bundle GET/SET/DEL/EXISTS/TTL operations into **one** HTTP request (or Redis pipeline).
Saves up to 10× HTTP overhead for LLM pipelines with many parallel cache lookups.

```go
// Optional: batchURL enables HTTP batching instead of Redis pipeline
client, _ := cachly.NewWithBatch(os.Getenv("CACHLY_URL"), os.Getenv("CACHLY_BATCH_URL"))

results, err := client.Batch(ctx, []cachly.BatchOp{
    {Op: "get",    Key: "user:1"},
    {Op: "get",    Key: "config:app"},
    {Op: "set",    Key: "visits", Value: "42", TTL: 86400},
    {Op: "exists", Key: "session:xyz"},
    {Op: "ttl",    Key: "token:abc"},
})
// results[0].Value      → *string (nil on miss)
// results[1].Value      → *string
// results[2].Ok         → *bool
// results[3].Exists     → *bool
// results[4].TTLSeconds → *int64  (-1 = no TTL, -2 = key missing)
```

Without `CACHLY_BATCH_URL` the client falls back automatically to a **Redis pipeline**.

---

## Gin / Chi Middleware

```go
func CacheMiddleware(client *cachly.Client, ttl time.Duration) gin.HandlerFunc {
    return func(c *gin.Context) {
        key := "http:" + c.Request.URL.String()
        var body []byte
        if err := client.Get(c, key, &body); err == nil {
            c.Data(200, "application/json", body)
            c.Abort()
            return
        }
        c.Next()
        // cache response body after handler
    }
}
```

---

## AI Dev Brain — Persistent Memory for Your Coding Assistant

cachly ships a **30-tool MCP server** that gives Claude Code, Cursor, GitHub Copilot, and Windsurf a persistent memory across sessions — so they never forget your architecture, lessons learned, or last session context.

```bash
# One-time setup
npx @cachly-dev/init
```

Or configure manually in your editor (`~/.vscode/mcp.json` / `.cursor/mcp.json`):

```json
{
  "servers": {
    "cachly": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@cachly-dev/mcp-server"],
      "env": { "CACHLY_JWT": "your-jwt-token" }
    }
  }
}
```

Add to your AI assistant instructions (e.g. `.github/copilot-instructions.md`):

```markdown
## cachly AI Brain

At the START of every session:
session_start(instance_id = "your-instance-id", focus = "what you're working on today")

At the END of every session:
session_end(instance_id = "your-instance-id", summary = "...", files_changed = [...])

After any bug fix or deploy:
learn_from_attempts(instance_id = "your-instance-id", topic = "category:keyword",
  outcome = "success", what_worked = "...", what_failed = "...", severity = "major")
```

`session_start` returns a full briefing in **one call**: last session summary, relevant lessons, open failures, brain health. 60 % fewer file reads, instant context, zero re-discovery.

→ Full docs: [cachly.dev/docs/ai-memory](https://cachly.dev/docs/ai-memory)

---

## LLM Response Caching Proxy

Use cachly as a **drop-in caching proxy** for OpenAI or Anthropic — no SDK changes needed:

```bash
# Instead of https://api.openai.com — use your cachly proxy URL:
OPENAI_BASE_URL=https://api.cachly.dev/v1/llm-proxy/YOUR_TOKEN/openai

# Anthropic:
ANTHROPIC_BASE_URL=https://api.cachly.dev/v1/llm-proxy/YOUR_TOKEN/anthropic
```

Identical requests are served from cache with `X-Cachly-Cache: HIT`. Check savings via `GET /v1/llm-proxy/YOUR_TOKEN/stats`.

---

## Agent Workflow Persistence

```go
base := fmt.Sprintf("https://api.cachly.dev/v1/workflow/%s", token)

// Save a checkpoint after each workflow step
body := `{"run_id":"my-run-123","step_index":0,"step_name":"research",
  "agent_name":"researcher","status":"completed",
  "state":"{\"topic\":\"AI caching\"}"}`
http.Post(base+"/checkpoints", "application/json", strings.NewReader(body))

// Resume: get the latest checkpoint
resp, _ := http.Get(base + "/runs/my-run-123/latest")
// → {"step_index": 2, "step_name": "write", "state": "...", "status": "completed"}
```

---

## Connection Pooling & Keep-Alive

```go
client, _ := cachly.NewWithConfig(cachly.Config{
    URL: os.Getenv("CACHLY_URL"),
    Pool: &cachly.PoolConfig{
        PoolSize:        20,                       // max connections (default: 10×GOMAXPROCS)
        MinIdleConns:    5,                        // keep 5 warm connections
        KeepAlive:       30 * time.Second,         // PING every 30s
        MaxRetries:      3,                        // retry failed commands
        MinRetryBackoff: 8 * time.Millisecond,
        MaxRetryBackoff: 512 * time.Millisecond,
        ConnMaxIdleTime: 5 * time.Minute,
        ConnMaxLifetime: 30 * time.Minute,
    },
})
defer client.Close()
```

Disable retries with `MaxRetries: -1`.

---

## OpenTelemetry Tracing

```go
import "github.com/redis/go-redis/extra/redisotel/v9"

client, _ := cachly.NewWithConfig(cachly.Config{URL: os.Getenv("CACHLY_URL")})
client.AddHook(redisotel.InstrumentTracing())

// Every Get/Set/Delete/Incr now produces OTEL spans:
//   span: "redis.get"  attributes: { db.statement: "get user:42" }
```

---

## API Reference

| Method | Signature | Description |
|---|---|---|
| `New` | `(url string) (*Client, error)` | Create client from Redis URL |
| `NewWithOptions` | `(opts *redis.Options) *Client` | Create client from custom redis.Options |
| `Ping` | `(ctx) error` | Check connectivity |
| `Get` | `(ctx, key, dst) error` | Get + JSON-decode; `redis.Nil` on miss |
| `Set` | `(ctx, key, value, ttl) error` | Set + JSON-encode; `ttl=0` = no expiry |
| `Delete` | `(ctx, ...keys) (int64, error)` | Delete one or more keys; returns count deleted |
| `Exists` | `(ctx, ...keys) (bool, error)` | True if all given keys exist |
| `Expire` | `(ctx, key, ttl) (bool, error)` | Update TTL; false if key not found |
| `TTL` | `(ctx, key) (time.Duration, error)` | Remaining TTL of a key |
| `Incr` | `(ctx, key) (int64, error)` | Atomic increment |
| `IncrBy` | `(ctx, key, n) (int64, error)` | Atomic increment by n |
| `GetOrSet` | `(ctx, key, fn, ttl) (any, error)` | Get-or-set; calls fn on miss and stores result |
| `Semantic` | `(embedFn EmbedFn) *SemanticCache` | Semantic AI cache helper |
| `Raw` | `() *redis.Client` | Direct go-redis access |
| `Close` | `() error` | Close connection and stop keep-alive |

### SemanticCache

| Method | Signature | Description |
|---|---|---|
| `GetOrSet` | `(ctx, prompt, fn, opts) (*SemanticResult, error)` | Get-or-set by semantic similarity |
| `Flush` | `(ctx, namespace) (int64, error)` | Delete all entries in namespace |
| `Size` | `(ctx, namespace) (int64, error)` | Count entries in namespace |

### Types

```go
type EmbedFn func(ctx context.Context, text string) ([]float64, error)

type SemanticOptions struct {
    Threshold float64       // cosine-similarity cutoff (0–1), default 0.85
    TTL       time.Duration // 0 = no expiry
    Namespace string        // key prefix, default "cachly:sem"
}

type SemanticResult struct {
    Value      any
    Hit        bool
    Similarity float64 // only set on cache hit
}
```

---

## Environment Variables

```bash
CACHLY_URL=redis://:your-password@my-app.cachly.dev:30101
CACHLY_BATCH_URL=https://api.cachly.dev/v1/cache/YOUR_TOKEN   # optional
```

---

## Links

- 📖 [cachly.dev docs](https://cachly.dev/docs)
- 🧠 [AI Memory / MCP Server](https://cachly.dev/docs/ai-memory)
- 🐛 [Issues](https://github.com/cachly-dev/sdk-go/issues)
- 📦 [pkg.go.dev](https://pkg.go.dev/github.com/cachly-dev/sdk-go)

---

MIT © [cachly.dev](https://cachly.dev)
