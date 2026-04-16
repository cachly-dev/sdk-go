# cachly Go SDK

Official Go SDK for [cachly.dev](https://cachly.dev) –
Managed Valkey/Redis cache. **DSGVO-compliant · German servers · 30s provisioning.**

## Installation

```bash
go get github.com/cachly-dev/sdk-go
```

> Requires Go 1.21+. Uses `github.com/redis/go-redis/v9`.

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

    // Get-or-Set (returns the value, no dst pointer)
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

## Semantic AI Cache (Speed / Business tiers)

```go
import (
    "github.com/cachly-dev/sdk-go/cachly"
)

// embedFn must match the EmbedFn signature: func(ctx, text) ([]float64, error)
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
    fmt.Printf("⚡ hit (similarity=%.3f)\n", result.Similarity)
} else {
    fmt.Println("🔄 miss")
}
fmt.Println(result.Value)
```

## Gin / Chi Middleware

```go
// Gin middleware example
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
| `Close` | `() error` | Close connection + stop keep-alive |

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

## Batch API – mehrere Ops in einem Round-Trip

Bündelt GET/SET/DEL/EXISTS/TTL-Ops in **einem** HTTP-Request.
Spart bis zu 10× HTTP-Overhead bei LLM-Pipelines.

```go
import "github.com/cachly-dev/sdk-go/cachly"

// Optional: batchURL aktiviert HTTP-Batching statt Redis-Pipeline
client, _ := cachly.NewWithBatch(os.Getenv("CACHLY_URL"), os.Getenv("CACHLY_BATCH_URL"))

results, err := client.Batch(ctx, []cachly.BatchOp{
    {Op: "get",    Key: "user:1"},
    {Op: "get",    Key: "config:app"},
    {Op: "set",    Key: "visits", Value: "42", TTL: 86400},
    {Op: "exists", Key: "session:xyz"},
    {Op: "ttl",    Key: "token:abc"},
})
// results[0].Value  → *string (nil on miss)
// results[1].Value  → *string
// results[2].Ok     → *bool
// results[3].Exists → *bool
// results[4].TTLSeconds → *int64  (-1 = kein TTL, -2 = nicht vorhanden)
```

Ohne `CACHLY_BATCH_URL` fällt der Client automatisch auf eine **Redis-Pipeline** zurück.

## Connection Pooling & Keep-Alive

Fine-tune connection behaviour for high-throughput apps:

```go
client, _ := cachly.NewWithConfig(cachly.Config{
    URL: os.Getenv("CACHLY_URL"),
    Pool: &cachly.PoolConfig{
        PoolSize:        20,                // max connections (default: 10×GOMAXPROCS)
        MinIdleConns:    5,                 // keep 5 warm connections
        KeepAlive:       30 * time.Second,  // PING every 30s (prevents firewall idle-disconnect)
        MaxRetries:      3,                 // retry failed commands
        MinRetryBackoff: 8 * time.Millisecond,
        MaxRetryBackoff: 512 * time.Millisecond,
        ConnMaxIdleTime: 5 * time.Minute,   // recycle idle connections
        ConnMaxLifetime: 30 * time.Minute,  // max connection age
    },
})
defer client.Close()
```

## LLM Response Caching Proxy

Use cachly as a **drop-in caching proxy** for OpenAI or Anthropic — no SDK changes
needed. Just swap the base URL:

```bash
# Instead of https://api.openai.com → use your cachly proxy URL:
OPENAI_BASE_URL=https://api.cachly.dev/v1/llm-proxy/YOUR_TOKEN/openai

# Anthropic:
ANTHROPIC_BASE_URL=https://api.cachly.dev/v1/llm-proxy/YOUR_TOKEN/anthropic
```

Identical requests are served from cache with `X-Cachly-Cache: HIT` header.
Check savings via `GET /v1/llm-proxy/YOUR_TOKEN/stats`.

## Agent Workflow Persistence

Checkpoint agent workflow state so agents can resume from the last step on crash:

```go
import "net/http"

base := fmt.Sprintf("https://api.cachly.dev/v1/workflow/%s", token)

// Save a checkpoint after each workflow step
body := `{"run_id":"my-run-123","step_index":0,"step_name":"research",
  "agent_name":"researcher","status":"completed",
  "state":"{\"topic\":\"AI caching\"}"}`
http.Post(base+"/checkpoints", "application/json", strings.NewReader(body))

// Resume: get the latest checkpoint for a run
resp, _ := http.Get(base + "/runs/my-run-123/latest")
// → {"step_index": 2, "step_name": "write", "state": "...", "status": "completed"}
```

## Environment Variables

```bash
CACHLY_URL=redis://:your-password@my-app.cachly.dev:30101
CACHLY_BATCH_URL=https://api.cachly.dev/v1/cache/YOUR_TOKEN   # optional
```

## Retry with Exponential Backoff

go-redis has built-in command retries via the `PoolConfig`:

```go
client, _ := cachly.NewWithConfig(cachly.Config{
    URL: os.Getenv("CACHLY_URL"),
    Pool: &cachly.PoolConfig{
        MaxRetries:      3,                      // retry up to 3× (default)
        MinRetryBackoff: 8 * time.Millisecond,   // first retry delay
        MaxRetryBackoff: 512 * time.Millisecond, // delay cap
    },
})
```

Disable retries with `MaxRetries: -1`.

## OpenTelemetry Tracing

Use the official `redisotel` hook to auto-instrument every command:

```go
import "github.com/redis/go-redis/extra/redisotel/v9"

client, _ := cachly.NewWithConfig(cachly.Config{URL: os.Getenv("CACHLY_URL")})
client.AddHook(redisotel.InstrumentTracing())

// Every Get/Set/Delete/Incr now produces OTEL spans:
//   span: "redis.get"  attributes: { db.statement: "get user:42" }
```

## License

MIT © [cachly.dev](https://cachly.dev)

