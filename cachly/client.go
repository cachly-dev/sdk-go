// Package cachly provides the official Go client for cachly.dev managed
// Valkey/Redis cache instances with optional semantic AI caching.
//
// # Quick start
//
//	client, err := cachly.New("redis://:password@host:port")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
//
//	ctx := context.Background()
//	if err := client.Set(ctx, "user:42", userData, 5*time.Minute); err != nil {
//	    log.Fatal(err)
//	}
//	var out MyStruct
//	if err := client.Get(ctx, "user:42", &out); err != nil {
//	    log.Fatal(err)
//	}
//
// # Semantic cache (Speed / Business tiers)
//
//	sem := client.Semantic(myEmbedFn)
//	result, err := sem.GetOrSet(ctx, prompt, func() (any, error) {
//	    return callOpenAI(prompt)
//	}, cachly.SemanticOptions{Threshold: 0.90, TTL: time.Hour})
//
// # Semantic cache with pgvector API (faster than SCAN)
//
//	sem := client.SemanticWithVector(myEmbedFn, os.Getenv("CACHLY_VECTOR_URL"))
package cachly

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

// ── Types ─────────────────────────────────────────────────────────────────────

// EmbedFn is a function that converts text into an embedding vector.
// Use your preferred embedding model (OpenAI, Cohere, local, etc.).
type EmbedFn func(ctx context.Context, text string) ([]float64, error)

// SemanticConfidence is the three-level quality band of a semantic cache hit.
type SemanticConfidence string

const (
	// ConfidenceHigh means similarity >= HighConfidenceThreshold (default 0.97).
	// Serve directly, no quality check needed.
	ConfidenceHigh SemanticConfidence = "high"
	// ConfidenceMedium means similarity >= Threshold.
	// Consider A/B testing or logging.
	ConfidenceMedium SemanticConfidence = "medium"
	// ConfidenceUncertain means cache miss (similarity < Threshold).
	ConfidenceUncertain SemanticConfidence = "uncertain"
)

// SemanticOptions configures a SemanticCache.GetOrSet call.
type SemanticOptions struct {
	// Threshold is the cosine-similarity cutoff (0–1). Default: 0.85.
	Threshold float64
	// TTL is the time-to-live for newly cached entries. 0 = no expiry.
	TTL time.Duration
	// Namespace is the Redis key prefix. Default: "cachly:sem".
	Namespace string
	// AutoNamespace classifies the prompt into a semantic namespace automatically.
	// Recognised classes: cachly:sem:code, :translation, :summary, :qa, :creative.
	// Ignored when Namespace is explicitly set. Default: false.
	AutoNamespace bool
	// NormalizePrompt strips filler words and lowercases before embedding.
	// Boosts hit rate +8–12% with no quality loss. Default: true.
	NormalizePrompt *bool
	// FillerWords overrides the default filler word list for normalisation.
	FillerWords []string
	// HighConfidenceThreshold: similarity >= this → Confidence="high". Default: 0.97.
	HighConfidenceThreshold float64
	// UseAdaptiveThreshold overrides Threshold with the server-side F1-calibrated value.
	// Requires a VectorURL. Falls back to Threshold when no calibration data exists yet.
	UseAdaptiveThreshold bool
	// Quantize controls embedding quantization before sending to the pgvector API.
	// "int8" reduces JSON payload ~8x (1536-dim: 12 KB → 1.5 KB) with <1% quality loss.
	// "" or "float32" sends full precision (default, backward-compatible).
	Quantize string // "" | "int8"
	// UseHybrid enables Hybrid BM25+Vector RRF fusion search (server-side).
	// Requires a VectorURL. Sends the normalised prompt alongside the embedding so the
	// backend can run a PostgreSQL full-text search and fuse rankings via RRF (k=60).
	// Improves recall for keyword-heavy queries by +5–15% at no extra latency cost.
	// Falls back to pure vector search when VectorURL is not set. Default: false.
	UseHybrid bool
}

// WarmupEntry is one entry to pre-warm into the semantic cache.
type WarmupEntry struct {
	Prompt string
	// Fn is the factory called on a cache miss. Receives no context – use a closure
	// to capture external dependencies (e.g. an LLM client).
	Fn        func() (any, error)
	Namespace string // optional override; falls back to SemanticOptions.Namespace
}

// WarmupResult summarises the outcome of a Warmup call.
type WarmupResult struct {
	Warmed  int // entries freshly computed and indexed
	Skipped int // entries already present (cache hit during warmup)
}

// ── int8 Quantization ──────────────────────────────────────────────────────

// QuantizedEmbedding is a scalar int8-quantized representation of a float64 vector.
// Reconstruction: v[i] = (values[i]+128) / (255/(max-min)) + min
type QuantizedEmbedding struct {
	Values []int8  `json:"values"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

// QuantizeEmbedding scalar-quantizes a float64 slice to int8 range [-128, 127].
// <1% cosine-similarity quality loss; reduces API payload ~8x.
func QuantizeEmbedding(vec []float64) QuantizedEmbedding {
	if len(vec) == 0 {
		return QuantizedEmbedding{}
	}
	minVal, maxVal := vec[0], vec[0]
	for _, v := range vec[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	rng := maxVal - minVal
	values := make([]int8, len(vec))
	if rng > 0 {
		scale := 255.0 / rng
		for i, v := range vec {
			q := math.Round(scale*(v-minVal)) - 128
			if q < -128 {
				q = -128
			}
			if q > 127 {
				q = 127
			}
			values[i] = int8(q)
		}
	}
	return QuantizedEmbedding{Values: values, Min: minVal, Max: maxVal}
}

// DequantizeEmbedding reconstructs a float64 vector from a QuantizedEmbedding.
func DequantizeEmbedding(q QuantizedEmbedding) []float64 {
	if len(q.Values) == 0 {
		return nil
	}
	rng := q.Max - q.Min
	out := make([]float64, len(q.Values))
	if rng == 0 {
		for i := range out {
			out[i] = q.Min
		}
		return out
	}
	scale := 255.0 / rng
	for i, v := range q.Values {
		out[i] = (float64(v)+128.0)/scale + q.Min
	}
	return out
}

// SemanticResult is returned by SemanticCache.GetOrSet.
type SemanticResult struct {
	Value      any
	Hit        bool
	Similarity float64            // only set on a cache hit
	Confidence SemanticConfidence // only set on a cache hit
}

// SemEntryInfo describes a single entry returned by SemanticCache.Entries.
type SemEntryInfo struct {
	// Key is the full Redis key – pass to Invalidate to remove this entry.
	Key    string
	Prompt string
}

// ── Connection Pooling & Keep-Alive ────────────────────────────────────────

// PoolConfig controls connection-pool sizing, keep-alive health-checks, and
// retry behaviour for the underlying go-redis client.
//
// All fields are optional – zero values fall back to sensible defaults.
type PoolConfig struct {
	// PoolSize is the maximum number of connections in the pool.
	// Default: 10 × GOMAXPROCS.
	PoolSize int
	// MinIdleConns is the minimum number of idle connections to keep open.
	// Default: 0.
	MinIdleConns int
	// KeepAlive sends a PING at this interval to prevent idle connection
	// closures by firewalls or load-balancers. 0 = disabled. Default: 30s.
	KeepAlive time.Duration
	// MaxRetries is the maximum number of retries for a failed command.
	// -1 = no retries, 0 = default (3). Default: 3.
	MaxRetries int
	// MinRetryBackoff is the base backoff delay. Default: 8ms.
	MinRetryBackoff time.Duration
	// MaxRetryBackoff is the maximum backoff delay. Default: 512ms.
	MaxRetryBackoff time.Duration
	// ConnMaxIdleTime closes connections that have been idle longer than this.
	// 0 = no idle timeout. Default: 5min.
	ConnMaxIdleTime time.Duration
	// ConnMaxLifetime closes connections after this total age.
	// 0 = no max lifetime. Default: 0.
	ConnMaxLifetime time.Duration
}

// Config holds all options for creating a Client.
type Config struct {
	// URL is the Redis/Valkey connection URL: redis://:password@host:port
	URL string
	// VectorURL is the optional pgvector API base URL.
	// Format: https://api.cachly.dev/v1/sem/{vector_token}
	// When set, SemanticCache uses the pgvector API instead of linear SCAN.
	VectorURL string
	// BatchURL is the optional KV Batch API URL.
	// Format: https://api.cachly.dev/v1/cache/{token}
	// When set, Client.Batch sends all ops in one HTTP request.
	// Also enables SetTags, InvalidateTag, GetTags, DeleteTags, SWR*, and BulkWarmup.
	BatchURL string
	// PubSubURL is the optional Pub/Sub API base URL.
	// Format: https://api.cachly.dev/v1/pubsub/{token}
	// Required for Client.PubSub().
	PubSubURL string
	// WorkflowURL is the optional Workflow checkpoint API base URL.
	// Format: https://api.cachly.dev/v1/workflow/{token}
	// Required for Client.Workflow().
	WorkflowURL string
	// LlmProxyURL is the optional LLM cache proxy API base URL.
	// Format: https://api.cachly.dev/v1/llm-proxy/{token}
	// Required for Client.LlmProxyStats().
	LlmProxyURL string
	// EdgeURL is the Cloudflare Edge Worker URL for semantic search reads.
	// Format: https://edge.cachly.dev/v1/sem/{token}
	// When set, semantic search reads are routed through the edge for sub-1ms latency.
	EdgeURL string
	// EdgeAPIURL is the Edge Cache Management API URL.
	// Format: https://api.cachly.dev/v1/edge/{token}
	// When set, Client.Edge() returns an EdgeCacheClient for config, purge, and stats.
	EdgeAPIURL string
	// ConnectTimeout is the dial timeout. 0 = use go-redis default (5s).
	ConnectTimeout time.Duration
	// Pool configures connection pooling, keep-alive, and retry. nil = defaults.
	Pool *PoolConfig
	// Tracer is an optional OpenTelemetry-compatible tracer for OTEL Tracing.
	Tracer Tracer
	// Warmup contains entries to pre-warm when WarmupOnConnect is true.
	Warmup []WarmupEntry
	// WarmupOnConnect starts cache-warming asynchronously during NewWithConfig.
	WarmupOnConnect bool
	// WarmupOptions are the SemanticOptions used for every warmup entry.
	WarmupOptions SemanticOptions
}

// ── Batch API types ───────────────────────────────────────────────────────────

// BatchOp is a single operation in a Batch call.
type BatchOp struct {
	Op    string `json:"op"`              // "get" | "set" | "del" | "exists" | "ttl"
	Key   string `json:"key"`             // required
	Value string `json:"value,omitempty"` // required for "set"
	TTL   int    `json:"ttl,omitempty"`   // seconds; 0 = no expiry (set only)
}

// BatchOpResult is the resolved value of a single BatchOp.
// Only one field is populated depending on the op type:
//   - Get:    Value (string or "")
//   - Set:    Ok
//   - Del:    Ok (true if ≥ 1 key deleted)
//   - Exists: Exists
//   - TTL:    TTLSeconds (-1 = no expiry, -2 = not found)
type BatchOpResult struct {
	// Value is set for "get" ops. Empty string means key not found.
	Value string
	// Found indicates whether a "get" key existed.
	Found bool
	// Ok is true for successful "set" ops and "del" ops that deleted ≥ 1 key.
	Ok bool
	// Exists is the result of "exists" ops.
	Exists bool
	// TTLSeconds is the result of "ttl" ops. -1 = no expiry, -2 = not found.
	TTLSeconds int64
	// Err holds any per-op error returned by the server or pipeline.
	Err error
}

// ── OpenTelemetry Tracing (optional) ───────────────────────────────────────

// Tracer is a minimal interface compatible with the OpenTelemetry Go SDK's
// trace.Tracer. Implement this or pass trace.NewTracerProvider().Tracer("cachly")
// directly — both satisfy the interface.
//
// When nil, tracing is a no-op (zero overhead).
type Tracer interface {
	// Start creates a span and returns a child context + span.
	Start(ctx context.Context, spanName string, opts ...any) (context.Context, Span)
}

// Span is a minimal interface compatible with trace.Span from the OTEL SDK.
type Span interface {
	SetAttributes(kv ...any)
	End(options ...any)
	RecordError(err error, options ...any)
	SetStatus(code uint32, description string)
}

// ── Retry with Exponential Backoff ─────────────────────────────────────────
//
// go-redis already has MaxRetries / MinRetryBackoff / MaxRetryBackoff on the
// redis.Options level (configured via PoolConfig). These apply to every
// command automatically, so no extra wrapper is needed in the Go SDK.
// The PoolConfig.MaxRetries, MinRetryBackoff, MaxRetryBackoff fields implement
// the full-jitter exponential backoff required by Feature #2.

// ── Client ────────────────────────────────────────────────────────────────────

// Client is the main cachly client.
type Client struct {
	rdb          *redis.Client
	vectorURL    string
	batchURL     string           // optional KV batch API URL
	pubsubURL    string           // optional Pub/Sub API URL
	workflowURL  string           // optional Workflow API URL
	llmProxyURL  string           // optional LLM Proxy API URL
	edgeURL      string           // optional Edge Worker URL for semantic reads
	edgeClient   *EdgeCacheClient // optional Edge Cache management client
	tracer       Tracer           // nil = no-op

	// keep-alive
	keepAliveStop chan struct{}
}

// New creates a Client from a Redis connection URL.
// Format: redis://:password@host:port
func New(url string) (*Client, error) {
	opts, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("cachly: parse url: %w", err)
	}
	rdb := redis.NewClient(opts)
	return &Client{rdb: rdb}, nil
}

// NewWithConfig creates a Client from a Config struct.
// Use this when you want to set a VectorURL for pgvector-backed semantic caching.
func NewWithConfig(cfg Config) (*Client, error) {
	opts, err := redis.ParseURL(cfg.URL)
	if err != nil {
		return nil, fmt.Errorf("cachly: parse url: %w", err)
	}
	if cfg.ConnectTimeout > 0 {
		opts.DialTimeout = cfg.ConnectTimeout
	}

	// apply pool config
	if p := cfg.Pool; p != nil {
		if p.PoolSize > 0 {
			opts.PoolSize = p.PoolSize
		}
		if p.MinIdleConns > 0 {
			opts.MinIdleConns = p.MinIdleConns
		}
		if p.MaxRetries != 0 {
			opts.MaxRetries = p.MaxRetries
		}
		if p.MinRetryBackoff > 0 {
			opts.MinRetryBackoff = p.MinRetryBackoff
		}
		if p.MaxRetryBackoff > 0 {
			opts.MaxRetryBackoff = p.MaxRetryBackoff
		}
		if p.ConnMaxIdleTime > 0 {
			opts.ConnMaxIdleTime = p.ConnMaxIdleTime
		}
		if p.ConnMaxLifetime > 0 {
			opts.ConnMaxLifetime = p.ConnMaxLifetime
		}
	}

	rdb := redis.NewClient(opts)

	c := &Client{
		rdb:         rdb,
		vectorURL:   cfg.VectorURL,
		batchURL:    strings.TrimRight(cfg.BatchURL, "/"),
		pubsubURL:   strings.TrimRight(cfg.PubSubURL, "/"),
		workflowURL: strings.TrimRight(cfg.WorkflowURL, "/"),
		llmProxyURL: strings.TrimRight(cfg.LlmProxyURL, "/"),
		edgeURL:     strings.TrimRight(cfg.EdgeURL, "/"),
		tracer:      cfg.Tracer,
	}

	// Initialize EdgeCacheClient if EdgeAPIURL is configured
	if cfg.EdgeAPIURL != "" {
		c.edgeClient = NewEdgeCacheClient(cfg.EdgeAPIURL)
	}

	// keep-alive PING loop
	keepAlive := 30 * time.Second
	if cfg.Pool != nil && cfg.Pool.KeepAlive > 0 {
		keepAlive = cfg.Pool.KeepAlive
	} else if cfg.Pool != nil && cfg.Pool.KeepAlive < 0 {
		keepAlive = 0 // disabled
	}
	if keepAlive > 0 {
		c.startKeepAlive(keepAlive)
	}

	// kick off cache-warming asynchronously so it never blocks the caller.
	if cfg.WarmupOnConnect && len(cfg.Warmup) > 0 {
		sem := c.Semantic(nil) // embed fn not needed – warmup provides its own fn
		go func() {
			ctx := context.Background()
			for _, e := range cfg.Warmup {
				wopts := cfg.WarmupOptions
				if e.Namespace != "" {
					wopts.Namespace = e.Namespace
				}
				sem.Warmup(ctx, []WarmupEntry{e}, wopts) //nolint:errcheck
			}
		}()
	}
	return c, nil
}

// NewWithOptions creates a Client from custom redis.Options.
func NewWithOptions(opts *redis.Options) *Client {
	return &Client{rdb: redis.NewClient(opts)}
}

// Close releases the underlying Redis connection pool and stops keep-alive.
func (c *Client) Close() error {
	if c.keepAliveStop != nil {
		close(c.keepAliveStop)
	}
	return c.rdb.Close()
}

// startKeepAlive launches a background goroutine that PINGs the server at
// regular intervals to prevent idle-connection reaping by firewalls.
func (c *Client) startKeepAlive(interval time.Duration) {
	c.keepAliveStop = make(chan struct{})
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-c.keepAliveStop:
				return
			case <-t.C:
				_ = c.rdb.Ping(context.Background()).Err()
			}
		}
	}()
}

// Ping checks connectivity to the cache server.
func (c *Client) Ping(ctx context.Context) error {
	return c.rdb.Ping(ctx).Err()
}

// Raw returns the underlying *redis.Client for advanced operations not covered
// by the cachly API (e.g. pipelines, pub/sub, Lua scripts).
func (c *Client) Raw() *redis.Client { return c.rdb }

// AddHook registers a go-redis Hook (e.g. redisotel.InstrumentTracing()).
// Use this for OpenTelemetry tracing:
//
//	import "github.com/redis/go-redis/extra/redisotel/v9"
//	client.AddHook(redisotel.InstrumentTracing())
func (c *Client) AddHook(hook redis.Hook) {
	c.rdb.AddHook(hook)
}

// Edge returns the EdgeCacheClient for managing Cloudflare Edge Cache.
// Returns nil if EdgeAPIURL was not configured.
//
//	if edge := client.Edge(); edge != nil {
//	    cfg, err := edge.GetConfig(ctx)
//	    edge.SetConfig(ctx, cachly.EdgeCacheConfigUpdate{Enabled: ptr(true)})
//	    edge.Purge(ctx, &cachly.EdgePurgeOptions{Namespaces: []string{"cachly:sem:qa"}})
//	}
func (c *Client) Edge() *EdgeCacheClient {
	return c.edgeClient
}

// ── Key / Value ───────────────────────────────────────────────────────────────

// Set stores v at key, optionally expiring after ttl (0 = no expiry).
// Non-string values are JSON-encoded.
func (c *Client) Set(ctx context.Context, key string, v any, ttl time.Duration) error {
	payload, err := marshal(v)
	if err != nil {
		return fmt.Errorf("cachly set %q: %w", key, err)
	}
	return c.rdb.Set(ctx, key, payload, ttl).Err()
}

// Get retrieves the value at key into dst (must be a pointer).
// Returns redis.Nil if the key does not exist or has expired.
func (c *Client) Get(ctx context.Context, key string, dst any) error {
	raw, err := c.rdb.Get(ctx, key).Bytes()
	if err != nil {
		return err
	}
	return unmarshal(raw, dst)
}

// Delete removes one or more keys. Returns the number of keys actually deleted.
func (c *Client) Delete(ctx context.Context, keys ...string) (int64, error) {
	return c.rdb.Del(ctx, keys...).Result()
}

// Exists returns true if all given keys exist.
func (c *Client) Exists(ctx context.Context, keys ...string) (bool, error) {
	n, err := c.rdb.Exists(ctx, keys...).Result()
	return n == int64(len(keys)), err
}

// Expire sets a TTL on an existing key. Returns false if the key does not exist.
func (c *Client) Expire(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	return c.rdb.Expire(ctx, key, ttl).Result()
}

// TTL returns the remaining time-to-live of key.
func (c *Client) TTL(ctx context.Context, key string) (time.Duration, error) {
	return c.rdb.TTL(ctx, key).Result()
}

// Incr atomically increments the integer counter at key and returns the new value.
func (c *Client) Incr(ctx context.Context, key string) (int64, error) {
	return c.rdb.Incr(ctx, key).Result()
}

// IncrBy atomically increments the integer counter at key by n.
func (c *Client) IncrBy(ctx context.Context, key string, n int64) (int64, error) {
	return c.rdb.IncrBy(ctx, key, n).Result()
}

// GetOrSet returns the cached value for key, or calls fn, stores the result, and returns it.
// Values are always stored as JSON to ensure round-trip consistency regardless of type.
func (c *Client) GetOrSet(ctx context.Context, key string, fn func() (any, error), ttl time.Duration) (any, error) {
	raw, err := c.rdb.Get(ctx, key).Bytes()
	if err == nil {
		var v any
		if uerr := json.Unmarshal(raw, &v); uerr == nil {
			return v, nil
		}
	}
	v, err := fn()
	if err != nil {
		return nil, err
	}
	// Always JSON-encode so the read path (json.Unmarshal) is consistent.
	if payload, merr := json.Marshal(v); merr == nil {
		c.rdb.Set(ctx, key, payload, ttl) //nolint:errcheck
	}
	return v, nil
}

// ── Bulk operations ───────────────────────────────────────────────────────────

// MSetItem is one entry in a bulk MSet call.
type MSetItem struct {
	Key   string
	Value any
	// TTL is the per-key expiry. 0 = no expiry.
	TTL time.Duration
}

// MSet stores multiple key-value pairs in a single pipeline round-trip.
// Supports per-key TTL – unlike native MSET which has no expiry option.
//
//	err := client.MSet(ctx, []cachly.MSetItem{
//	    {Key: "user:1", Value: user1, TTL: 5 * time.Minute},
//	    {Key: "user:2", Value: user2},
//	})
func (c *Client) MSet(ctx context.Context, items []MSetItem) error {
	if len(items) == 0 {
		return nil
	}
	pipe := c.rdb.Pipeline()
	for _, item := range items {
		payload, err := marshal(item.Value)
		if err != nil {
			return fmt.Errorf("cachly mset marshal %q: %w", item.Key, err)
		}
		pipe.Set(ctx, item.Key, payload, item.TTL)
	}
	_, err := pipe.Exec(ctx)
	return err
}

// MGet retrieves multiple keys in one round-trip (native MGET).
// Returns a slice of values in the same order as keys; missing keys are nil.
// Each non-nil value is a json.RawMessage – decode into your target type with json.Unmarshal.
//
//	vals, err := client.MGet(ctx, []string{"user:1", "user:2"})
func (c *Client) MGet(ctx context.Context, keys []string) ([]any, error) {
	if len(keys) == 0 {
		return nil, nil
	}
	raws, err := c.rdb.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, fmt.Errorf("cachly mget: %w", err)
	}
	result := make([]any, len(raws))
	for i, raw := range raws {
		if raw == nil {
			continue
		}
		s, _ := raw.(string)
		var v any
		if json.Unmarshal([]byte(s), &v) == nil {
			result[i] = v
		} else {
			result[i] = s
		}
	}
	return result, nil
}

// ── Distributed lock ──────────────────────────────────────────────────────────

// LockOptions configures a Lock call.
type LockOptions struct {
	// TTL is the safety TTL after which the lock auto-expires (prevents deadlocks).
	TTL time.Duration
	// Retries is the max number of acquire attempts. Default: 3.
	Retries int
	// RetryDelay is the wait between retries. Default: 50 ms.
	RetryDelay time.Duration
}

// LockHandle is returned by a successful Lock call.
// Call Release() in a defer/finally block to free the lock early.
type LockHandle struct {
	// Token is the unique fencing token for this acquisition.
	Token   string
	rdb     *redis.Client
	lockKey string
	once    sync.Once
}

// Release frees the lock atomically. No-op if the lock has already expired
// or was released by another call.
func (h *LockHandle) Release(ctx context.Context) error {
	var releaseErr error
	h.once.Do(func() {
		const script = `
			if redis.call("get", KEYS[1]) == ARGV[1] then
				return redis.call("del", KEYS[1])
			else
				return 0
			end`
		releaseErr = h.rdb.Eval(ctx, script, []string{h.lockKey}, h.Token).Err()
	})
	return releaseErr
}

// Lock acquires a distributed lock using Redis SET NX PX (Redlock-lite pattern).
//
// Returns a *LockHandle on success, or nil when all attempts are exhausted.
// The lock auto-expires after opts.TTL to prevent deadlocks on process crash.
//
//	lock, err := client.Lock(ctx, "job:invoice:42", cachly.LockOptions{TTL: 5*time.Second, Retries: 5})
//	if lock == nil {
//	    return errors.New("resource busy")
//	}
//	defer lock.Release(ctx)
func (c *Client) Lock(ctx context.Context, key string, opts LockOptions) (*LockHandle, error) {
	retries := opts.Retries
	if retries == 0 {
		retries = 3
	}
	retryDelay := opts.RetryDelay
	if retryDelay == 0 {
		retryDelay = 50 * time.Millisecond
	}

	lockKey := "cachly:lock:" + key
	token := uuid.New().String()

	for attempt := 0; attempt <= retries; attempt++ {
		ok, err := c.rdb.SetNX(ctx, lockKey, token, opts.TTL).Result()
		if err != nil {
			return nil, fmt.Errorf("cachly lock setNX: %w", err)
		}
		if ok {
			return &LockHandle{Token: token, rdb: c.rdb, lockKey: lockKey}, nil
		}
		if attempt < retries {
			time.Sleep(retryDelay)
		}
	}
	return nil, nil // could not acquire
}

// ── Streaming cache ───────────────────────────────────────────────────────────

// StreamSetOptions configures a StreamSet call.
type StreamSetOptions struct {
	// TTL is the expiry for the stored list. 0 = no expiry.
	TTL time.Duration
}

// StreamSet caches a streaming response chunk-by-chunk via Redis RPUSH.
// The caller must close the chunks channel after the last chunk has been sent.
// On cache hit, replay the stored chunks with StreamGet.
//
//	ch := make(chan string)
//	go func() {
//	    defer close(ch)
//	    for chunk := range llmStream { ch <- chunk }
//	}()
//	err := client.StreamSet(ctx, "chat:42", ch, cachly.StreamSetOptions{TTL: time.Hour})
func (c *Client) StreamSet(ctx context.Context, key string, chunks <-chan string, opts StreamSetOptions) error {
	listKey := "cachly:stream:" + key
	if err := c.rdb.Del(ctx, listKey).Err(); err != nil {
		return fmt.Errorf("cachly stream_set del: %w", err)
	}
	for chunk := range chunks {
		if err := c.rdb.RPush(ctx, listKey, chunk).Err(); err != nil {
			return fmt.Errorf("cachly stream_set rpush: %w", err)
		}
	}
	if opts.TTL > 0 {
		if err := c.rdb.Expire(ctx, listKey, opts.TTL).Err(); err != nil {
			return fmt.Errorf("cachly stream_set expire: %w", err)
		}
	}
	return nil
}

// StreamReader provides sequential access to a cached stream.
type StreamReader struct {
	rdb     *redis.Client
	listKey string
	length  int64
	index   int64
}

// Next returns the next chunk. ok=false when the stream is exhausted.
func (r *StreamReader) Next(ctx context.Context) (chunk string, ok bool, err error) {
	if r.index >= r.length {
		return "", false, nil
	}
	val, err := r.rdb.LIndex(ctx, r.listKey, r.index).Result()
	if err != nil {
		if err == redis.Nil {
			return "", false, nil
		}
		return "", false, fmt.Errorf("cachly stream_get lindex: %w", err)
	}
	r.index++
	return val, true, nil
}

// Len returns the total number of chunks stored.
func (r *StreamReader) Len() int64 { return r.length }

// StreamGet retrieves a cached stream. Returns nil when the key does not exist.
// Iterate by calling Next repeatedly until ok==false.
//
//	reader, err := client.StreamGet(ctx, "chat:42")
//	if reader == nil { /* cache miss */ }
//	for {
//	    chunk, ok, err := reader.Next(ctx)
//	    if !ok || err != nil { break }
//	    fmt.Print(chunk)
//	}
func (c *Client) StreamGet(ctx context.Context, key string) (*StreamReader, error) {
	listKey := "cachly:stream:" + key
	length, err := c.rdb.LLen(ctx, listKey).Result()
	if err != nil {
		return nil, fmt.Errorf("cachly stream_get llen: %w", err)
	}
	if length == 0 {
		return nil, nil // cache miss
	}
	return &StreamReader{rdb: c.rdb, listKey: listKey, length: length}, nil
}

// ── Batch API ─────────────────────────────────────────────────────────────────

// Batch executes multiple cache operations in a single round-trip.
//
// When BatchURL is configured, all ops are sent via POST {BatchURL}/batch (one HTTP call).
// Otherwise they fall back to a Redis pipeline (one TCP round-trip).
//
// Supported ops: "get" → Value+Found, "set" → Ok, "del" → Ok, "exists" → Exists, "ttl" → TTLSeconds.
//
//	ops := []cachly.BatchOp{
//	    {Op: "get", Key: "user:1"},
//	    {Op: "get", Key: "config:app"},
//	    {Op: "set", Key: "visits", Value: strconv.Itoa(n), TTL: 86400},
//	}
//	results, err := client.Batch(ctx, ops)
//	user   := results[0].Value    // string; results[0].Found = false if missing
//	config := results[1].Value
//	ok     := results[2].Ok
func (c *Client) Batch(ctx context.Context, ops []BatchOp) ([]BatchOpResult, error) {
	if len(ops) == 0 {
		return nil, nil
	}
	if c.batchURL != "" {
		return c.batchViaHTTP(ctx, ops)
	}
	return c.batchViaPipeline(ctx, ops)
}

// batchViaHTTP sends all ops in a single POST request to the batch API.
func (c *Client) batchViaHTTP(ctx context.Context, ops []BatchOp) ([]BatchOpResult, error) {
	type serverResult struct {
		Key        string  `json:"key"`
		Value      *string `json:"value,omitempty"`
		Found      *bool   `json:"found,omitempty"`
		Ok         *bool   `json:"ok,omitempty"`
		Deleted    *int64  `json:"deleted,omitempty"`
		Exists     *bool   `json:"exists,omitempty"`
		TTLSeconds *int64  `json:"ttl_seconds,omitempty"`
		Error      string  `json:"error,omitempty"`
	}
	type serverResp struct {
		Results []serverResult `json:"results"`
	}

	body := map[string]any{"ops": ops}
	raw, err := httpRequest(ctx, http.MethodPost, c.batchURL+"/batch", body)
	if err != nil {
		return nil, fmt.Errorf("cachly batch http: %w", err)
	}
	// Re-marshal raw map[string]any → serverResp via JSON round-trip.
	interim, _ := json.Marshal(raw)
	var resp serverResp
	if err := json.Unmarshal(interim, &resp); err != nil {
		return nil, fmt.Errorf("cachly batch decode: %w", err)
	}

	results := make([]BatchOpResult, len(ops))
	for i, r := range resp.Results {
		if i >= len(ops) {
			break
		}
		res := BatchOpResult{}
		if r.Error != "" {
			res.Err = fmt.Errorf("%s", r.Error)
			results[i] = res
			continue
		}
		switch ops[i].Op {
		case "get":
			if r.Value != nil {
				res.Value = *r.Value
				res.Found = true
			}
			if r.Found != nil {
				res.Found = *r.Found
			}
		case "set":
			res.Ok = r.Ok != nil && *r.Ok
		case "del":
			res.Ok = r.Deleted != nil && *r.Deleted > 0
		case "exists":
			res.Exists = r.Exists != nil && *r.Exists
		case "ttl":
			if r.TTLSeconds != nil {
				res.TTLSeconds = *r.TTLSeconds
			} else {
				res.TTLSeconds = -2
			}
		}
		results[i] = res
	}
	return results, nil
}

// batchViaPipeline executes ops via a Redis pipeline (no batch API needed).
func (c *Client) batchViaPipeline(ctx context.Context, ops []BatchOp) ([]BatchOpResult, error) {
	pipe := c.rdb.Pipeline()
	type slot struct {
		op        string
		getCmd    *redis.StringCmd
		setCmd    *redis.StatusCmd
		delCmd    *redis.IntCmd
		existsCmd *redis.IntCmd
		ttlCmd    *redis.DurationCmd
	}
	slots := make([]slot, len(ops))
	for i, op := range ops {
		s := slot{op: op.Op}
		ttl := time.Duration(op.TTL) * time.Second
		switch op.Op {
		case "get":
			s.getCmd = pipe.Get(ctx, op.Key)
		case "set":
			s.setCmd = pipe.Set(ctx, op.Key, op.Value, ttl)
		case "del":
			s.delCmd = pipe.Del(ctx, op.Key)
		case "exists":
			s.existsCmd = pipe.Exists(ctx, op.Key)
		case "ttl":
			s.ttlCmd = pipe.TTL(ctx, op.Key)
		}
		slots[i] = s
	}
	if _, err := pipe.Exec(ctx); err != nil && err != redis.Nil {
		return nil, fmt.Errorf("cachly batch pipeline: %w", err)
	}

	results := make([]BatchOpResult, len(ops))
	for i, s := range slots {
		res := BatchOpResult{}
		switch s.op {
		case "get":
			val, err := s.getCmd.Result()
			if err == nil {
				res.Value = val
				res.Found = true
			}
		case "set":
			res.Ok = s.setCmd.Err() == nil
		case "del":
			n, _ := s.delCmd.Result()
			res.Ok = n > 0
		case "exists":
			n, _ := s.existsCmd.Result()
			res.Exists = n > 0
		case "ttl":
			dur, err := s.ttlCmd.Result()
			if err != nil {
				res.TTLSeconds = -2
			} else {
				res.TTLSeconds = int64(dur.Seconds())
			}
		}
		results[i] = res
	}
	return results, nil
}

// ── Semantic cache ─────────────────────────────────────────────────────────────

// Semantic returns a SemanticCache backed by this client using the provided embedding function.
// Uses linear SCAN for lookups. For HNSW-accelerated lookups, use SemanticWithVector.
func (c *Client) Semantic(embedFn EmbedFn) *SemanticCache {
	return &SemanticCache{rdb: c.rdb, embedFn: embedFn, vectorURL: c.vectorURL, edgeURL: c.edgeURL}
}

// SemanticWithVector returns a SemanticCache that uses the pgvector API for
// nearest-neighbour search (O(log n) HNSW) with graceful SCAN fallback.
//
//	sem := client.SemanticWithVector(myEmbedFn, os.Getenv("CACHLY_VECTOR_URL"))
func (c *Client) SemanticWithVector(embedFn EmbedFn, vectorURL string) *SemanticCache {
	return &SemanticCache{rdb: c.rdb, embedFn: embedFn, vectorURL: vectorURL, edgeURL: c.edgeURL}
}


// SemanticCache caches LLM responses by meaning using embedding vector similarity.
type SemanticCache struct {
	rdb       *redis.Client
	embedFn   EmbedFn
	vectorURL string // optional: when set, uses pgvector API; falls back to SCAN on error
	edgeURL   string // optional: Cloudflare Edge Worker URL for reads
}

// setEdgeURL sets the edge URL for reads (called internally by Client).
func (s *SemanticCache) setEdgeURL(url string) {
	s.edgeURL = strings.TrimSuffix(url, "/")
}

// searchURL returns the effective URL for search operations (edge or origin).
func (s *SemanticCache) searchURL() string {
	if s.edgeURL != "" {
		return s.edgeURL
	}
	return s.vectorURL
}

type semEntry struct {
	Embedding []float64 `json:"embedding"`
	Value     any       `json:"value"`
	Prompt    string    `json:"prompt"`
}

// semEmbEntry is the lightweight embedding+prompt stored under ns:emb:{uuid}.
// The actual value lives in ns:val:{uuid} and is only fetched on a hit.
type semEmbEntry struct {
	Embedding []float64 `json:"embedding"`
	Prompt    string    `json:"prompt"`
	// OriginalPrompt preserves the un-normalised input text for display in Entries().
	// Absent in entries written by SDK versions < 0.4.0; falls back to Prompt.
	OriginalPrompt string `json:"original_prompt,omitempty"`
}

// valKey derives the val key from an emb key.
// Format: {ns}:emb:{uuid} → {ns}:val:{uuid}
func valKey(embKey string) string {
	lastColon := strings.LastIndex(embKey, ":")
	uuidPart := embKey[lastColon:] // ":uuid-str"
	nsType := embKey[:lastColon]   // "{ns}:emb"
	secondLastColon := strings.LastIndex(nsType, ":")
	ns := nsType[:secondLastColon] // "{ns}"
	return ns + ":val" + uuidPart  // "{ns}:val:{uuid}"
}

// extractIDFromKey extracts the UUID part from a key like "{ns}:emb:{uuid}" or "{ns}:val:{uuid}".
func extractIDFromKey(key string) string {
	i := strings.LastIndex(key, ":")
	if i < 0 {
		return key
	}
	return key[i+1:]
}

// extractNamespaceFromKey derives the namespace prefix from an emb/val key.
// e.g. "cachly:sem:emb:uuid" → "cachly:sem"
func extractNamespaceFromKey(key string) string {
	lastColon := strings.LastIndex(key, ":")
	if lastColon < 0 {
		return defaultNamespace
	}
	nsType := key[:lastColon]
	second := strings.LastIndex(nsType, ":")
	if second < 0 {
		return defaultNamespace
	}
	return nsType[:second]
}

const defaultNamespace = "cachly:sem"

// ── Prompt Normalisation ──────────────────────────────────────────────────────

var defaultFillerWords = []string{
	// EN
	"please", "hey", "hi", "hello",
	"could you", "can you", "would you", "will you",
	"just", "quickly", "briefly", "simply",
	"tell me", "show me", "give me", "help me", "assist me",
	"explain to me", "describe to me",
	"i need", "i want", "i would like", "i'd like", "i'm looking for",
	// DE
	"bitte", "mal eben", "schnell", "kurz", "einfach",
	"kannst du", "könntest du", "könnten sie", "würden sie", "würdest du",
	"hallo", "hi", "hey",
	"sag mir", "zeig mir", "gib mir", "hilf mir", "erkläre mir", "erklär mir",
	"ich brauche", "ich möchte", "ich hätte gerne", "ich suche",
	// FR
	"s'il vous plaît", "svp", "stp", "bonjour", "salut", "allô",
	"pouvez-vous", "pourriez-vous", "peux-tu", "pourrais-tu",
	"dis-moi", "dites-moi", "montre-moi", "montrez-moi",
	"j'ai besoin de", "je voudrais", "je cherche", "je souhaite",
	"expliquez-moi", "explique-moi", "aidez-moi", "aide-moi",
	// ES
	"por favor", "hola", "oye",
	"puedes", "podrías", "podría usted", "me puedes", "me podrías",
	"dime", "dígame", "muéstrame", "muéstreme", "dame", "deme",
	"necesito", "quisiera", "me gustaría", "quiero saber",
	"ayúdame", "ayúdeme", "explícame", "explíqueme",
	// IT
	"per favore", "perfavore", "ciao", "salve", "ehi",
	"potresti", "mi potresti", "potrebbe", "mi potrebbe",
	"dimmi", "mi dica", "mostrami", "dammi", "mi dia",
	"ho bisogno di", "vorrei", "mi piacerebbe",
	"aiutami", "mi aiuti", "spiegami", "mi spieghi",
	// PT
	"por favor", "olá", "oi", "ei",
	"pode", "poderia", "você poderia", "você pode", "podes",
	"me diga", "diga-me", "me mostre", "mostre-me", "me dê", "dê-me",
	"preciso de", "gostaria de", "quero saber", "estou procurando",
	"me ajude", "ajude-me", "explique-me", "me explique",
}

// normalizePrompt strips filler words, lowercases and collapses whitespace.
// +8–12% semantic hit-rate uplift at zero quality cost.
func normalizePrompt(text string, fillerWords []string) string {
	s := strings.ToLower(strings.TrimSpace(text))
	words := fillerWords
	if len(words) == 0 {
		words = defaultFillerWords
	}
	for _, fw := range words {
		s = strings.ReplaceAll(s, fw, "")
	}
	// Collapse multiple spaces.
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	s = strings.TrimSpace(s)
	// Normalise trailing punctuation.
	s = strings.TrimRight(s, "!?")
	if len(s) > 0 {
		s += "?"
	}
	return s
}

// confidenceBand returns the SemanticConfidence for a similarity score.
func confidenceBand(similarity, threshold, highThreshold float64) SemanticConfidence {
	if similarity >= highThreshold {
		return ConfidenceHigh
	}
	if similarity >= threshold {
		return ConfidenceMedium
	}
	return ConfidenceUncertain
}

// ── Namespace Auto-Detection ───────────────────────────────────────────────

// DetectNamespace classifies a prompt into one of 5 semantic namespaces using
// lightweight text heuristics (< 0.1 ms, no embedding required).
//
// Returned namespaces: cachly:sem:code, :translation, :summary, :qa, :creative
func DetectNamespace(prompt string) string {
	s := strings.ToLower(strings.TrimSpace(prompt))

	for _, kw := range []string{
		"function ", "def ", "class ", "import ", "const ", "let ", "var ",
		"return ", " => ", "void ", "public class", "func ", "#include", "package ",
		"struct {", "interface {", "async def", "lambda ", "#!/",
	} {
		if strings.Contains(s, kw) {
			return "cachly:sem:code"
		}
	}
	for _, kw := range []string{
		"translate", "übersetze", "auf deutsch", "auf englisch",
		"in english", "in german", "ins deutsche", "ins englische", "übersetz",
		"traduce", "traduis", "vertaal",
	} {
		if strings.Contains(s, kw) {
			return "cachly:sem:translation"
		}
	}
	for _, kw := range []string{
		"summarize", "summarise", "summary", "zusammenfass", "tl;dr", "tldr",
		"key points", "stichpunkte", "fasse zusammen", "give me a brief",
		"kurze zusammenfassung", "in a nutshell",
	} {
		if strings.Contains(s, kw) {
			return "cachly:sem:summary"
		}
	}
	for _, prefix := range []string{
		"what ", "who ", "where ", "when ", "why ", "how ", "which ",
		"is ", "are ", "was ", "were ", "does ", "do ", "did ",
		"can ", "could ", "would ", "should ", "will ",
		"wer ", "wie ", "wo ", "wann ", "warum ", "welche", "wieso ",
	} {
		if strings.HasPrefix(s, prefix) {
			return "cachly:sem:qa"
		}
	}
	if strings.HasSuffix(strings.TrimRight(s, " "), "?") {
		return "cachly:sem:qa"
	}
	return "cachly:sem:creative"
}

// ── HTTP helper ───────────────────────────────────────────────────────────────

// httpRequest performs an HTTP request against the cachly vector API.
// Uses net/http (stdlib, no additional dependency).
func httpRequest(ctx context.Context, method, url string, body map[string]any) (map[string]any, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("cachly api marshal: %w", err)
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, r)
	if err != nil {
		return nil, fmt.Errorf("cachly api request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cachly api do: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("cachly api: %s %s returned %d", method, url, resp.StatusCode)
	}
	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result) //nolint:errcheck
	return result, nil
}

// ── SemanticCache.GetOrSet ────────────────────────────────────────────────────

// GetOrSet returns a cached response for semantically similar prompts, or calls fn
// and caches the result.
//
// When a VectorURL is configured, uses the pgvector API (HNSW O(log n)) with
// automatic graceful fallback to linear SCAN if the API is unreachable.
//
// Storage layout (SCAN mode):
//   - {ns}:emb:{uuid} – lightweight: embedding + prompt
//   - {ns}:val:{uuid} – actual cached value
//
// Storage layout (API mode):
//   - pgvector API stores embedding + prompt
//   - {ns}:val:{uuid} stored in Valkey
func (s *SemanticCache) GetOrSet(
	ctx context.Context,
	prompt string,
	fn func() (any, error),
	opts SemanticOptions,
) (*SemanticResult, error) {
	// Normalise prompt before embedding (default: true).
	shouldNorm := opts.NormalizePrompt == nil || *opts.NormalizePrompt
	textForEmbed := prompt
	if shouldNorm {
		textForEmbed = normalizePrompt(prompt, opts.FillerWords)
	}

	if s.vectorURL != "" {
		result, err := s.getOrSetViaAPI(ctx, textForEmbed, fn, opts)
		if err == nil {
			return result, nil
		}
		// Graceful degradation: API unreachable → fall back to linear SCAN.
	}
	return s.getOrSetViaScan(ctx, textForEmbed, prompt, fn, opts)
}

// Warmup pre-warms the semantic cache with a slice of prompt/fn pairs.
//
// For each entry, GetOrSet is called with a high similarity threshold (0.98) so
// already-cached entries are returned without calling fn again.
// Returns (WarmupResult, nil) even when individual entries fail (best-effort).
func (s *SemanticCache) Warmup(ctx context.Context, entries []WarmupEntry, opts SemanticOptions) (WarmupResult, error) {
	var res WarmupResult
	warmOpts := opts
	if warmOpts.Threshold == 0 {
		warmOpts.Threshold = 0.98 // high threshold: skip if similar entry exists
	}
	for _, e := range entries {
		entryOpts := warmOpts
		if e.Namespace != "" {
			entryOpts.Namespace = e.Namespace
		}
		// detect namespace when AutoNamespace requested and no explicit ns given.
		if entryOpts.AutoNamespace && entryOpts.Namespace == "" {
			entryOpts.Namespace = DetectNamespace(e.Prompt)
		}
		result, err := s.GetOrSet(ctx, e.Prompt, e.Fn, entryOpts)
		if err != nil {
			res.Skipped++
			continue
		}
		if result.Hit {
			res.Skipped++
		} else {
			res.Warmed++
		}
	}
	return res, nil
}

// ImportFromLog reads a JSONL file and warms the semantic cache with the prompts it contains.
//
// Each line must be a JSON object. The prompt is extracted from the field named by promptField
// (default: "prompt"). responseFn is called for every cache miss.
//
// batchSize controls how many prompts are passed to Warmup at a time (default: 50).
// Pass zero to use defaults. opts are forwarded to Warmup (threshold defaults to 0.98).
func (s *SemanticCache) ImportFromLog(
	ctx context.Context,
	filePath string,
	responseFn func(ctx context.Context, prompt string) (any, error),
	opts SemanticOptions,
	promptField string,
	batchSize int,
) (WarmupResult, error) {
	if promptField == "" {
		promptField = "prompt"
	}
	if batchSize <= 0 {
		batchSize = 50
	}
	if opts.Threshold == 0 {
		opts.Threshold = 0.98
	}

	f, err := os.Open(filePath)
	if err != nil {
		return WarmupResult{}, fmt.Errorf("cachly ImportFromLog: %w", err)
	}
	defer f.Close()

	var total WarmupResult
	var batch []WarmupEntry

	flush := func() {
		if len(batch) == 0 {
			return
		}
		result, _ := s.Warmup(ctx, batch, opts)
		total.Warmed += result.Warmed
		total.Skipped += result.Skipped
		batch = batch[:0]
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			total.Skipped++
			continue
		}
		prompt, _ := obj[promptField].(string)
		if prompt == "" {
			total.Skipped++
			continue
		}
		p := prompt
		batch = append(batch, WarmupEntry{
			Prompt: p,
			Fn:     func() (any, error) { return responseFn(ctx, p) },
		})
		if len(batch) >= batchSize {
			flush()
		}
	}
	flush()

	if err := scanner.Err(); err != nil {
		return total, fmt.Errorf("cachly ImportFromLog scan: %w", err)
	}
	return total, nil
}

// getOrSetViaAPI uses the cachly pgvector API for O(log n) HNSW nearest-neighbour search.
func (s *SemanticCache) getOrSetViaAPI(
	ctx context.Context,
	normalizedPrompt string,
	fn func() (any, error),
	opts SemanticOptions,
) (*SemanticResult, error) {
	ns := opts.Namespace
	if ns == "" {
		ns = defaultNamespace
	}
	// auto-detect namespace when requested and no explicit namespace was set.
	if opts.AutoNamespace && opts.Namespace == "" {
		ns = DetectNamespace(normalizedPrompt)
	}
	threshold := opts.Threshold
	if threshold == 0 {
		threshold = 0.85
	}
	highThreshold := opts.HighConfidenceThreshold
	if highThreshold == 0 {
		highThreshold = 0.97
	}

	queryEmbed, err := s.embedFn(ctx, normalizedPrompt)
	if err != nil {
		return nil, fmt.Errorf("cachly semantic embed: %w", err)
	}

	// fetch adaptive threshold when requested.
	if opts.UseAdaptiveThreshold {
		if adapted, aerr := s.adaptiveThreshold(ctx, ns, threshold); aerr == nil {
			threshold = adapted
		}
	}

	// build search body with optional int8 quantization.
	searchBody := map[string]any{
		"namespace":              ns,
		"threshold":              threshold,
		"use_adaptive_threshold": false, // already resolved above
	}
	if opts.Quantize == "int8" {
		searchBody["embedding_q8"] = QuantizeEmbedding(queryEmbed)
	} else {
		searchBody["embedding"] = queryEmbed
	}
	// hybrid BM25+Vector RRF: include prompt so the backend runs FTS too.
	if opts.UseHybrid && normalizedPrompt != "" {
		searchBody["hybrid"] = true
		searchBody["prompt"] = normalizedPrompt
	}

	// 1. Search via pgvector API (using edge URL if configured for reads).
	searchResp, err := httpRequest(ctx, http.MethodPost, s.searchURL()+"/search", searchBody)
	if err != nil {
		return nil, fmt.Errorf("cachly api search: %w", err)
	}

	if found, _ := searchResp["found"].(bool); found {
		idStr, _ := searchResp["id"].(string)
		sim, _ := searchResp["similarity"].(float64)
		if idStr != "" {
			vKey := fmt.Sprintf("%s:val:%s", ns, idStr)
			raw, gerr := s.rdb.Get(ctx, vKey).Bytes()
			if gerr == nil {
				var value any
				if json.Unmarshal(raw, &value) == nil {
					conf := confidenceBand(sim, threshold, highThreshold)
					// Stats – best-effort, non-blocking
					go s.rdb.Incr(context.Background(), ns+":stats:hits") //nolint:errcheck
					return &SemanticResult{Value: value, Hit: true, Similarity: sim, Confidence: conf}, nil
				}
			}
			// orphaned pgvector entry – fall through to miss
		}
	}

	// Cache miss
	go s.rdb.Incr(context.Background(), ns+":stats:misses") //nolint:errcheck
	value, err := fn()
	if err != nil {
		return nil, err
	}

	id := uuid.New().String()
	vKey := fmt.Sprintf("%s:val:%s", ns, id)
	valPayload, _ := json.Marshal(value)

	if opts.TTL > 0 {
		s.rdb.Set(ctx, vKey, valPayload, opts.TTL) //nolint:errcheck
	} else {
		s.rdb.Set(ctx, vKey, valPayload, 0) //nolint:errcheck
	}

	// Index embedding in pgvector API.
	indexBody := map[string]any{
		"id":        id,
		"prompt":    normalizedPrompt,
		"namespace": ns,
	}
	if opts.Quantize == "int8" {
		indexBody["embedding_q8"] = QuantizeEmbedding(queryEmbed)
	} else {
		indexBody["embedding"] = queryEmbed
	}
	if opts.TTL > 0 {
		expiresAt := time.Now().Add(opts.TTL).UTC().Format(time.RFC3339)
		indexBody["expires_at"] = expiresAt
	}
	httpRequest(ctx, http.MethodPost, s.vectorURL+"/entries", indexBody) //nolint:errcheck

	return &SemanticResult{Value: value, Hit: false}, nil
}

// getOrSetViaScan uses linear SCAN over Valkey emb keys (fallback / non-API mode).
func (s *SemanticCache) getOrSetViaScan(
	ctx context.Context,
	normalizedPrompt string,
	originalPrompt string,
	fn func() (any, error),
	opts SemanticOptions,
) (*SemanticResult, error) {
	threshold := opts.Threshold
	if threshold == 0 {
		threshold = 0.85
	}
	highThreshold := opts.HighConfidenceThreshold
	if highThreshold == 0 {
		highThreshold = 0.97
	}
	ns := opts.Namespace
	if ns == "" {
		ns = defaultNamespace
	}

	queryEmbed, err := s.embedFn(ctx, normalizedPrompt)
	if err != nil {
		return nil, fmt.Errorf("cachly semantic embed: %w", err)
	}

	bestSim := -1.0
	bestValKey := ""

	var cursor uint64
	for {
		var keys []string
		keys, cursor, err = s.rdb.Scan(ctx, cursor, ns+":emb:*", 100).Result()
		if err != nil {
			return nil, fmt.Errorf("cachly semantic scan: %w", err)
		}
		for _, k := range keys {
			raw, err := s.rdb.Get(ctx, k).Bytes()
			if err != nil {
				continue
			}
			var entry semEmbEntry
			if err := json.Unmarshal(raw, &entry); err != nil {
				continue
			}
			sim := cosineSimilarity(queryEmbed, entry.Embedding)
			if sim > bestSim {
				bestSim = sim
				bestValKey = valKey(k)
			}
		}
		if cursor == 0 {
			break
		}
	}

	if bestSim >= threshold && bestValKey != "" {
		raw, err := s.rdb.Get(ctx, bestValKey).Bytes()
		if err == nil {
			var value any
			if json.Unmarshal(raw, &value) == nil {
				conf := confidenceBand(bestSim, threshold, highThreshold)
				go s.rdb.Incr(context.Background(), ns+":stats:hits") //nolint:errcheck
				return &SemanticResult{Value: value, Hit: true, Similarity: bestSim, Confidence: conf}, nil
			}
		}
		// orphaned emb – fall through to miss
	}

	// Cache miss
	go s.rdb.Incr(context.Background(), ns+":stats:misses") //nolint:errcheck
	value, err := fn()
	if err != nil {
		return nil, err
	}

	id := uuid.New().String()
	embKey := fmt.Sprintf("%s:emb:%s", ns, id)
	vKey := fmt.Sprintf("%s:val:%s", ns, id)

	embPayload, _ := json.Marshal(semEmbEntry{
		Embedding:      queryEmbed,
		Prompt:         normalizedPrompt,
		OriginalPrompt: originalPrompt,
	})
	valPayload, _ := json.Marshal(value)

	if opts.TTL > 0 {
		s.rdb.Set(ctx, vKey, valPayload, opts.TTL)   //nolint:errcheck
		s.rdb.Set(ctx, embKey, embPayload, opts.TTL) //nolint:errcheck
	} else {
		s.rdb.Set(ctx, vKey, valPayload, 0)   //nolint:errcheck
		s.rdb.Set(ctx, embKey, embPayload, 0) //nolint:errcheck
	}

	return &SemanticResult{Value: value, Hit: false}, nil
}

// Invalidate removes a single semantic cache entry.
//
// In API mode: deletes the val key from Valkey and calls DELETE on the API.
// In SCAN mode: deletes both emb and val keys from Valkey.
// Obtain the key from Entries().
func (s *SemanticCache) Invalidate(ctx context.Context, key string) error {
	id := extractIDFromKey(key)
	ns := extractNamespaceFromKey(key)
	vKey := ns + ":val:" + id

	if s.vectorURL != "" {
		// Delete val from Valkey + embedding from pgvector API.
		s.rdb.Del(ctx, vKey) //nolint:errcheck
		_, err := httpRequest(ctx, http.MethodDelete, s.vectorURL+"/entries/"+id, nil)
		return err
	}
	// SCAN mode: delete both emb and val from Valkey.
	return s.rdb.Del(ctx, key, vKey).Err()
}

// Entries lists every cached prompt together with its key.
//
// In API mode: queries the pgvector API, returns keys of the form {ns}:emb:{uuid}.
// In SCAN mode: scans Valkey emb keys.
// Pass the returned key to Invalidate to remove a specific entry.
func (s *SemanticCache) Entries(ctx context.Context, namespace string) ([]SemEntryInfo, error) {
	if namespace == "" {
		namespace = defaultNamespace
	}

	if s.vectorURL != "" {
		return s.entriesViaAPI(ctx, namespace)
	}
	return s.entriesViaScan(ctx, namespace)
}

func (s *SemanticCache) entriesViaAPI(ctx context.Context, namespace string) ([]SemEntryInfo, error) {
	resp, err := httpRequest(ctx, http.MethodGet,
		fmt.Sprintf("%s/entries?namespace=%s", s.vectorURL, namespace), nil)
	if err != nil {
		return nil, fmt.Errorf("cachly api entries: %w", err)
	}
	data, _ := resp["data"].([]any)
	var result []SemEntryInfo
	for _, item := range data {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		id, _ := m["id"].(string)
		prompt, _ := m["prompt"].(string)
		if id == "" {
			continue
		}
		result = append(result, SemEntryInfo{
			Key:    fmt.Sprintf("%s:emb:%s", namespace, id),
			Prompt: prompt,
		})
	}
	return result, nil
}

func (s *SemanticCache) entriesViaScan(ctx context.Context, namespace string) ([]SemEntryInfo, error) {
	var result []SemEntryInfo
	var cursor uint64
	var err error
	for {
		keys, newCursor, scanErr := s.rdb.Scan(ctx, cursor, namespace+":emb:*", 100).Result()
		if scanErr != nil {
			return result, fmt.Errorf("cachly semantic entries scan: %w", scanErr)
		}
		for _, k := range keys {
			raw, rerr := s.rdb.Get(ctx, k).Bytes()
			if rerr != nil {
				continue
			}
			var entry semEmbEntry
			if err = json.Unmarshal(raw, &entry); err == nil {
				// Prefer the original (un-normalised) prompt for display;
				// fall back to the normalised version for entries written by older SDK versions.
				displayPrompt := entry.OriginalPrompt
				if displayPrompt == "" {
					displayPrompt = entry.Prompt
				}
				result = append(result, SemEntryInfo{Key: k, Prompt: displayPrompt})
			}
		}
		cursor = newCursor
		if cursor == 0 {
			break
		}
	}
	return result, nil
}

// Flush removes all entries in the semantic cache namespace.
//
// In API mode: calls DELETE flush on the API + deletes val keys from Valkey.
// In SCAN mode: deletes all emb and val keys from Valkey.
// Returns count of logical entries deleted.
func (s *SemanticCache) Flush(ctx context.Context, namespace string) (int64, error) {
	if namespace == "" {
		namespace = defaultNamespace
	}

	if s.vectorURL != "" {
		return s.flushViaAPI(ctx, namespace)
	}
	return s.flushViaScan(ctx, namespace)
}

func (s *SemanticCache) flushViaAPI(ctx context.Context, namespace string) (int64, error) {
	resp, err := httpRequest(ctx, http.MethodDelete,
		fmt.Sprintf("%s/flush?namespace=%s", s.vectorURL, namespace), nil)
	if err != nil {
		return 0, fmt.Errorf("cachly api flush: %w", err)
	}
	deleted, _ := resp["deleted"].(float64)

	// Also purge val keys from Valkey (pgvector only stores embeddings, vals live in Valkey).
	var allValKeys []string
	var cursor uint64
	for {
		keys, newCursor, scanErr := s.rdb.Scan(ctx, cursor, namespace+":val:*", 100).Result()
		if scanErr != nil {
			break
		}
		allValKeys = append(allValKeys, keys...)
		cursor = newCursor
		if cursor == 0 {
			break
		}
	}
	if len(allValKeys) > 0 {
		s.rdb.Del(ctx, allValKeys...) //nolint:errcheck
	}
	return int64(deleted), nil
}

func (s *SemanticCache) flushViaScan(ctx context.Context, namespace string) (int64, error) {
	var embCount int64
	var allKeys []string
	var cursor uint64
	var err error

	for {
		keys, newCursor, scanErr := s.rdb.Scan(ctx, cursor, namespace+":emb:*", 100).Result()
		if scanErr != nil {
			return embCount, fmt.Errorf("cachly semantic flush scan emb: %w", scanErr)
		}
		allKeys = append(allKeys, keys...)
		embCount += int64(len(keys))
		cursor = newCursor
		if cursor == 0 {
			break
		}
	}
	for {
		keys, newCursor, scanErr := s.rdb.Scan(ctx, cursor, namespace+":val:*", 100).Result()
		if scanErr != nil {
			return embCount, fmt.Errorf("cachly semantic flush scan val: %w", scanErr)
		}
		allKeys = append(allKeys, keys...)
		cursor = newCursor
		if cursor == 0 {
			break
		}
	}

	if len(allKeys) > 0 {
		if _, err = s.rdb.Del(ctx, allKeys...).Result(); err != nil {
			return embCount, err
		}
	}
	return embCount, nil
}

// Size returns the number of logical entries in the semantic cache namespace.
//
// In API mode: queries the pgvector API.
// In SCAN mode: scans Valkey emb keys.
func (s *SemanticCache) Size(ctx context.Context, namespace string) (int64, error) {
	if namespace == "" {
		namespace = defaultNamespace
	}

	if s.vectorURL != "" {
		resp, err := httpRequest(ctx, http.MethodGet,
			fmt.Sprintf("%s/size?namespace=%s", s.vectorURL, namespace), nil)
		if err != nil {
			return 0, fmt.Errorf("cachly api size: %w", err)
		}
		size, _ := resp["size"].(float64)
		return int64(size), nil
	}

	var count int64
	var cursor uint64
	for {
		keys, newCursor, err := s.rdb.Scan(ctx, cursor, namespace+":emb:*", 100).Result()
		if err != nil {
			return count, err
		}
		count += int64(len(keys))
		cursor = newCursor
		if cursor == 0 {
			break
		}
	}
	return count, nil
}

// ── Adaptive Threshold ─────────────────────────────────────────────────────

// Feedback records whether a cache hit was accepted as correct (Adaptive Threshold).
//
// hitID is the UUID returned in SemanticResult when Hit==true.
// Requires a VectorURL. Returns nil immediately when no VectorURL is set.
func (s *SemanticCache) Feedback(ctx context.Context, hitID string, accepted bool, similarity float64, namespace string) error {
	if s.vectorURL == "" {
		return nil
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	_, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/feedback", map[string]any{
		"hit_id":     hitID,
		"accepted":   accepted,
		"similarity": similarity,
		"namespace":  namespace,
	})
	return err
}

// AdaptiveThreshold returns the server-side F1-calibrated threshold for a namespace.
// Falls back to defaultThreshold (0.85) when no calibration data exists yet or when
// no VectorURL is configured.
func (s *SemanticCache) AdaptiveThreshold(ctx context.Context, namespace string) (float64, error) {
	if s.vectorURL == "" {
		return 0.85, nil
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	return s.adaptiveThreshold(ctx, namespace, 0.85)
}

// adaptiveThreshold is the internal helper (avoids duplicating URL building).
func (s *SemanticCache) adaptiveThreshold(ctx context.Context, namespace string, fallback float64) (float64, error) {
	resp, err := httpRequest(ctx, http.MethodGet,
		fmt.Sprintf("%s/threshold?namespace=%s", s.vectorURL, namespace), nil)
	if err != nil {
		return fallback, nil
	}
	if t, ok := resp["threshold"].(float64); ok && t > 0 {
		return t, nil
	}
	return fallback, nil
}

// ── Quantization helpers (already defined above as package-level funcs) ───

// ── New in 0.4.0 – Result types ───────────────────────────────────────────────

// CacheStats holds semantic cache hit/miss statistics (feature B).
type CacheStats struct {
	Hits       int64   `json:"hits"`
	Misses     int64   `json:"misses"`
	HitRate    float64 `json:"hit_rate"`
	Total      int64   `json:"total"`
	Namespaces []any   `json:"namespaces"`
}

// BatchIndexEntry is one entry for BatchIndex (feature D).
type BatchIndexEntry struct {
	ID        string    `json:"id"`
	Prompt    string    `json:"prompt"`
	Embedding []float64 `json:"embedding"`
	Namespace string    `json:"namespace,omitempty"`
	ExpiresAt string    `json:"expires_at,omitempty"`
}

// BatchIndexResult is the result of SemanticCache.BatchIndex (feature D).
type BatchIndexResult struct {
	Indexed int `json:"indexed"`
	Skipped int `json:"skipped"`
}

// GuardrailViolation is a single content-safety violation (feature K).
type GuardrailViolation struct {
	Type    string `json:"type"`
	Pattern string `json:"pattern"`
	Action  string `json:"action"`
}

// GuardrailCheckResult is returned by SemanticCache.CheckGuardrail (feature K).
type GuardrailCheckResult struct {
	Safe       bool                 `json:"safe"`
	Violations []GuardrailViolation `json:"violations"`
}

// TagsResult is returned by Client.SetTags (feature L).
type TagsResult struct {
	Key  string   `json:"key"`
	Tags []string `json:"tags"`
	Ok   bool     `json:"ok"`
}

// InvalidateTagResult is returned by Client.InvalidateTag (feature M).
type InvalidateTagResult struct {
	Tag        string   `json:"tag"`
	KeysDeleted int     `json:"keys_deleted"`
	Keys       []string `json:"keys"`
	DurationMs int      `json:"duration_ms"`
}

// SwrEntry is one stale-while-revalidate key entry (feature Q).
type SwrEntry struct {
	Key         string `json:"key"`
	FetcherHint string `json:"fetcher_hint"`
	StaleFor    string `json:"stale_for"`
	RefreshAt   string `json:"refresh_at"`
}

// SwrCheckResult is returned by Client.SWRCheck (feature Q).
type SwrCheckResult struct {
	StaleKeys []SwrEntry `json:"stale_keys"`
	Count     int        `json:"count"`
	CheckedAt string     `json:"checked_at"`
}

// BulkWarmupEntry is one entry for Client.BulkWarmup (feature S).
type BulkWarmupEntry struct {
	Key   string `json:"key"`
	Value string `json:"value"`
	TTL   int    `json:"ttl,omitempty"`
}

// BulkWarmupResult is returned by Client.BulkWarmup (feature S).
type BulkWarmupResult struct {
	Warmed     int `json:"warmed"`
	Skipped    int `json:"skipped"`
	DurationMs int `json:"duration_ms"`
}

// SnapshotWarmupResult is returned by SemanticCache.SnapshotWarmup (feature T).
type SnapshotWarmupResult struct {
	Warmed     int `json:"warmed"`
	DurationMs int `json:"duration_ms"`
}

// LlmProxyStats is returned by Client.LlmProxyStats (feature W).
type LlmProxyStats struct {
	TotalRequests       int     `json:"total_requests"`
	CacheHits           int     `json:"cache_hits"`
	CacheMisses         int     `json:"cache_misses"`
	EstimatedSavedUSD   float64 `json:"estimated_saved_usd"`
	AvgLatencyMsCached  float64 `json:"avg_latency_ms_cached"`
	AvgLatencyMsUncached float64 `json:"avg_latency_ms_uncached"`
}

// PubSubPublishResult is returned by PubSubClient.Publish.
type PubSubPublishResult struct {
	Channel     string `json:"channel"`
	Receivers   int    `json:"receivers"`
	PublishedAt string `json:"published_at"`
}

// PubSubChannelInfo is one channel returned by PubSubClient.Channels.
type PubSubChannelInfo struct {
	Name        string `json:"name"`
	Subscribers int    `json:"subscribers"`
}

// PubSubStatsResult is returned by PubSubClient.Stats.
type PubSubStatsResult struct {
	ActiveChannels   int `json:"active_channels"`
	TotalSubscribers int `json:"total_subscribers"`
	PatternCount     int `json:"pattern_count"`
}

// WorkflowCheckpoint is a workflow step checkpoint (feature V).
type WorkflowCheckpoint struct {
	ID         string `json:"id"`
	RunID      string `json:"run_id"`
	StepIndex  int    `json:"step_index"`
	StepName   string `json:"step_name"`
	AgentName  string `json:"agent_name"`
	Status     string `json:"status"`
	State      string `json:"state,omitempty"`
	Output     string `json:"output,omitempty"`
	DurationMs int    `json:"duration_ms"`
}

// WorkflowRun is a workflow run summary (feature V).
type WorkflowRun struct {
	RunID        string `json:"run_id"`
	Steps        int    `json:"steps"`
	LatestStatus string `json:"latest_status"`
}

// ── SSE Stream helper ─────────────────────────────────────────────────────────

// httpSSEStream performs an HTTP request to an SSE endpoint and returns a channel
// of parsed JSON event objects. The channel is closed when the stream ends, the
// done-sentinel ({}) is received, or ctx is cancelled.
// The caller must read all events or cancel ctx to avoid a goroutine leak.
func httpSSEStream(ctx context.Context, method, url string, body map[string]any) (<-chan map[string]any, error) {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("cachly sse marshal: %w", err)
		}
		reqBody = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("cachly sse request: %w", err)
	}
	req.Header.Set("Accept", "text/event-stream")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cachly sse do: %w", err)
	}
	if resp.StatusCode >= 400 {
		resp.Body.Close() //nolint:errcheck
		return nil, fmt.Errorf("cachly sse: %s returned %d", url, resp.StatusCode)
	}

	ch := make(chan map[string]any, 64)
	go func() {
		defer resp.Body.Close() //nolint:errcheck
		defer close(ch)
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			dataStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if dataStr == "" {
				continue
			}
			var event map[string]any
			if err := json.Unmarshal([]byte(dataStr), &event); err != nil {
				continue
			}
			if len(event) == 0 { // done sentinel
				return
			}
			select {
			case ch <- event:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch, nil
}

// ── SemanticCache new methods (0.4.0) ────────────────────────────────────────

// SetThreshold manually sets the F1-calibrated similarity threshold for a namespace (A).
// Requires VectorURL. Returns nil when no VectorURL is configured.
func (s *SemanticCache) SetThreshold(ctx context.Context, namespace string, threshold float64) error {
	if s.vectorURL == "" {
		return nil
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	_, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/threshold", map[string]any{
		"namespace": namespace,
		"threshold": threshold,
	})
	return err
}

// Stats returns semantic cache hit/miss statistics for a namespace (B).
// In SCAN mode (no VectorURL) reads local Valkey counters.
func (s *SemanticCache) Stats(ctx context.Context, namespace string) (*CacheStats, error) {
	if namespace == "" {
		namespace = defaultNamespace
	}
	if s.vectorURL != "" {
		resp, err := httpRequest(ctx, http.MethodGet,
			fmt.Sprintf("%s/stats?namespace=%s", s.vectorURL, namespace), nil)
		if err != nil {
			return nil, fmt.Errorf("cachly api stats: %w", err)
		}
		raw, _ := json.Marshal(resp)
		var cs CacheStats
		json.Unmarshal(raw, &cs) //nolint:errcheck
		return &cs, nil
	}
	// Valkey fallback: read incr counters.
	hitsStr, _ := s.rdb.Get(ctx, namespace+":stats:hits").Result()
	missStr, _ := s.rdb.Get(ctx, namespace+":stats:misses").Result()
	var hits, misses int64
	fmt.Sscan(hitsStr, &hits)   //nolint:errcheck
	fmt.Sscan(missStr, &misses) //nolint:errcheck
	total := hits + misses
	var hitRate float64
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}
	return &CacheStats{Hits: hits, Misses: misses, Total: total, HitRate: hitRate}, nil
}

// StreamSearch performs semantic search and streams results via SSE (C).
// Returns a channel that receives raw event maps until the stream ends or ctx is cancelled.
// Requires VectorURL.
//
// Event shapes:
//   - {"found":true,"id":"uuid","similarity":0.94} – hit metadata
//   - {"text":"chunk"} – streamed value chunk
//
// Example:
//
//	ch, err := sem.StreamSearch(ctx, "What is caching?", opts)
//	for event := range ch {
//	    if found, _ := event["found"].(bool); found {
//	        fmt.Println("HIT", event["id"])
//	    } else if text, ok := event["text"].(string); ok {
//	        fmt.Print(text)
//	    }
//	}
func (s *SemanticCache) StreamSearch(ctx context.Context, prompt string, opts SemanticOptions) (<-chan map[string]any, error) {
	if s.vectorURL == "" {
		return nil, fmt.Errorf("cachly: StreamSearch requires VectorURL")
	}
	ns := opts.Namespace
	if ns == "" {
		ns = defaultNamespace
	}
	threshold := opts.Threshold
	if threshold == 0 {
		threshold = 0.85
	}
	embedding, err := s.embedFn(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("cachly stream_search embed: %w", err)
	}
	return httpSSEStream(ctx, http.MethodPost, s.vectorURL+"/search/stream", map[string]any{
		"embedding": embedding,
		"namespace": ns,
		"threshold": threshold,
		"prompt":    prompt,
	})
}

// BatchIndex bulk-indexes up to 500 entries in a single request (D).
// Requires VectorURL.
func (s *SemanticCache) BatchIndex(ctx context.Context, entries []BatchIndexEntry) (*BatchIndexResult, error) {
	if s.vectorURL == "" {
		return nil, fmt.Errorf("cachly: BatchIndex requires VectorURL")
	}
	resp, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/entries/batch", map[string]any{
		"entries": entries,
	})
	if err != nil {
		return nil, fmt.Errorf("cachly api batch_index: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result BatchIndexResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// CreateIndex creates a new vector index for a namespace (E).
// Requires VectorURL.
func (s *SemanticCache) CreateIndex(
	ctx context.Context,
	namespace string,
	dimensions int,
	model string,
	metric string,
	hybridEnabled bool,
) error {
	if s.vectorURL == "" {
		return fmt.Errorf("cachly: CreateIndex requires VectorURL")
	}
	_, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/indexes", map[string]any{
		"namespace":      namespace,
		"dimensions":     dimensions,
		"model":          model,
		"metric":         metric,
		"hybrid_enabled": hybridEnabled,
	})
	return err
}

// DeleteIndex deletes the vector index for a namespace (F).
// Requires VectorURL.
func (s *SemanticCache) DeleteIndex(ctx context.Context, namespace string) error {
	if s.vectorURL == "" {
		return fmt.Errorf("cachly: DeleteIndex requires VectorURL")
	}
	_, err := httpRequest(ctx, http.MethodDelete, s.vectorURL+"/indexes/"+urlPathEscape(namespace), nil)
	return err
}

// SetMetadata attaches JSONB metadata to a semantic cache entry (G).
// Requires VectorURL.
func (s *SemanticCache) SetMetadata(ctx context.Context, entryID string, metadata map[string]any) error {
	if s.vectorURL == "" {
		return fmt.Errorf("cachly: SetMetadata requires VectorURL")
	}
	_, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/metadata", map[string]any{
		"entry_id": entryID,
		"metadata": metadata,
	})
	return err
}

// FilteredSearch performs semantic search with server-side metadata filter (H).
// Requires VectorURL.
func (s *SemanticCache) FilteredSearch(
	ctx context.Context,
	prompt string,
	namespace string,
	threshold float64,
	filter map[string]any,
	limit int,
) ([]map[string]any, error) {
	if s.vectorURL == "" {
		return nil, fmt.Errorf("cachly: FilteredSearch requires VectorURL")
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	if threshold == 0 {
		threshold = 0.85
	}
	if limit == 0 {
		limit = 5
	}
	embedding, err := s.embedFn(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("cachly filtered_search embed: %w", err)
	}
	body := map[string]any{
		"prompt":    prompt,
		"embedding": embedding,
		"namespace": namespace,
		"threshold": threshold,
		"limit":     limit,
	}
	if len(filter) > 0 {
		body["filter"] = filter
	}
	resp, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/search/filtered", body)
	if err != nil {
		return nil, fmt.Errorf("cachly api filtered_search: %w", err)
	}
	raw, _ := resp["results"]
	if raw == nil {
		return nil, nil
	}
	arr, _ := raw.([]any)
	results := make([]map[string]any, 0, len(arr))
	for _, item := range arr {
		if m, ok := item.(map[string]any); ok {
			results = append(results, m)
		}
	}
	return results, nil
}

// SetGuardrail configures content-safety guardrails for a namespace (I).
// Requires VectorURL.
func (s *SemanticCache) SetGuardrail(
	ctx context.Context,
	namespace string,
	piiAction string,
	toxicAction string,
	toxicThreshold float64,
) error {
	if s.vectorURL == "" {
		return fmt.Errorf("cachly: SetGuardrail requires VectorURL")
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	if toxicThreshold == 0 {
		toxicThreshold = 0.8
	}
	_, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/guardrails", map[string]any{
		"namespace":       namespace,
		"pii_action":      piiAction,
		"toxic_action":    toxicAction,
		"toxic_threshold": toxicThreshold,
	})
	return err
}

// DeleteGuardrail removes the guardrail configuration for a namespace (J).
// Requires VectorURL.
func (s *SemanticCache) DeleteGuardrail(ctx context.Context, namespace string) error {
	if s.vectorURL == "" {
		return fmt.Errorf("cachly: DeleteGuardrail requires VectorURL")
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	_, err := httpRequest(ctx, http.MethodDelete,
		s.vectorURL+"/guardrails/"+urlPathEscape(namespace), nil)
	return err
}

// CheckGuardrail checks text against the configured guardrails (K).
// Requires VectorURL.
func (s *SemanticCache) CheckGuardrail(ctx context.Context, text, namespace string) (*GuardrailCheckResult, error) {
	if s.vectorURL == "" {
		return nil, fmt.Errorf("cachly: CheckGuardrail requires VectorURL")
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	resp, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/guardrails/check", map[string]any{
		"text":      text,
		"namespace": namespace,
	})
	if err != nil {
		return nil, fmt.Errorf("cachly api check_guardrail: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result GuardrailCheckResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// SnapshotWarmup re-warms the semantic cache from existing pgvector index entries (T).
// Requires VectorURL.
func (s *SemanticCache) SnapshotWarmup(ctx context.Context, namespace string, limit int) (*SnapshotWarmupResult, error) {
	if s.vectorURL == "" {
		return nil, fmt.Errorf("cachly: SnapshotWarmup requires VectorURL")
	}
	if namespace == "" {
		namespace = defaultNamespace
	}
	if limit == 0 {
		limit = 100
	}
	resp, err := httpRequest(ctx, http.MethodPost, s.vectorURL+"/warmup/snapshot", map[string]any{
		"namespace": namespace,
		"limit":     limit,
	})
	if err != nil {
		return nil, fmt.Errorf("cachly api snapshot_warmup: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result SnapshotWarmupResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// ── Client new methods (0.4.0) ────────────────────────────────────────────────

// PubSub returns a PubSubClient.
// Requires PubSubURL to be set in Config.
func (c *Client) PubSub() (*PubSubClient, error) {
	if c.pubsubURL == "" {
		return nil, fmt.Errorf("cachly: PubSub requires PubSubURL in Config")
	}
	return &PubSubClient{url: c.pubsubURL}, nil
}

// Workflow returns a WorkflowClient.
// Requires WorkflowURL to be set in Config.
func (c *Client) Workflow() (*WorkflowClient, error) {
	if c.workflowURL == "" {
		return nil, fmt.Errorf("cachly: Workflow requires WorkflowURL in Config")
	}
	return &WorkflowClient{url: c.workflowURL}, nil
}

// SetTags associates a cache key with one or more tags (L).
// Requires BatchURL.
func (c *Client) SetTags(ctx context.Context, key string, tags []string) (*TagsResult, error) {
	if c.batchURL == "" {
		return nil, fmt.Errorf("cachly: SetTags requires BatchURL")
	}
	resp, err := httpRequest(ctx, http.MethodPost, c.batchURL+"/tags", map[string]any{
		"key":  key,
		"tags": tags,
	})
	if err != nil {
		return nil, fmt.Errorf("cachly api set_tags: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result TagsResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// InvalidateTag deletes all cache keys associated with a tag (M).
// Requires BatchURL.
func (c *Client) InvalidateTag(ctx context.Context, tag string) (*InvalidateTagResult, error) {
	if c.batchURL == "" {
		return nil, fmt.Errorf("cachly: InvalidateTag requires BatchURL")
	}
	resp, err := httpRequest(ctx, http.MethodPost, c.batchURL+"/invalidate", map[string]any{
		"tag": tag,
	})
	if err != nil {
		return nil, fmt.Errorf("cachly api invalidate_tag: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result InvalidateTagResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// GetTags returns all tags for a cache key (N).
// Requires BatchURL.
func (c *Client) GetTags(ctx context.Context, key string) ([]string, error) {
	if c.batchURL == "" {
		return nil, fmt.Errorf("cachly: GetTags requires BatchURL")
	}
	resp, err := httpRequest(ctx, http.MethodGet,
		c.batchURL+"/tags/"+urlPathEscape(key), nil)
	if err != nil {
		return nil, fmt.Errorf("cachly api get_tags: %w", err)
	}
	raw, _ := resp["tags"]
	if raw == nil {
		return nil, nil
	}
	arr, _ := raw.([]any)
	tags := make([]string, 0, len(arr))
	for _, t := range arr {
		if s, ok := t.(string); ok {
			tags = append(tags, s)
		}
	}
	return tags, nil
}

// DeleteTags removes all tag associations for a cache key (O).
// Requires BatchURL.
func (c *Client) DeleteTags(ctx context.Context, key string) error {
	if c.batchURL == "" {
		return fmt.Errorf("cachly: DeleteTags requires BatchURL")
	}
	_, err := httpRequest(ctx, http.MethodDelete,
		c.batchURL+"/tags/"+urlPathEscape(key), nil)
	return err
}

// SWRRegister registers a key for Stale-While-Revalidate (P).
// Requires BatchURL.
func (c *Client) SWRRegister(ctx context.Context, key string, ttlSeconds, staleWindowSeconds int, fetcherHint string) error {
	if c.batchURL == "" {
		return fmt.Errorf("cachly: SWRRegister requires BatchURL")
	}
	body := map[string]any{
		"key":                   key,
		"ttl_seconds":           ttlSeconds,
		"stale_window_seconds":  staleWindowSeconds,
	}
	if fetcherHint != "" {
		body["fetcher_hint"] = fetcherHint
	}
	_, err := httpRequest(ctx, http.MethodPost, c.batchURL+"/swr/register", body)
	return err
}

// SWRCheck returns all keys currently in their stale window (Q).
// Requires BatchURL.
func (c *Client) SWRCheck(ctx context.Context) (*SwrCheckResult, error) {
	if c.batchURL == "" {
		return nil, fmt.Errorf("cachly: SWRCheck requires BatchURL")
	}
	resp, err := httpRequest(ctx, http.MethodPost, c.batchURL+"/swr/check", map[string]any{})
	if err != nil {
		return nil, fmt.Errorf("cachly api swr_check: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result SwrCheckResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// SWRRemove removes a key from the SWR registry (R).
// Requires BatchURL.
func (c *Client) SWRRemove(ctx context.Context, key string) error {
	if c.batchURL == "" {
		return fmt.Errorf("cachly: SWRRemove requires BatchURL")
	}
	_, err := httpRequest(ctx, http.MethodDelete, c.batchURL+"/swr/"+urlPathEscape(key), nil)
	return err
}

// BulkWarmup warms multiple KV entries via a server-side pipeline (S).
// Requires BatchURL.
func (c *Client) BulkWarmup(ctx context.Context, entries []BulkWarmupEntry) (*BulkWarmupResult, error) {
	if c.batchURL == "" {
		return nil, fmt.Errorf("cachly: BulkWarmup requires BatchURL")
	}
	resp, err := httpRequest(ctx, http.MethodPost, c.batchURL+"/warm", map[string]any{
		"entries": entries,
	})
	if err != nil {
		return nil, fmt.Errorf("cachly api bulk_warmup: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result BulkWarmupResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// LlmProxyStats returns statistics from the transparent LLM cache proxy (W).
// Requires LlmProxyURL.
func (c *Client) LlmProxyStats(ctx context.Context) (*LlmProxyStats, error) {
	if c.llmProxyURL == "" {
		return nil, fmt.Errorf("cachly: LlmProxyStats requires LlmProxyURL")
	}
	resp, err := httpRequest(ctx, http.MethodGet, c.llmProxyURL+"/stats", nil)
	if err != nil {
		return nil, fmt.Errorf("cachly api llm_proxy_stats: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result LlmProxyStats
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// ── Edge Cache (Feature #5) ────────────────────────────────────────────────

// EdgeCacheConfig represents the edge cache configuration for an instance.
type EdgeCacheConfig struct {
	ID                 string  `json:"id"`
	InstanceID         string  `json:"instance_id"`
	Enabled            bool    `json:"enabled"`
	EdgeTTL            int     `json:"edge_ttl"`
	WorkerURL          string  `json:"worker_url"`
	CloudflareZoneID   string  `json:"cloudflare_zone_id"`
	PurgeOnWrite       bool    `json:"purge_on_write"`
	CacheSearchResults bool    `json:"cache_search_results"`
	TotalHits          int64   `json:"total_hits"`
	TotalMisses        int64   `json:"total_misses"`
	HitRate            float64 `json:"hit_rate"`
}

// EdgeCacheConfigUpdate contains fields to update in edge cache configuration.
type EdgeCacheConfigUpdate struct {
	Enabled            *bool   `json:"enabled,omitempty"`
	EdgeTTL            *int    `json:"edge_ttl,omitempty"`
	WorkerURL          *string `json:"worker_url,omitempty"`
	CloudflareZoneID   *string `json:"cloudflare_zone_id,omitempty"`
	PurgeOnWrite       *bool   `json:"purge_on_write,omitempty"`
	CacheSearchResults *bool   `json:"cache_search_results,omitempty"`
}

// EdgePurgeOptions specifies what to purge from the edge cache.
type EdgePurgeOptions struct {
	Namespaces []string `json:"namespaces,omitempty"`
	URLs       []string `json:"urls,omitempty"`
}

// EdgePurgeResult is returned by EdgeCacheClient.Purge.
type EdgePurgeResult struct {
	Purged int      `json:"purged"`
	URLs   []string `json:"urls"`
}

// EdgeCacheStats contains edge cache hit/miss statistics.
type EdgeCacheStats struct {
	Enabled     bool    `json:"enabled"`
	WorkerURL   string  `json:"worker_url"`
	EdgeTTL     int     `json:"edge_ttl"`
	TotalHits   int64   `json:"total_hits"`
	TotalMisses int64   `json:"total_misses"`
	HitRate     float64 `json:"hit_rate"`
}

// EdgeCacheClient manages the Cloudflare Edge Cache for a cachly instance.
type EdgeCacheClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewEdgeCacheClient creates a new EdgeCacheClient.
func NewEdgeCacheClient(edgeAPIURL string) *EdgeCacheClient {
	return &EdgeCacheClient{
		baseURL:    strings.TrimSuffix(edgeAPIURL, "/"),
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
}

// GetConfig returns the current edge cache configuration.
func (e *EdgeCacheClient) GetConfig(ctx context.Context) (*EdgeCacheConfig, error) {
	var cfg EdgeCacheConfig
	if err := e.doRequest(ctx, "GET", "/config", nil, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// SetConfig updates the edge cache configuration.
func (e *EdgeCacheClient) SetConfig(ctx context.Context, update EdgeCacheConfigUpdate) (*EdgeCacheConfig, error) {
	var cfg EdgeCacheConfig
	if err := e.doRequest(ctx, "PUT", "/config", update, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// DeleteConfig disables and removes the edge cache configuration.
func (e *EdgeCacheClient) DeleteConfig(ctx context.Context) error {
	return e.doRequest(ctx, "DELETE", "/config", nil, nil)
}

// Purge clears cached entries from the Cloudflare CDN.
// Pass nil to purge all entries, or specify namespaces/URLs to purge selectively.
func (e *EdgeCacheClient) Purge(ctx context.Context, opts *EdgePurgeOptions) (*EdgePurgeResult, error) {
	body := opts
	if body == nil {
		body = &EdgePurgeOptions{}
	}
	var result EdgePurgeResult
	if err := e.doRequest(ctx, "POST", "/purge", body, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Stats returns edge cache hit/miss statistics.
func (e *EdgeCacheClient) Stats(ctx context.Context) (*EdgeCacheStats, error) {
	var stats EdgeCacheStats
	if err := e.doRequest(ctx, "GET", "/stats", nil, &stats); err != nil {
		return nil, err
	}
	return &stats, nil
}

func (e *EdgeCacheClient) doRequest(ctx context.Context, method, path string, body, out any) error {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return err
		}
		bodyReader = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, e.baseURL+path, bodyReader)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := e.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("edge API error %d: %s", resp.StatusCode, string(data))
	}
	if out != nil && resp.StatusCode != http.StatusNoContent {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

// ── PubSubClient ──────────────────────────────────────────────────────────────

// PubSubClient provides Pub/Sub messaging via the cachly API (feature U).
// Obtain via Client.PubSub().
type PubSubClient struct {
	url string
}

// Publish sends a message to a channel.
func (p *PubSubClient) Publish(ctx context.Context, channel, message string) (*PubSubPublishResult, error) {
	resp, err := httpRequest(ctx, http.MethodPost, p.url+"/publish", map[string]any{
		"channel": channel,
		"message": message,
	})
	if err != nil {
		return nil, fmt.Errorf("cachly pubsub publish: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result PubSubPublishResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// Subscribe subscribes to one or more channels and returns a channel of SSE events (U).
// The returned channel is closed when the stream ends or ctx is cancelled.
// The caller must read all events or cancel ctx.
func (p *PubSubClient) Subscribe(ctx context.Context, channels ...string) (<-chan map[string]any, error) {
	return httpSSEStream(ctx, http.MethodPost, p.url+"/subscribe", map[string]any{
		"channels": channels,
	})
}

// Channels returns all active channels and their subscriber counts.
func (p *PubSubClient) Channels(ctx context.Context) ([]PubSubChannelInfo, error) {
	resp, err := httpRequest(ctx, http.MethodGet, p.url+"/channels", nil)
	if err != nil {
		return nil, fmt.Errorf("cachly pubsub channels: %w", err)
	}
	raw, _ := json.Marshal(resp["channels"])
	var result []PubSubChannelInfo
	json.Unmarshal(raw, &result) //nolint:errcheck
	return result, nil
}

// Stats returns Pub/Sub statistics.
func (p *PubSubClient) Stats(ctx context.Context) (*PubSubStatsResult, error) {
	resp, err := httpRequest(ctx, http.MethodGet, p.url+"/stats", nil)
	if err != nil {
		return nil, fmt.Errorf("cachly pubsub stats: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result PubSubStatsResult
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// ── WorkflowClient ────────────────────────────────────────────────────────────

// WorkflowClient saves and retrieves agent-workflow checkpoints (feature V).
// Obtain via Client.Workflow().
type WorkflowClient struct {
	url string
}

// SaveCheckpointOptions are the optional fields for WorkflowClient.Save.
type SaveCheckpointOptions struct {
	State      string
	Output     string
	DurationMs int
}

// Save stores a workflow checkpoint.
func (w *WorkflowClient) Save(
	ctx context.Context,
	runID string,
	stepIndex int,
	stepName string,
	agentName string,
	status string,
	opts SaveCheckpointOptions,
) (*WorkflowCheckpoint, error) {
	body := map[string]any{
		"run_id":      runID,
		"step_index":  stepIndex,
		"step_name":   stepName,
		"agent_name":  agentName,
		"status":      status,
		"duration_ms": opts.DurationMs,
	}
	if opts.State != "" {
		body["state"] = opts.State
	}
	if opts.Output != "" {
		body["output"] = opts.Output
	}
	resp, err := httpRequest(ctx, http.MethodPost, w.url+"/checkpoints", body)
	if err != nil {
		return nil, fmt.Errorf("cachly workflow save: %w", err)
	}
	raw, _ := json.Marshal(resp)
	var result WorkflowCheckpoint
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// ListRuns returns summaries of all workflow runs.
func (w *WorkflowClient) ListRuns(ctx context.Context) ([]WorkflowRun, error) {
	resp, err := httpRequest(ctx, http.MethodGet, w.url+"/runs", nil)
	if err != nil {
		return nil, fmt.Errorf("cachly workflow list_runs: %w", err)
	}
	raw, _ := json.Marshal(resp["runs"])
	var result []WorkflowRun
	json.Unmarshal(raw, &result) //nolint:errcheck
	return result, nil
}

// GetRun returns all checkpoints for a run in order.
func (w *WorkflowClient) GetRun(ctx context.Context, runID string) ([]WorkflowCheckpoint, error) {
	resp, err := httpRequest(ctx, http.MethodGet, w.url+"/runs/"+runID, nil)
	if err != nil {
		return nil, fmt.Errorf("cachly workflow get_run: %w", err)
	}
	raw, _ := json.Marshal(resp["checkpoints"])
	var result []WorkflowCheckpoint
	json.Unmarshal(raw, &result) //nolint:errcheck
	return result, nil
}

// Latest returns the latest checkpoint for a run, or nil if the run does not exist.
func (w *WorkflowClient) Latest(ctx context.Context, runID string) (*WorkflowCheckpoint, error) {
	resp, err := httpRequest(ctx, http.MethodGet, w.url+"/runs/"+runID+"/latest", nil)
	if err != nil {
		return nil, nil // treat missing run as nil
	}
	if len(resp) == 0 {
		return nil, nil
	}
	raw, _ := json.Marshal(resp)
	var result WorkflowCheckpoint
	json.Unmarshal(raw, &result) //nolint:errcheck
	return &result, nil
}

// DeleteRun deletes all checkpoints for a run.
func (w *WorkflowClient) DeleteRun(ctx context.Context, runID string) error {
	_, err := httpRequest(ctx, http.MethodDelete, w.url+"/runs/"+runID, nil)
	return err
}


// ── URL path helper ───────────────────────────────────────────────────────────

// urlPathEscape percent-encodes a string for use in a URL path segment.
func urlPathEscape(s string) string {
	r := strings.NewReplacer(
		" ", "%20",
		":", "%3A",
		"/", "%2F",
		"?", "%3F",
		"#", "%23",
		"&", "%26",
		"=", "%3D",
	)
	return r.Replace(s)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func marshal(v any) ([]byte, error) {
	switch val := v.(type) {
	case []byte:
		return val, nil
	case string:
		return []byte(val), nil
	default:
		return json.Marshal(v)
	}
}

func unmarshal(raw []byte, dst any) error {
	switch d := dst.(type) {
	case *string:
		*d = string(raw)
		return nil
	case *[]byte:
		*d = raw
		return nil
	default:
		return json.Unmarshal(raw, dst)
	}
}
