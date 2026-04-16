// Package cachly – unit tests using miniredis (no live Redis required).
package cachly

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

// newTestClient starts an in-memory miniredis server and returns a Client
// backed by it together with a cleanup function.
func newTestClient(t *testing.T) (*Client, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	return &Client{rdb: rdb}, mr
}

// ── Set / Get ─────────────────────────────────────────────────────────────────

func TestSetAndGet_String(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	if err := c.Set(ctx, "k", "hello", 0); err != nil {
		t.Fatalf("Set: %v", err)
	}
	var got string
	if err := c.Get(ctx, "k", &got); err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != "hello" {
		t.Errorf("want %q got %q", "hello", got)
	}
}

func TestSetAndGet_Struct(t *testing.T) {
	type User struct {
		ID   int    `json:"id"`
		Name string `json:"name"`
	}
	c, _ := newTestClient(t)
	ctx := context.Background()

	want := User{ID: 42, Name: "Alice"}
	if err := c.Set(ctx, "user:42", want, 0); err != nil {
		t.Fatalf("Set: %v", err)
	}
	var got User
	if err := c.Get(ctx, "user:42", &got); err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != want {
		t.Errorf("want %+v got %+v", want, got)
	}
}

func TestGet_Missing_ReturnsRedisNil(t *testing.T) {
	c, _ := newTestClient(t)
	var s string
	err := c.Get(context.Background(), "nonexistent", &s)
	if err != redis.Nil {
		t.Errorf("expected redis.Nil, got %v", err)
	}
}

// ── Delete ────────────────────────────────────────────────────────────────────

func TestDelete(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	_ = c.Set(ctx, "a", "1", 0)
	_ = c.Set(ctx, "b", "2", 0)

	n, err := c.Delete(ctx, "a", "b", "missing")
	if err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if n != 2 {
		t.Errorf("expected 2 deletions, got %d", n)
	}
}

// ── Exists ────────────────────────────────────────────────────────────────────

func TestExists_True(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	_ = c.Set(ctx, "x", "1", 0)
	ok, err := c.Exists(ctx, "x")
	if err != nil || !ok {
		t.Errorf("expected true, got ok=%v err=%v", ok, err)
	}
}

func TestExists_False(t *testing.T) {
	c, _ := newTestClient(t)
	ok, err := c.Exists(context.Background(), "ghost")
	if err != nil || ok {
		t.Errorf("expected false, got ok=%v err=%v", ok, err)
	}
}

// ── Incr ──────────────────────────────────────────────────────────────────────

func TestIncr(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	v1, _ := c.Incr(ctx, "counter")
	v2, _ := c.Incr(ctx, "counter")
	v3, _ := c.IncrBy(ctx, "counter", 10)

	if v1 != 1 || v2 != 2 || v3 != 12 {
		t.Errorf("counter progression: %d %d %d (want 1 2 12)", v1, v2, v3)
	}
}

// ── Expire / TTL ──────────────────────────────────────────────────────────────

func TestExpireAndTTL(t *testing.T) {
	c, mr := newTestClient(t)
	ctx := context.Background()

	_ = c.Set(ctx, "tmp", "data", 0)
	ok, err := c.Expire(ctx, "tmp", 60*time.Second)
	if err != nil || !ok {
		t.Fatalf("Expire: ok=%v err=%v", ok, err)
	}

	ttl, err := c.TTL(ctx, "tmp")
	if err != nil {
		t.Fatalf("TTL: %v", err)
	}
	if ttl <= 0 || ttl > 60*time.Second {
		t.Errorf("unexpected TTL %v", ttl)
	}

	// Advance miniredis clock past expiry
	mr.FastForward(61 * time.Second)

	var s string
	if err := c.Get(ctx, "tmp", &s); err != redis.Nil {
		t.Errorf("expected key to be expired, got err=%v val=%q", err, s)
	}
}

// ── GetOrSet ──────────────────────────────────────────────────────────────────

func TestGetOrSet_MissThenHit(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	calls := 0

	fn := func() (any, error) {
		calls++
		return "computed", nil
	}

	v1, err := c.GetOrSet(ctx, "gos", fn, time.Minute)
	if err != nil || v1 != "computed" {
		t.Fatalf("first call: v=%v err=%v", v1, err)
	}
	v2, err := c.GetOrSet(ctx, "gos", fn, time.Minute)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	// fn should only have been called once
	if calls != 1 {
		t.Errorf("fn called %d times, expected 1", calls)
	}
	_ = v2
}

// ── Semantic Cache ────────────────────────────────────────────────────────────

// deterministicEmbed returns a simple unit vector derived from the input text.
// It handles both original prompts and the normalised forms produced by normalizePrompt
// (lower-cased, filler words stripped, trailing "?" appended).
func deterministicEmbed(text string) []float64 {
	switch text {
	case "what is the capital of France",
		"what is the capital of france",
		"what is the capital of france?":
		return []float64{1, 0, 0}
	case "capital city of France",
		"capital city of france",
		"capital city of france?":
		return []float64{0.99, 0.14, 0} // ~cos 0.99 → hit above 0.90
	case "best pizza recipe",
		"best pizza recipe?":
		return []float64{0, 1, 0} // orthogonal to France → miss
	default:
		return []float64{0, 0, 1}
	}
}

func makeEmbedFn(t *testing.T) EmbedFn {
	t.Helper()
	return func(_ context.Context, text string) ([]float64, error) {
		return deterministicEmbed(text), nil
	}
}

func TestSemantic_MissThenHit(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	sem := c.Semantic(makeEmbedFn(t))

	calls := 0
	fn := func() (any, error) {
		calls++
		return "Paris", nil
	}

	r1, err := sem.GetOrSet(ctx, "what is the capital of France", fn,
		SemanticOptions{Threshold: 0.90, TTL: time.Minute})
	if err != nil || r1.Hit || r1.Value != "Paris" {
		t.Fatalf("first call (miss): %+v err=%v", r1, err)
	}

	r2, err := sem.GetOrSet(ctx, "capital city of France", fn,
		SemanticOptions{Threshold: 0.90, TTL: time.Minute})
	if err != nil || !r2.Hit || r2.Value != "Paris" {
		t.Fatalf("second call (should be hit): %+v err=%v", r2, err)
	}
	if calls != 1 {
		t.Errorf("fn called %d times, expected 1", calls)
	}
}

func TestSemantic_BelowThreshold_IsMiss(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	sem := c.Semantic(makeEmbedFn(t))

	fn := func() (any, error) { return "answer", nil }

	_, _ = sem.GetOrSet(ctx, "what is the capital of France", fn,
		SemanticOptions{Threshold: 0.90, TTL: time.Minute})

	r, err := sem.GetOrSet(ctx, "best pizza recipe", fn,
		SemanticOptions{Threshold: 0.90, TTL: time.Minute})
	if err != nil || r.Hit {
		t.Errorf("expected cache miss for unrelated prompt, got hit=%v err=%v", r.Hit, err)
	}
}

func TestSemantic_Flush(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	sem := c.Semantic(makeEmbedFn(t))

	fn := func() (any, error) { return "Paris", nil }
	_, _ = sem.GetOrSet(ctx, "what is the capital of France", fn,
		SemanticOptions{Threshold: 0.90, TTL: time.Minute, Namespace: "test:sem"})

	n, err := sem.Flush(ctx, "test:sem")
	if err != nil || n == 0 {
		t.Errorf("Flush: n=%d err=%v", n, err)
	}

	size, _ := sem.Size(ctx, "test:sem")
	if size != 0 {
		t.Errorf("expected empty namespace after flush, got size=%d", size)
	}
}

func TestSemantic_Invalidate(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	sem := c.Semantic(makeEmbedFn(t))

	fn := func() (any, error) { return "Paris", nil }
	_, _ = sem.GetOrSet(ctx, "what is the capital of France", fn,
		SemanticOptions{Threshold: 0.90, TTL: time.Minute, Namespace: "test:inv"})

	entries, err := sem.Entries(ctx, "test:inv")
	if err != nil || len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d, err=%v", len(entries), err)
	}

	if err := sem.Invalidate(ctx, entries[0].Key); err != nil {
		t.Fatalf("Invalidate: %v", err)
	}

	size, _ := sem.Size(ctx, "test:inv")
	if size != 0 {
		t.Errorf("expected size 0 after Invalidate, got %d", size)
	}
}

func TestSemantic_Entries(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	sem := c.Semantic(makeEmbedFn(t))
	ns := "test:entries"

	prompts := []string{"what is the capital of France", "best pizza recipe"}
	for _, p := range prompts {
		fn := func() (any, error) { return "answer", nil }
		_, _ = sem.GetOrSet(ctx, p, fn, SemanticOptions{Namespace: ns})
	}

	entries, err := sem.Entries(ctx, ns)
	if err != nil {
		t.Fatalf("Entries: %v", err)
	}
	if len(entries) != 2 {
		t.Errorf("expected 2 entries, got %d", len(entries))
	}
	seen := make(map[string]bool)
	for _, e := range entries {
		seen[e.Prompt] = true
		if e.Key == "" {
			t.Errorf("entry has empty Key: %+v", e)
		}
	}
	for _, p := range prompts {
		if !seen[p] {
			t.Errorf("prompt %q not found in entries", p)
		}
	}
}

// ── cosineSimilarity ─────────────────────────────────────────────────────────

func TestCosineSimilarity_Identical(t *testing.T) {
	v := []float64{1, 2, 3}
	sim := cosineSimilarity(v, v)
	if sim < 0.9999 || sim > 1.0001 {
		t.Errorf("identical vectors: want ~1.0, got %f", sim)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{0, 1}
	sim := cosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("orthogonal vectors: want 0, got %f", sim)
	}
}

func TestCosineSimilarity_DifferentLength(t *testing.T) {
	sim := cosineSimilarity([]float64{1, 2}, []float64{1})
	if sim != 0 {
		t.Errorf("different length: want 0, got %f", sim)
	}
}

