// Package cachly – unit tests for 0.4.0 new features (A–W).
// HTTP-dependent features use net/http/httptest; Valkey-only features use miniredis.
package cachly

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

// ── helpers ───────────────────────────────────────────────────────────────────

// newTestClientWithURLs creates a Client backed by miniredis and optional API URLs.
func newTestClientWithURLs(t *testing.T, vectorURL, batchURL, pubsubURL, workflowURL, llmProxyURL string) (*Client, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	return &Client{
		rdb:         rdb,
		vectorURL:   strings.TrimRight(vectorURL, "/"),
		batchURL:    strings.TrimRight(batchURL, "/"),
		pubsubURL:   strings.TrimRight(pubsubURL, "/"),
		workflowURL: strings.TrimRight(workflowURL, "/"),
		llmProxyURL: strings.TrimRight(llmProxyURL, "/"),
	}, mr
}

// respondJSON writes a JSON body with status 200.
func respondJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

// simpleEmbed returns a fixed vector so tests don't need a real embedding model.
func simpleEmbed(_ context.Context, _ string) ([]float64, error) {
	return []float64{1, 0, 0}, nil
}

// ── A – SetThreshold ─────────────────────────────────────────────────────────

func TestSetThreshold_CallsAPI(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/threshold" {
			called = true
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if ns, _ := body["namespace"].(string); ns != "cachly:sem" {
				t.Errorf("unexpected namespace: %q", ns)
			}
			if thr, _ := body["threshold"].(float64); thr != 0.92 {
				t.Errorf("unexpected threshold: %f", thr)
			}
			respondJSON(w, map[string]any{"ok": true, "namespace": "cachly:sem", "threshold": 0.92})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	if err := sem.SetThreshold(context.Background(), "cachly:sem", 0.92); err != nil {
		t.Fatalf("SetThreshold: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

func TestSetThreshold_NoVectorURL_IsNoop(t *testing.T) {
	c, _ := newTestClient(t)
	sem := c.Semantic(simpleEmbed)
	if err := sem.SetThreshold(context.Background(), "", 0.90); err != nil {
		t.Errorf("expected nil for no-op, got %v", err)
	}
}

// ── B – Stats ────────────────────────────────────────────────────────────────

func TestStats_ViaAPI(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/stats" {
			respondJSON(w, map[string]any{
				"hits":     float64(100),
				"misses":   float64(10),
				"hit_rate": 0.909,
				"total":    float64(110),
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	stats, err := sem.Stats(context.Background(), "cachly:sem")
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if stats.Hits != 100 || stats.Misses != 10 || stats.Total != 110 {
		t.Errorf("unexpected stats: %+v", stats)
	}
}

func TestStats_Valkey_Fallback(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()
	// Pre-seed counters manually
	c.rdb.Set(ctx, "cachly:sem:stats:hits", "7", 0)   //nolint:errcheck
	c.rdb.Set(ctx, "cachly:sem:stats:misses", "3", 0) //nolint:errcheck

	sem := c.Semantic(simpleEmbed)
	stats, err := sem.Stats(ctx, "cachly:sem")
	if err != nil {
		t.Fatalf("Stats (fallback): %v", err)
	}
	if stats.Hits != 7 || stats.Misses != 3 || stats.Total != 10 {
		t.Errorf("unexpected stats: %+v", stats)
	}
	want := float64(7) / float64(10)
	if stats.HitRate < want-0.001 || stats.HitRate > want+0.001 {
		t.Errorf("hit_rate: want %f got %f", want, stats.HitRate)
	}
}

// ── C – StreamSearch ─────────────────────────────────────────────────────────

func TestStreamSearch_ReceivesEvents(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/search/stream" {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			flusher, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, "SSE not supported", http.StatusInternalServerError)
				return
			}
			events := []string{
				`{"found":true,"id":"abc","similarity":0.95}`,
				`{"text":"Paris"}`,
				`{}`, // done sentinel
			}
			for _, e := range events {
				fmt.Fprintf(w, "data: %s\n\n", e)
				flusher.Flush()
			}
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ch, err := sem.StreamSearch(ctx, "What is caching?", SemanticOptions{Namespace: "cachly:sem"})
	if err != nil {
		t.Fatalf("StreamSearch: %v", err)
	}

	var events []map[string]any
	for ev := range ch {
		events = append(events, ev)
	}

	if len(events) != 2 {
		t.Fatalf("expected 2 events (sentinel excluded), got %d", len(events))
	}
	if found, _ := events[0]["found"].(bool); !found {
		t.Errorf("first event should have found=true")
	}
	if text, _ := events[1]["text"].(string); text != "Paris" {
		t.Errorf("second event should have text='Paris', got %q", text)
	}
}

func TestStreamSearch_RequiresVectorURL(t *testing.T) {
	c, _ := newTestClient(t)
	sem := c.Semantic(simpleEmbed)
	_, err := sem.StreamSearch(context.Background(), "test", SemanticOptions{})
	if err == nil {
		t.Error("expected error when no VectorURL set")
	}
}

// ── D – BatchIndex ───────────────────────────────────────────────────────────

func TestBatchIndex_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/entries/batch" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			respondJSON(w, map[string]any{"indexed": float64(3), "skipped": float64(0)})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)

	entries := []BatchIndexEntry{
		{ID: "id1", Prompt: "p1", Embedding: []float64{1, 0}, Namespace: "cachly:sem"},
		{ID: "id2", Prompt: "p2", Embedding: []float64{0, 1}, Namespace: "cachly:sem"},
		{ID: "id3", Prompt: "p3", Embedding: []float64{1, 1}, Namespace: "cachly:sem"},
	}
	result, err := sem.BatchIndex(context.Background(), entries)
	if err != nil {
		t.Fatalf("BatchIndex: %v", err)
	}
	if result.Indexed != 3 || result.Skipped != 0 {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestBatchIndex_RequiresVectorURL(t *testing.T) {
	c, _ := newTestClient(t)
	sem := c.Semantic(simpleEmbed)
	_, err := sem.BatchIndex(context.Background(), []BatchIndexEntry{{ID: "x", Prompt: "y", Embedding: []float64{1}}})
	if err == nil {
		t.Error("expected error when no VectorURL set")
	}
}

// ── E – CreateIndex ──────────────────────────────────────────────────────────

func TestCreateIndex_CallsAPI(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/indexes" {
			called = true
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if ns, _ := body["namespace"].(string); ns != "my:ns" {
				t.Errorf("unexpected namespace: %q", ns)
			}
			respondJSON(w, map[string]any{"ok": true})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	if err := sem.CreateIndex(context.Background(), "my:ns", 1536, "text-embedding-3-small", "cosine", true); err != nil {
		t.Fatalf("CreateIndex: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── F – DeleteIndex ──────────────────────────────────────────────────────────

func TestDeleteIndex_CallsAPI(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && strings.Contains(r.URL.Path, "/indexes/") {
			called = true
			respondJSON(w, map[string]any{"ok": true})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	if err := sem.DeleteIndex(context.Background(), "my:ns"); err != nil {
		t.Fatalf("DeleteIndex: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── G – SetMetadata ──────────────────────────────────────────────────────────

func TestSetMetadata_CallsAPI(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/metadata" {
			called = true
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if id, _ := body["entry_id"].(string); id != "some-uuid" {
				t.Errorf("unexpected entry_id: %q", id)
			}
			respondJSON(w, map[string]any{"ok": true})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	err := sem.SetMetadata(context.Background(), "some-uuid", map[string]any{"lang": "de"})
	if err != nil {
		t.Fatalf("SetMetadata: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── H – FilteredSearch ───────────────────────────────────────────────────────

func TestFilteredSearch_ReturnsResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/search/filtered" {
			respondJSON(w, map[string]any{
				"results": []any{
					map[string]any{"id": "abc", "similarity": 0.91},
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	results, err := sem.FilteredSearch(context.Background(), "caching", "cachly:sem", 0.85, map[string]any{"lang": "de"}, 5)
	if err != nil {
		t.Fatalf("FilteredSearch: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if id, _ := results[0]["id"].(string); id != "abc" {
		t.Errorf("unexpected id: %q", id)
	}
}

func TestFilteredSearch_EmptyResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		respondJSON(w, map[string]any{"results": []any{}})
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	results, err := sem.FilteredSearch(context.Background(), "xyz", "", 0, nil, 0)
	if err != nil {
		t.Fatalf("FilteredSearch: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results, got %d", len(results))
	}
}

// ── I – SetGuardrail ─────────────────────────────────────────────────────────

func TestSetGuardrail_CallsAPI(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/guardrails" {
			called = true
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if action, _ := body["pii_action"].(string); action != "block" {
				t.Errorf("unexpected pii_action: %q", action)
			}
			respondJSON(w, map[string]any{"ok": true})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	if err := sem.SetGuardrail(context.Background(), "cachly:sem", "block", "flag", 0.8); err != nil {
		t.Fatalf("SetGuardrail: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── J – DeleteGuardrail ──────────────────────────────────────────────────────

func TestDeleteGuardrail_CallsAPI(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && strings.Contains(r.URL.Path, "/guardrails/") {
			called = true
			respondJSON(w, map[string]any{"ok": true})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	if err := sem.DeleteGuardrail(context.Background(), "cachly:sem"); err != nil {
		t.Fatalf("DeleteGuardrail: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── K – CheckGuardrail ───────────────────────────────────────────────────────

func TestCheckGuardrail_Unsafe(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/guardrails/check" {
			respondJSON(w, map[string]any{
				"safe": false,
				"violations": []any{
					map[string]any{"type": "pii", "pattern": "email", "action": "block"},
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	result, err := sem.CheckGuardrail(context.Background(), "my email is user@example.com", "cachly:sem")
	if err != nil {
		t.Fatalf("CheckGuardrail: %v", err)
	}
	if result.Safe {
		t.Error("expected safe=false")
	}
	if len(result.Violations) != 1 || result.Violations[0].Type != "pii" {
		t.Errorf("unexpected violations: %+v", result.Violations)
	}
}

func TestCheckGuardrail_Safe(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		respondJSON(w, map[string]any{"safe": true, "violations": []any{}})
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	result, err := sem.CheckGuardrail(context.Background(), "Hello world", "")
	if err != nil {
		t.Fatalf("CheckGuardrail: %v", err)
	}
	if !result.Safe || len(result.Violations) != 0 {
		t.Errorf("expected safe with no violations: %+v", result)
	}
}

// ── T – SnapshotWarmup ───────────────────────────────────────────────────────

func TestSnapshotWarmup_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/warmup/snapshot" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if lim, _ := body["limit"].(float64); lim != 50 {
				t.Errorf("expected limit=50, got %v", lim)
			}
			respondJSON(w, map[string]any{"warmed": float64(35), "duration_ms": float64(120)})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	result, err := sem.SnapshotWarmup(context.Background(), "cachly:sem", 50)
	if err != nil {
		t.Fatalf("SnapshotWarmup: %v", err)
	}
	if result.Warmed != 35 {
		t.Errorf("expected warmed=35, got %d", result.Warmed)
	}
}

func TestSnapshotWarmup_DefaultLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
		if lim, _ := body["limit"].(float64); lim != 100 {
			t.Errorf("expected default limit=100, got %v", lim)
		}
		respondJSON(w, map[string]any{"warmed": float64(10), "duration_ms": float64(50)})
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, srv.URL, "", "", "", "")
	sem := c.Semantic(simpleEmbed)
	_, _ = sem.SnapshotWarmup(context.Background(), "", 0) // 0 → default 100
}

// ── L – SetTags ──────────────────────────────────────────────────────────────

func TestSetTags_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/tags" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if key, _ := body["key"].(string); key != "user:42" {
				t.Errorf("unexpected key: %q", key)
			}
			respondJSON(w, map[string]any{
				"key":  "user:42",
				"tags": []any{"user:7", "orders"},
				"ok":   true,
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	result, err := c.SetTags(context.Background(), "user:42", []string{"user:7", "orders"})
	if err != nil {
		t.Fatalf("SetTags: %v", err)
	}
	if !result.Ok || result.Key != "user:42" || len(result.Tags) != 2 {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestSetTags_RequiresBatchURL(t *testing.T) {
	c, _ := newTestClient(t)
	_, err := c.SetTags(context.Background(), "key", []string{"tag"})
	if err == nil {
		t.Error("expected error when no BatchURL set")
	}
}

// ── M – InvalidateTag ────────────────────────────────────────────────────────

func TestInvalidateTag_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/invalidate" {
			respondJSON(w, map[string]any{
				"tag":          "user:7",
				"keys_deleted": float64(5),
				"keys":         []any{"user:42", "cart:42"},
				"duration_ms":  float64(12),
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	result, err := c.InvalidateTag(context.Background(), "user:7")
	if err != nil {
		t.Fatalf("InvalidateTag: %v", err)
	}
	if result.Tag != "user:7" || result.KeysDeleted != 5 {
		t.Errorf("unexpected result: %+v", result)
	}
}

// ── N – GetTags ──────────────────────────────────────────────────────────────

func TestGetTags_ReturnsTags(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/tags/") {
			respondJSON(w, map[string]any{
				"key":  "user:42",
				"tags": []any{"user:7", "orders"},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	tags, err := c.GetTags(context.Background(), "user:42")
	if err != nil {
		t.Fatalf("GetTags: %v", err)
	}
	if len(tags) != 2 || tags[0] != "user:7" || tags[1] != "orders" {
		t.Errorf("unexpected tags: %v", tags)
	}
}

func TestGetTags_Empty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		respondJSON(w, map[string]any{"key": "x", "tags": nil})
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	tags, err := c.GetTags(context.Background(), "x")
	if err != nil {
		t.Fatalf("GetTags: %v", err)
	}
	if tags != nil {
		t.Errorf("expected nil tags, got %v", tags)
	}
}

// ── O – DeleteTags ───────────────────────────────────────────────────────────

func TestDeleteTags_Success(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/tags/") {
			called = true
			respondJSON(w, map[string]any{"ok": true, "key": "user:42"})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	if err := c.DeleteTags(context.Background(), "user:42"); err != nil {
		t.Fatalf("DeleteTags: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── P – SWRRegister ──────────────────────────────────────────────────────────

func TestSWRRegister_WithHint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/swr/register" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if hint, _ := body["fetcher_hint"].(string); hint != "reload_config" {
				t.Errorf("unexpected fetcher_hint: %q", hint)
			}
			respondJSON(w, map[string]any{"ok": true})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	if err := c.SWRRegister(context.Background(), "config:app", 3600, 300, "reload_config"); err != nil {
		t.Fatalf("SWRRegister: %v", err)
	}
}

func TestSWRRegister_NoHint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
		if _, ok := body["fetcher_hint"]; ok {
			t.Error("fetcher_hint should not be sent when empty")
		}
		respondJSON(w, map[string]any{"ok": true})
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	if err := c.SWRRegister(context.Background(), "key", 3600, 300, ""); err != nil {
		t.Fatalf("SWRRegister (no hint): %v", err)
	}
}

// ── Q – SWRCheck ─────────────────────────────────────────────────────────────

func TestSWRCheck_ReturnsStaleKeys(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/swr/check" {
			respondJSON(w, map[string]any{
				"stale_keys": []any{
					map[string]any{"key": "config:app", "fetcher_hint": "reload_config", "stale_for": "5m", "refresh_at": "2026-04-14T12:00:00Z"},
				},
				"count":      float64(1),
				"checked_at": "2026-04-14T11:55:00Z",
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	result, err := c.SWRCheck(context.Background())
	if err != nil {
		t.Fatalf("SWRCheck: %v", err)
	}
	if result.Count != 1 || len(result.StaleKeys) != 1 {
		t.Errorf("unexpected result: %+v", result)
	}
	if result.StaleKeys[0].Key != "config:app" {
		t.Errorf("unexpected stale key: %q", result.StaleKeys[0].Key)
	}
}

// ── R – SWRRemove ────────────────────────────────────────────────────────────

func TestSWRRemove_Success(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/swr/") {
			called = true
			respondJSON(w, map[string]any{"ok": true})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	if err := c.SWRRemove(context.Background(), "config:app"); err != nil {
		t.Fatalf("SWRRemove: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── S – BulkWarmup ───────────────────────────────────────────────────────────

func TestBulkWarmup_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/warm" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			entries, _ := body["entries"].([]any)
			respondJSON(w, map[string]any{
				"warmed":      float64(len(entries)),
				"skipped":     float64(0),
				"duration_ms": float64(80),
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", srv.URL, "", "", "")
	entries := []BulkWarmupEntry{
		{Key: "user:1", Value: `{"name":"Alice"}`, TTL: 3600},
		{Key: "user:2", Value: `{"name":"Bob"}`, TTL: 3600},
	}
	result, err := c.BulkWarmup(context.Background(), entries)
	if err != nil {
		t.Fatalf("BulkWarmup: %v", err)
	}
	if result.Warmed != 2 || result.Skipped != 0 {
		t.Errorf("unexpected result: %+v", result)
	}
}

// ── W – LlmProxyStats ────────────────────────────────────────────────────────

func TestLlmProxyStats_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/stats" {
			respondJSON(w, map[string]any{
				"total_requests":          float64(500),
				"cache_hits":              float64(420),
				"cache_misses":            float64(80),
				"estimated_saved_usd":     1.234,
				"avg_latency_ms_cached":   float64(2),
				"avg_latency_ms_uncached": float64(890),
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", "", "", srv.URL)
	stats, err := c.LlmProxyStats(context.Background())
	if err != nil {
		t.Fatalf("LlmProxyStats: %v", err)
	}
	if stats.TotalRequests != 500 || stats.CacheHits != 420 || stats.CacheMisses != 80 {
		t.Errorf("unexpected stats: %+v", stats)
	}
	if stats.EstimatedSavedUSD < 1.23 || stats.EstimatedSavedUSD > 1.24 {
		t.Errorf("unexpected estimated_saved_usd: %f", stats.EstimatedSavedUSD)
	}
}

func TestLlmProxyStats_RequiresLlmProxyURL(t *testing.T) {
	c, _ := newTestClient(t)
	_, err := c.LlmProxyStats(context.Background())
	if err == nil {
		t.Error("expected error when no LlmProxyURL set")
	}
}

// ── U – PubSubClient ─────────────────────────────────────────────────────────

func TestPubSub_RequiresPubSubURL(t *testing.T) {
	c, _ := newTestClient(t)
	_, err := c.PubSub()
	if err == nil {
		t.Error("expected error when no PubSubURL set")
	}
}

func TestPubSubPublish_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/publish" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if ch, _ := body["channel"].(string); ch != "events" {
				t.Errorf("unexpected channel: %q", ch)
			}
			respondJSON(w, map[string]any{
				"channel":      "events",
				"receivers":    float64(3),
				"published_at": "2026-04-14T10:00:00Z",
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", srv.URL, "", "")
	ps, err := c.PubSub()
	if err != nil {
		t.Fatalf("PubSub: %v", err)
	}
	result, err := ps.Publish(context.Background(), "events", "hello")
	if err != nil {
		t.Fatalf("Publish: %v", err)
	}
	if result.Channel != "events" || result.Receivers != 3 {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestPubSubChannels_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/channels" {
			respondJSON(w, map[string]any{
				"channels": []any{
					map[string]any{"name": "events", "subscribers": float64(3)},
					map[string]any{"name": "updates", "subscribers": float64(1)},
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", srv.URL, "", "")
	ps, _ := c.PubSub()
	channels, err := ps.Channels(context.Background())
	if err != nil {
		t.Fatalf("Channels: %v", err)
	}
	if len(channels) != 2 || channels[0].Name != "events" || channels[0].Subscribers != 3 {
		t.Errorf("unexpected channels: %+v", channels)
	}
}

func TestPubSubStats_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/stats" {
			respondJSON(w, map[string]any{
				"active_channels":   float64(2),
				"total_subscribers": float64(7),
				"pattern_count":     float64(0),
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", srv.URL, "", "")
	ps, _ := c.PubSub()
	stats, err := ps.Stats(context.Background())
	if err != nil {
		t.Fatalf("PubSubStats: %v", err)
	}
	if stats.ActiveChannels != 2 || stats.TotalSubscribers != 7 {
		t.Errorf("unexpected stats: %+v", stats)
	}
}

func TestPubSubSubscribe_ReceivesEvents(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/subscribe" {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			fmt.Fprintf(w, "data: {\"channel\":\"events\",\"message\":\"hello\",\"at\":\"2026-04-14T10:00:00Z\"}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {}\n\n") // done sentinel
			flusher.Flush()
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", srv.URL, "", "")
	ps, _ := c.PubSub()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ch, err := ps.Subscribe(ctx, "events")
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}

	var events []map[string]any
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	if msg, _ := events[0]["message"].(string); msg != "hello" {
		t.Errorf("unexpected message: %q", msg)
	}
}

// ── V – WorkflowClient ───────────────────────────────────────────────────────

func TestWorkflow_RequiresWorkflowURL(t *testing.T) {
	c, _ := newTestClient(t)
	_, err := c.Workflow()
	if err == nil {
		t.Error("expected error when no WorkflowURL set")
	}
}

func TestWorkflowSave_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/checkpoints" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
			if runID, _ := body["run_id"].(string); runID != "run-42" {
				t.Errorf("unexpected run_id: %q", runID)
			}
			respondJSON(w, map[string]any{
				"id":          "ckpt-uuid",
				"run_id":      "run-42",
				"step_index":  float64(0),
				"step_name":   "fetch_data",
				"agent_name":  "researcher",
				"status":      "completed",
				"duration_ms": float64(1200),
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", "", srv.URL, "")
	wf, err := c.Workflow()
	if err != nil {
		t.Fatalf("Workflow: %v", err)
	}
	cp, err := wf.Save(context.Background(), "run-42", 0, "fetch_data", "researcher", "completed",
		SaveCheckpointOptions{State: `{"step": 0}`, Output: `{"data": "ok"}`, DurationMs: 1200})
	if err != nil {
		t.Fatalf("Save: %v", err)
	}
	if cp.RunID != "run-42" || cp.StepName != "fetch_data" || cp.Status != "completed" {
		t.Errorf("unexpected checkpoint: %+v", cp)
	}
}

func TestWorkflowListRuns_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/runs" {
			respondJSON(w, map[string]any{
				"runs": []any{
					map[string]any{"run_id": "run-42", "steps": float64(3), "latest_status": "completed"},
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", "", srv.URL, "")
	wf, _ := c.Workflow()
	runs, err := wf.ListRuns(context.Background())
	if err != nil {
		t.Fatalf("ListRuns: %v", err)
	}
	if len(runs) != 1 || runs[0].RunID != "run-42" || runs[0].Steps != 3 {
		t.Errorf("unexpected runs: %+v", runs)
	}
}

func TestWorkflowGetRun_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/runs/run-42" {
			respondJSON(w, map[string]any{
				"run_id": "run-42",
				"checkpoints": []any{
					map[string]any{"id": "c1", "run_id": "run-42", "step_index": float64(0), "step_name": "s1", "agent_name": "a1", "status": "completed", "duration_ms": float64(100)},
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", "", srv.URL, "")
	wf, _ := c.Workflow()
	checkpoints, err := wf.GetRun(context.Background(), "run-42")
	if err != nil {
		t.Fatalf("GetRun: %v", err)
	}
	if len(checkpoints) != 1 || checkpoints[0].ID != "c1" {
		t.Errorf("unexpected checkpoints: %+v", checkpoints)
	}
}

func TestWorkflowLatest_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/runs/run-42/latest" {
			respondJSON(w, map[string]any{
				"id": "c-latest", "run_id": "run-42", "step_index": float64(2),
				"step_name": "final", "agent_name": "agent", "status": "completed", "duration_ms": float64(50),
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", "", srv.URL, "")
	wf, _ := c.Workflow()
	cp, err := wf.Latest(context.Background(), "run-42")
	if err != nil {
		t.Fatalf("Latest: %v", err)
	}
	if cp == nil || cp.ID != "c-latest" {
		t.Errorf("unexpected latest checkpoint: %+v", cp)
	}
}

func TestWorkflowLatest_NotFound_ReturnsNil(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", "", srv.URL, "")
	wf, _ := c.Workflow()
	cp, err := wf.Latest(context.Background(), "nonexistent")
	if err != nil {
		t.Fatalf("Latest on missing run should not error: %v", err)
	}
	if cp != nil {
		t.Errorf("expected nil for missing run, got %+v", cp)
	}
}

func TestWorkflowDeleteRun_Success(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && r.URL.Path == "/runs/run-42" {
			called = true
			respondJSON(w, map[string]any{"ok": true, "run_id": "run-42"})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	c, _ := newTestClientWithURLs(t, "", "", "", srv.URL, "")
	wf, _ := c.Workflow()
	if err := wf.DeleteRun(context.Background(), "run-42"); err != nil {
		t.Fatalf("DeleteRun: %v", err)
	}
	if !called {
		t.Error("API endpoint was not called")
	}
}

// ── Result type field coverage ────────────────────────────────────────────────

func TestResultTypes_FieldsExist(t *testing.T) {
	// Compile-time check: verify all new result types have expected fields.
	_ = CacheStats{Hits: 1, Misses: 2, HitRate: 0.5, Total: 3, Namespaces: nil}
	_ = BatchIndexEntry{ID: "x", Prompt: "p", Embedding: []float64{1}, Namespace: "ns", ExpiresAt: "2026-01-01T00:00:00Z"}
	_ = BatchIndexResult{Indexed: 1, Skipped: 0}
	_ = GuardrailViolation{Type: "pii", Pattern: "email", Action: "block"}
	_ = GuardrailCheckResult{Safe: false, Violations: []GuardrailViolation{}}
	_ = TagsResult{Key: "k", Tags: []string{"t"}, Ok: true}
	_ = InvalidateTagResult{Tag: "t", KeysDeleted: 1, Keys: []string{"k"}, DurationMs: 5}
	_ = SwrEntry{Key: "k", FetcherHint: "hint", StaleFor: "5m", RefreshAt: "2026-01-01T00:00:00Z"}
	_ = SwrCheckResult{StaleKeys: nil, Count: 0, CheckedAt: "2026-01-01T00:00:00Z"}
	_ = BulkWarmupEntry{Key: "k", Value: "v", TTL: 3600}
	_ = BulkWarmupResult{Warmed: 1, Skipped: 0, DurationMs: 10}
	_ = SnapshotWarmupResult{Warmed: 5, DurationMs: 50}
	_ = LlmProxyStats{TotalRequests: 500, CacheHits: 420, CacheMisses: 80, EstimatedSavedUSD: 1.2, AvgLatencyMsCached: 2, AvgLatencyMsUncached: 890}
	_ = PubSubPublishResult{Channel: "c", Receivers: 3, PublishedAt: "2026-01-01T00:00:00Z"}
	_ = PubSubChannelInfo{Name: "c", Subscribers: 1}
	_ = PubSubStatsResult{ActiveChannels: 2, TotalSubscribers: 7, PatternCount: 0}
	_ = WorkflowCheckpoint{ID: "id", RunID: "r", StepIndex: 0, StepName: "s", AgentName: "a", Status: "ok", State: "", Output: "", DurationMs: 100}
	_ = WorkflowRun{RunID: "r", Steps: 3, LatestStatus: "completed"}
	_ = SaveCheckpointOptions{State: "s", Output: "o", DurationMs: 100}
}

// ── Config field coverage ─────────────────────────────────────────────────────

func TestConfig_NewFields(t *testing.T) {
	cfg := Config{
		URL:         "redis://:pass@localhost:6379",
		PubSubURL:   "https://api.cachly.dev/v1/pubsub/token",
		WorkflowURL: "https://api.cachly.dev/v1/workflow/token",
		LlmProxyURL: "https://api.cachly.dev/v1/llm-proxy/token",
	}
	// Field existence check (compile-time).
	if cfg.PubSubURL == "" || cfg.WorkflowURL == "" || cfg.LlmProxyURL == "" {
		t.Error("Config URL fields should not be empty")
	}
}

