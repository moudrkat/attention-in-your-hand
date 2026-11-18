// One attention head of a language model, on 64 LEDs, while the model writes.
//
// A 4-million-parameter model lives in the board's flash and writes small programs ("draw a smiley" -> code).
// For every token it writes, each of its 36 attention heads decides how much to look at each earlier token.
// This sketch keeps the row of ONE head (runtime/attention.h) and shows the last eight rows on an 8x8 square
// of LEDs (leds.h). The screen shows the text being written and which head is on the LEDs.
//
//   BOOT button      the next head
//   over serial      a request ("draw a house"), or "@head 3 4" (layer 3, head 4), or "@tri" (wiring test)
//   left alone       it writes one request after another by itself
//
// The model runtime (runtime/llm.h) is slvDev/esp32-ai's, verified there against PyTorch; see THIRD_PARTY.md.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#define LLM_INT8_ACT 1                  // int8 activations, required by the staged int8 kernel
#define LLM_ATTN_VIZ 1                  // keep one head's weights: runtime/attention.h
#include "../../runtime/llm.h"
#include "../../runtime/bpe_tokenizer.h"
#include "generated/vocab.h"
#include "generated/tokenizer_encoder.h"
#include "generated/prompt_format.h"
#include "display.h"
#include "leds.h"

#ifndef ATTN_LAYER
#define ATTN_LAYER 3                    // the head shown at start. In the model that ships here, this one of the 36 keeps
#define ATTN_HEAD 4                     // 91 % of its weight inside the last 8 tokens and changes most from token to token
#endif                                  // (measured on the chip). Another model: press BOOT until you like one.
#define IDLE_MS 4000                    // left alone this long, it gives itself the next request
static const char *REQUESTS[] = {REQUEST_LIST};    // generated/prompt_format.h
static const int N_GENERATE = 240;

static BpeTokenizer tokenizer;          // encodes typed requests on the device itself
static int eot_id = -1;                 // the end-of-text token: a program stops there

// --- serial and the button ------------------------------------------------------------------------------------
static char in_buf[256];
static size_t in_len = 0;
static char pending_line[256];
static bool pending_ready = false;

Model model;
Scratch s;

static void head_label() {              // the screen's last row: which head the LEDs show
  char row[22];
  snprintf(row, sizeof row, "layer %d  head %d", g_attn.layer, g_attn.head);
  oled.fillRect(0, 54, SCR_W, 10, 0);
  oled.drawFastHLine(0, 54, SCR_W, OLED_WHITE);
  oled.setCursor(0, 56); oled.print(row);
  oled.display();
}
static void set_head(int layer, int head) {             // out of range wraps; a new head starts with a dark square
  int L = model.c.n_layers, H = model.c.n_heads;
  if (head >= H) { head = 0; layer++; }
  g_attn.layer = ((layer % L) + L) % L; g_attn.head = ((head % H) + H) % H;
  attn_test = false; attn_rows_clear(); attn_show(); head_label();
  Serial.printf("attention: layer %d head %d\n", g_attn.layer, g_attn.head);
}

// Polled between tokens as well as when idle. A typed line parks in pending_line until loop() takes it;
// BOOT changes the head without interrupting what is being written.
static bool poll_input() {
  if (pending_ready) return true;
  static bool was = false; static unsigned long at = 0;
  bool down = digitalRead(0) == LOW;
  if (down && !was && millis() - at > 300) { at = millis(); was = true; set_head(g_attn.layer, g_attn.head + 1); }
  if (!down) was = false;
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      in_buf[in_len] = 0;
      size_t had = in_len;
      in_len = 0;
      if (had) { memcpy(pending_line, in_buf, had + 1); pending_ready = true; return true; }
    } else if (in_len < sizeof(in_buf) - 1) {
      in_buf[in_len++] = ch;
    }
  }
  return false;
}

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (p) psram_used += n;
  return p;
}
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) {
    Serial.printf("FATAL: required PSRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  return p;
}
static void *sram_or_die(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("FATAL: required SRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  sram_used += n;
  return p;
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core.
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}

static void matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

// Copy RMSNorm weights from mapped flash to internal SRAM.
static void copy_norms_to_sram() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim;
  const float **vecs[3 * 32 + 2];
  int sizes[3 * 32 + 2], n_vec = 0;
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = sram_or_die(bytes, "norm vector");
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
  }
  Serial.printf("norms  -> SRAM   %d vectors\n", n_vec);
}

static void alloc_scratch() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, S = c->seq_len;
  // hot working set -> internal SRAM
  s.x     = (float *)sram_or_die(D * 4, "x");
  s.h     = (float *)sram_or_die((F > D ? F : D) * 4, "h");
  s.qkv   = (float *)sram_or_die(3 * D * 4, "qkv");
  s.att   = (float *)sram_or_die(D * 4, "att");
  s.g1    = (float *)sram_or_die(F * 4, "g1");
  s.g2    = (float *)sram_or_die((P > F ? P : F) * 4, "g2");
  s.ple   = (float *)sram_or_die(L * P * 4, "ple");
  s.tmpP  = (float *)sram_or_die(L * P * 4, "tmpP");
  s.trow  = (float *)sram_or_die(L * P * 4, "trow");
  s.scores = (float *)sram_or_die(S * 4, "scores");
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
  // KV cache: 1.1MB, read once per position rather than per matvec.
  s.kcache = (float *)ps_or_die((size_t)L * S * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * S * D * 4, "vcache");
}

static void emit(int tok) {             // one token to serial and to the screen
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);   // never stall when nobody is listening
  display_puts(bytes, len);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== attention in your hand ===");
  pinMode(0, INPUT_PULLUP);             // BOOT

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads,
                c->ffn, c->ple_dim, part->size / 1e6);

  display_begin();
  display_bottom = 54;                  // the last row belongs to the head's label
  leds.begin();

  // The model header states how many logits it produces; vocab.h carries the
  // decode table. If they disagree, every emitted token would be decoded
  // against the wrong table.
  if (VOCAB_N != model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: vocab.h %d, model %d\n",
                  VOCAB_N, model.out_vocab);
    return;
  }

  alloc_scratch();
  copy_norms_to_sram();
  Serial.printf("hot set-> SRAM   %u B dynamic + %u B static = %u B managed\n",
                (unsigned)sram_used, (unsigned)STATIC_SRAM_BYTES,
                (unsigned)(sram_used + STATIC_SRAM_BYTES));

  // Stage every per-position tensor to int8 in PSRAM.
  int want = llm_core_stage_count(&model);
  int staged = llm_stage_core_int8_alloc(&model, ps);
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
  // The tied head is read per token, not per position, so the core helper does
  // not walk it. Stage it too: it is 85%% of the dense MACs.
  {
    void *b = ps_or_die(llm_stage_int8_bytes(&model.out_head), "staged head");
    llm_stage_int8(&model.out_head, b);
    ++staged;
  }
  Serial.printf("weights-> PSRAM  %d tensors int8, %.2f MB allocated\n",
                staged, psram_used / 1048576.0);

  main_h = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(worker_main, "mv", 4096, NULL, 2, &worker_h, 0) == pdPASS) {
    // After the worker exists: matvec_par notifies worker_h.
    model.layer_matvec = matvec_par;
    model.head_matvec  = matvec_par;
  } else {
    Serial.println("dual-core worker failed; running single core");
  }

  // FNV-1a over the mapped image. scripts/deploy.sh prints the same value for
  // the file it flashed; the two must agree.
  {
    const uint8_t *img = (const uint8_t *)base;
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < model.image_bytes; i++) { fp ^= img[i]; fp *= 16777619u; }
    Serial.printf("build: bytes=%u fp=%08x sram=%uB psram=%.2fMB\n",
                  (unsigned)model.image_bytes, fp,
                  (unsigned)(sram_used + STATIC_SRAM_BYTES),
                  psram_used / 1048576.0);
  }
  Serial.printf("free: sram %.0f KB | psram %.2f MB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);

  if (bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE, &tokenizer))
    Serial.println("tokenizer load failed; requests disabled");
  {
    static const char eot[] = "<|endoftext|>";
    const int n = (int)sizeof(eot) - 1;
    for (int v = 0; v < VOCAB_N; v++)
      if (VOCAB_OFF[v + 1] - VOCAB_OFF[v] == n &&
          memcmp(VOCAB_BLOB + VOCAB_OFF[v], eot, (size_t)n) == 0) { eot_id = v; break; }
    Serial.printf("eot id: %d\n", eot_id);
  }
  g_attn.layer = ATTN_LAYER % model.c.n_layers; g_attn.head = ATTN_HEAD % model.c.n_heads;
  head_label();
  Serial.println("ready; type a request (draw a house), @head 3 4, or @tri");
}

// Write one program: the request first, then the model's own tokens until it ends the document.
// After every token the LEDs get the watched head's newest row.
static void write_program(const char *request) {
  char wrapped[320];
  snprintf(wrapped, sizeof(wrapped), "%s%s%s", PROMPT_PREFIX, request, PROMPT_SUFFIX);
  uint16_t ids[96];
  int n = bpe_encode_ascii(&tokenizer, wrapped, ids, (int)(sizeof(ids) / sizeof(ids[0])));
  if (n <= 0) { Serial.println("(ascii only, or request too long)"); return; }

  Serial.print(">>> ");
  display_home(); head_label();
  attn_test = false;
  int pos = 0, tok = 0;
  for (int i = 0; i < n; i++) {         // the request
    tok = ids[i];
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
    attn_show();
  }
  int decoded = 0;
  int64_t t0 = esp_timer_get_time();
  for (int step = 0; step < N_GENERATE && pos < model.c.seq_len; step++) {
    int best = 0; float bv = -1e30f;    // greedy: the same request always gives the same program
    for (int v = 0; v < model.out_vocab; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    tok = best;
    if (tok == eot_id) break;
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
    attn_show();
    decoded++;
    if ((step & 7) == 0) delay(0);      // feed the watchdog
    if (poll_input()) { Serial.println("\n(interrupted)"); return; }
  }
  float secs = (esp_timer_get_time() - t0) / 1e6f;
  Serial.printf("\n\n--- %d tokens in %.2f s (%.2f tok/s) ---\n", decoded, secs, decoded / secs);
}

void loop() {
  static unsigned long idle_since = millis();
  static int next = 0;
  if (poll_input()) {
    pending_ready = false;
    int l, h;
    if (!strcmp(pending_line, "@tri")) { attn_triangle(); Serial.println("triangle: red, white diagonal, green = the newest token on itself"); }
    else if (sscanf(pending_line, "@head %d %d", &l, &h) == 2) set_head(l, h);
    else if (!strcmp(pending_line, "@head")) set_head(g_attn.layer, g_attn.head + 1);
    else if (pending_line[0] != '@') write_program(pending_line);
    idle_since = millis();
  } else if (!attn_test && millis() - idle_since > IDLE_MS) {
    write_program(REQUESTS[next++ % (int)(sizeof(REQUESTS) / sizeof(REQUESTS[0]))]);
    idle_since = millis();
  }
  delay(20);
}
