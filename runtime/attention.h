// One attention head of the model, kept as a square: the last 8 tokens looking at the last 8 tokens.
//
// llm_forward() computes, for every head, how much the token being written looks at each earlier token: a row of
// weights that sums to one. It uses the row and throws it away. With -DLLM_ATTN_VIZ the row of ONE head is kept
// here, and the last eight rows make the square the LEDs show.
#ifndef ATTENTION_H
#define ATTENTION_H

#include <math.h>
#include <string.h>

#define ATTN_WIN 8

typedef struct { int layer, head; } AttnViz;
static AttnViz g_attn = {0, 0};

// Row r is one token's look back (r = ATTN_WIN-1 is the newest), kept as its weights on the ATTN_WIN positions
// ending at itself, so g_attn_rows[r][ATTN_WIN-1] is a token looking at itself. What it gave to older tokens
// is not kept.
static float g_attn_rows[ATTN_WIN][ATTN_WIN];
static inline void attn_rows_clear(void) { memset(g_attn_rows, 0, sizeof g_attn_rows); }

// Weight of the token in row r on the token in column c, both counted from the oldest of the last ATTN_WIN.
// Above the diagonal there is nothing: a token cannot look at what has not been written yet.
static inline float attn_cell(int r, int c) { return c > r ? 0.f : g_attn_rows[r][c - r + ATTN_WIN - 1]; }

// Called from the head loop in llm_forward() once the softmax denominator is known. The weights are not stored
// there, but the scores, their maximum and the denominator are still in scope: weight = exp(score - max) / denom.
static inline void attn_capture(const float *scores, float maxs, float denom, int pos) {
  if (pos == 0) attn_rows_clear();                       // a new text
  memmove(g_attn_rows[0], g_attn_rows[1], sizeof(float) * ATTN_WIN * (ATTN_WIN - 1));
  for (int j = 0; j < ATTN_WIN; j++) {
    int k = pos - (ATTN_WIN - 1) + j;
    g_attn_rows[ATTN_WIN - 1][j] = k >= 0 ? expf(scores[k] - maxs) / denom : 0.f;
  }
}

#endif
