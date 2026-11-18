// The 8x8 square of WS2812B LEDs, and one attention head drawn on it.
//
// The whole frame is one waveform, handed to the SPI peripheral with DMA: the hardware clocks it out in one piece
// and nothing the chip is doing meanwhile can stretch a pulse. Four SPI bits are one LED bit at 3.2 MHz:
// 1000 is a zero (0.31 us high), 1110 is a one (0.94 us high), a bit lasts 1.25 us.
//
// What failed first, on this board, while the model was writing: the NeoPixel library and the RMT peripheral driven
// directly (both need an interrupt to refill a 48-symbol buffer every ~30 us; a refill comes late and the LEDs read
// noise: rows of full white, then random colours), and SPI fed by the CPU in 64-byte pieces (a wrong pixel now
// and then). A square that stands still looks fine with all of them. The fault only shows when the chip is busy.
#ifndef LEDS_H
#define LEDS_H

#include "driver/spi_master.h"

#ifndef PIN_LEDS
#define PIN_LEDS 21            // the square's IN (also printed DIN)
#endif
#ifndef LEDS_FLIP_X
#define LEDS_FLIP_X 0          // 1 if the picture comes out mirrored
#endif
#ifndef LEDS_SERPENTINE
#define LEDS_SERPENTINE 0      // 1 if every other row of your square runs backwards
#endif
#ifndef LEDS_BRIGHTNESS
#define LEDS_BRIGHTNESS 40     // of 255. 64 LEDs at full white would pull over 3 A; this stays far below USB's limit
#endif

struct Leds64 {
  uint8_t px[64][3]; bool up = false;
  spi_device_handle_t dev = NULL; uint8_t *wave = NULL;
  enum { LEAD = 4, BODY = 64 * 24 / 2, TAIL = 44 };            // TAIL: the line low for > 100 us, the LEDs latch
  void begin() {
    spi_bus_config_t bus = {}; bus.mosi_io_num = PIN_LEDS; bus.miso_io_num = -1; bus.sclk_io_num = -1;
    bus.quadwp_io_num = -1; bus.quadhd_io_num = -1; bus.max_transfer_sz = LEAD + BODY + TAIL;
    spi_device_interface_config_t cfg = {}; cfg.clock_speed_hz = 3200000; cfg.mode = 0; cfg.spics_io_num = -1; cfg.queue_size = 1;
    wave = (uint8_t *)heap_caps_calloc(1, LEAD + BODY + TAIL, MALLOC_CAP_DMA);
    up = wave && spi_bus_initialize(SPI3_HOST, &bus, SPI_DMA_CH_AUTO) == ESP_OK && spi_bus_add_device(SPI3_HOST, &cfg, &dev) == ESP_OK;
    clear(); show();
  }
  void clear() { memset(px, 0, sizeof px); }
  void set(int i, int r, int g, int b) {                        // the wire order is G R B
    if (i < 0 || i >= 64) return;
    px[i][0] = g < 0 ? 0 : g > 255 ? 255 : g; px[i][1] = r < 0 ? 0 : r > 255 ? 255 : r; px[i][2] = b < 0 ? 0 : b > 255 ? 255 : b;
  }
  void show() {
    if (!up) return;
    int n = LEAD;
    for (int i = 0; i < 64; i++)
      for (int c = 0; c < 3; c++) {
        uint8_t v = (uint16_t)px[i][c] * LEDS_BRIGHTNESS / 255;
        for (int bit = 7; bit >= 0; bit -= 2)                   // two LED bits in one SPI byte
          wave[n++] = ((v & (1 << bit)) ? 0xE0 : 0x80) | ((v & (1 << (bit - 1))) ? 0x0E : 0x08);
      }
    spi_transaction_t t = {}; t.length = (LEAD + BODY + TAIL) * 8; t.tx_buffer = wave;
    up = spi_device_transmit(dev, &t) == ESP_OK;
  }
};
static Leds64 leds;

// Row r, column c of the attention square -> the LED's place in the chain. The newest row is the chain's first
// eight LEDs, next to the wires; the corner opposite the wires is never lit.
static int led_at(int c, int r) {
  int x = LEDS_FLIP_X ? 7 - c : c, y = ATTN_WIN - 1 - r;
  if (LEDS_SERPENTINE && (y & 1)) x = 7 - x;
  return y * 8 + x;
}

// One head, as the model writes: row = the token that looks, column = the token it looks at, the last 8 tokens
// both ways. Brightness is the square root of the weight (the eye is not linear either). The half above the
// diagonal stays dark: the causal mask.
static bool attn_test = false;                          // the test picture stays until the next sentence or head
static void attn_show() {
  static const uint8_t col[3] = {255, 110, 0};
  if (attn_test) return;
  leds.clear();
  for (int r = 0; r < ATTN_WIN; r++)
    for (int c = 0; c <= r; c++) {
      float b = sqrtf(attn_cell(r, c));
      if (b < 0.04f) continue;
      leds.set(led_at(c, r), (int)(col[0] * b), (int)(col[1] * b), (int)(col[2] * b));
    }
  leds.show();
}

// The shape attention must have, standing still, to check the wiring and which way round the square is:
// a red triangle, its diagonal white, and green where the newest token looks at itself.
static void attn_triangle() {
  leds.clear();
  for (int r = 0; r < ATTN_WIN; r++)
    for (int c = 0; c <= r; c++) leds.set(led_at(c, r), c == r ? 200 : 120, c == r ? 200 : 0, c == r ? 200 : 0);
  leds.set(led_at(ATTN_WIN - 1, ATTN_WIN - 1), 0, 255, 0);
  attn_test = true; leds.show();
}

#endif
