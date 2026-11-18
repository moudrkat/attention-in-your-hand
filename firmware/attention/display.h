// The screen: a 128x64 I2C OLED. Four wires: GND->GND, VCC->3V3, SDA->GPIO 18, SCL->GPIO 8.
// 0.96" panels are usually SSD1306, 1.3" usually SH1106: if only the top row is right, switch OLED_CONTROLLER.
#ifndef DISPLAY_H
#define DISPLAY_H

// ==================== 128x64 I2C OLED (SH1106 or SSD1306) ===================
#include <Adafruit_GFX.h>
#include <Wire.h>

// Set to match the panel you wired: 1.3" is usually SH1106, 0.96" usually SSD1306.
// Wrong choice -> shifted/garbled image; just switch this and reflash.
#define OLED_SH1106  1
#define OLED_SSD1306 2
#ifndef OLED_CONTROLLER
#define OLED_CONTROLLER OLED_SSD1306
#endif

#define OLED_SDA 18
#define OLED_SCL 8
#define OLED_ADDR 0x3C          // some panels are 0x3D
#define SCR_W 128
#define SCR_H 64
#define CW 6                    // 6x8 base glyph at text size 1
#define CH 8

#if OLED_CONTROLLER == OLED_SH1106
#include <Adafruit_SH110X.h>
static Adafruit_SH1106G oled = Adafruit_SH1106G(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SH110X_WHITE
#else
#include <Adafruit_SSD1306.h>
static Adafruit_SSD1306 oled = Adafruit_SSD1306(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SSD1306_WHITE
#endif

static int ox = 0, oy = 0;
static int display_bottom = SCR_H;   // text wraps to the top when it reaches this row

static void display_home() {
  oled.clearDisplay();
  ox = 0; oy = 0;
}

static void display_begin() {
  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(400000);   // 400kHz fast I2C -> ~4x quicker frame flush than default
#if OLED_CONTROLLER == OLED_SH1106
  oled.begin(OLED_ADDR, true);
#else
  oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
#endif
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(OLED_WHITE);
  oled.setTextWrap(false);      // we wrap at token boundaries ourselves
  oled.display();
  display_home();
}

// Append a token's bytes; whole-token wrap, clear+home when the screen fills,
// flush the framebuffer once per token so text visibly appears.
static void display_puts(const unsigned char *s, int len) {
  if (ox + len * CW > SCR_W) { oy += CH; ox = 0; }
  if (oy + CH > display_bottom) display_home();
  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') { oy += CH; ox = 0; }
    else if (c >= 32 && c < 127) {
      if (ox + CW > SCR_W) { oy += CH; ox = 0; }
      if (oy + CH > display_bottom) display_home();
      oled.setCursor(ox, oy);
      oled.write(c);
      ox += CW;
    }
    if (oy + CH > display_bottom) display_home();
  }
  oled.display();
}

#endif
