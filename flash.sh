#!/bin/sh
# Build the firmware and put it on the board: the model into its flash partition, then the program.
#   ./flash.sh [port]            (default /dev/ttyACM0)
#   BUILD_ONLY=1 ./flash.sh      just compile
#
# Another model that already runs on this chip with slvDev/esp32-ai's runtime: point MODEL at a folder holding its
# model.bin and tokenizer.json, and say what it should be asked ("|" between requests, PREFIX/SUFFIX around each):
#   MODEL=~/maxwell-on-a-chip/model REQUESTS="The temperature of a body is|Heat is" ./flash.sh
#
# Needs arduino-cli with the esp32 core and two libraries for the screen:
#   arduino-cli core install esp32:esp32
#   arduino-cli lib install "Adafruit SSD1306" "Adafruit GFX Library"
set -e
cd "$(dirname "$0")"
PORT=${1:-/dev/ttyACM0}
FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi'
GEN=firmware/attention/generated
if [ -n "${MODEL:-}" ]; then
  [ -f "$MODEL/model.bin" ] && [ -f "$MODEL/tokenizer.json" ] || { echo "$MODEL needs model.bin and tokenizer.json"; exit 1; }
  [ -n "${REQUESTS:-}" ] || { echo 'say what to ask it: REQUESTS="first request|second request"'; exit 1; }
  echo "=== model from $MODEL"
  python3 tools/generate_vocab.py --tokenizer "$MODEL/tokenizer.json" --out $GEN/vocab.h
  python3 tools/generate_tokenizer_header.py --tokenizer "$MODEL/tokenizer.json" --out $GEN/tokenizer_encoder.h > /dev/null
  LIST=$(printf '%s' "$REQUESTS" | sed 's/\\/\\\\/g; s/"/\\"/g; s/|/", "/g')
  printf '// Written by flash.sh for %s\n#define PROMPT_PREFIX "%s"\n#define PROMPT_SUFFIX "%s"\n#define REQUEST_LIST "%s"\n' \
    "$MODEL" "${PREFIX:-}" "${SUFFIX:-}" "$LIST" > $GEN/prompt_format.h
  echo "(the three headers in $GEN now belong to that model; git checkout $GEN brings the shipped one back)"
else
  MODEL=model
fi
ESPTOOL=$(command -v esptool || command -v esptool.py || ls -d "${ARDUINO_DATA_DIR:-$HOME/.arduino15}"/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1)
[ -n "$ESPTOOL" ] || { echo "esptool not found: pip install esptool"; exit 1; }
echo "=== build"
arduino-cli compile --fqbn "$FQBN" --build-path build firmware/attention
[ "${BUILD_ONLY:-0}" = 1 ] && exit 0
echo "=== model -> $PORT (partition 'model' at 0x110000)"
"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 write_flash 0x110000 "$MODEL/model.bin"
echo "=== firmware -> $PORT"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir build firmware/attention
echo "done. It starts writing by itself. To type to it: python talk.py   (pip install pyserial)"
