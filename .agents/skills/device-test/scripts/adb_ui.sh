#!/usr/bin/env bash
# Text-anchored UI automation for the Android example app.
# Blind `adb input tap <x> <y>` with hardcoded coordinates is fragile: layouts
# shift between devices/orientations, and a tap that lands on the wrong element
# can long-press into an OS context menu or navigate away. These helpers locate
# an element by its visible text in the uiautomator hierarchy and tap its
# CENTER, which is stable across layouts.
#
# Usage:
#   source scripts/adb_ui.sh
#   ui_tap "Simple Chat"      # tap the element whose text contains "Simple Chat"
#   ui_type "hello world"     # type into the focused field (spaces handled)
#   ui_has "Type your message" && echo "on chat screen"
#   ui_dump                   # print all visible text (for discovering anchors)
#   ui_wait "Initialize" 30   # wait up to 30s for text to appear
#
# Everything keys off `uiautomator dump`, so it needs no accessibility grant.

_ui_xml() { adb shell "uiautomator dump /sdcard/ui.xml >/dev/null 2>&1; cat /sdcard/ui.xml" 2>/dev/null; }

# Center "x y" of the first node whose text contains $1. The center is
# computed in awk (not the shell) so this works identically under bash and zsh
# -- zsh does not word-split unquoted variables, which breaks `set -- $bounds`.
_ui_center() {
  _ui_xml | tr '>' '\n' | grep -m1 "text=\"[^\"]*$1" \
    | grep -oE 'bounds="\[[0-9]+,[0-9]+\]\[[0-9]+,[0-9]+\]"' \
    | grep -oE '[0-9]+' | paste -sd' ' - \
    | awk '{ if (NF==4) printf "%d %d", ($1+$3)/2, ($2+$4)/2 }'
}

ui_tap() {
  local c; c=$(_ui_center "$1")
  if [ -z "$c" ]; then echo "ui_tap: no element matching '$1'" >&2; return 1; fi
  echo "tap '$1' -> ${c/ /,}"
  adb shell input tap ${c}
}

# Type text; spaces are sent as %s which `adb input text` maps to a space.
ui_type() { adb shell input text "$(printf '%s' "$1" | sed 's/ /%s/g')"; }

ui_has() { _ui_xml | grep -q "text=\"[^\"]*$1"; }

ui_dump() { _ui_xml | tr '>' '\n' | grep -oE 'text="[^"]{2,60}"' | grep -v '=""'; }

# ui_wait <text> <seconds>
ui_wait() {
  local n=0 max=$(( ${2:-30} / 2 ))
  until ui_has "$1" || [ "$n" -ge "$max" ]; do sleep 2; n=$((n+1)); done
  ui_has "$1"
}
