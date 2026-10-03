#!/usr/bin/env python3
"""Smoke checks for the 2x Spark server: text answer, tool call, vision. No output caps. Prints PASS/FAIL per check.

  python3 scripts/smoke.py [http://<head>:8000]
API key: $DSV41_API_KEY, else the file $KEY_FILE (default ~/.dsv41-two-sparks/api_key, written by launch.sh)."""
import base64, json, os, struct, sys, urllib.request, zlib

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
KEY = os.environ.get("DSV41_API_KEY") or \
    open(os.path.expanduser(os.environ.get("KEY_FILE", "~/.dsv41-two-sparks/api_key"))).read().strip()


def call(payload):
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


def model():
    r = urllib.request.Request(URL + "/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(r, timeout=30))["data"][0]["id"]


def png_quadrants():
    """128x128 PNG: left half red, right half blue."""
    w = h = 128
    rows = b"".join(b"\x00" + b"".join((b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff") for x in range(w)) for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


M = model(); ok = True
r = call({"model": M, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}], "temperature": 0})
txt = r["choices"][0]["message"]["content"] or ""
print("text  ", "PASS" if "391" in txt else "FAIL", repr(txt[-120:]), r["usage"]); ok &= "391" in txt

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = call({"model": M, "messages": [{"role": "user", "content": "What's the weather in Warsaw right now?"}], "tools": tools,
          "temperature": 0})
tc = r["choices"][0]["message"].get("tool_calls") or []
good = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "warsaw" in tc[0]["function"]["arguments"].lower()
print("tools ", "PASS" if good else "FAIL", json.dumps(tc)[:200]); ok &= good

img = "data:image/png;base64," + base64.b64encode(png_quadrants()).decode()
r = call({"model": M, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": img}},
    {"type": "text", "text": "This image has two halves. What colour is the left half and what colour is the right half?"}]}]})
txt = (r["choices"][0]["message"]["content"] or "").lower()
good = "red" in txt and "blue" in txt and txt.find("red") < txt.find("blue")
print("vision", "PASS" if good else "FAIL", repr(txt[-160:]), r["usage"]); ok &= good
sys.exit(0 if ok else 1)
