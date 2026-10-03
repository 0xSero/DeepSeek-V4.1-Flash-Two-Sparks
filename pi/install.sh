#!/bin/bash
# Install the Pi coding agent (if missing) and register the two-Spark DeepSeek-V4.1-Flash server as a Pi provider.
#
#   pi/install.sh http://<head-spark-ip>:8000/v1
#   export DSV41_API_KEY=...        # the key launch.sh wrote to ~/.dsv41-two-sparks/api_key on the head
#   pi --model dsv41-sparks/deepseek-v4.1-flash
#
# What it changes (existing config is kept; a timestamped backup is written before any edit):
#   <pi dir>/models.json             adds or replaces only the "dsv41-sparks" provider
#   <pi dir>/settings.json           if you use an "enabledModels" list, appends dsv41-sparks/deepseek-v4.1-flash to it;
#                                    with SET_DEFAULT=1 also makes it the startup model
#   <pi dir>/extensions/dsv41-sparks.ts   removes Pi's output-token limit from requests to this provider only
# <pi dir> is $PI_CODING_AGENT_DIR or ~/.pi/agent. The API key is read from $DSV41_API_KEY at request time and is
# never written to disk by this script. Override the variable name with KEY_ENV=OTHER_NAME.
set -euo pipefail
BASE_URL=${1:-${DSV41_BASE_URL:-}}
[ -n "$BASE_URL" ] || { echo "usage: $0 http://<head-spark-ip>:8000/v1" >&2; exit 2; }
case "$BASE_URL" in */v1) ;; */) BASE_URL=${BASE_URL}v1 ;; *) BASE_URL=$BASE_URL/v1 ;; esac
KEY_ENV=${KEY_ENV:-DSV41_API_KEY}
HERE=$(cd "$(dirname "$0")" && pwd)
PI_DIR=${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}

if ! command -v pi >/dev/null 2>&1; then
  command -v npm >/dev/null 2>&1 || { echo "Pi needs Node.js + npm (https://nodejs.org); install them and re-run" >&2; exit 1; }
  echo "installing Pi: npm install -g @earendil-works/pi-coding-agent"
  npm install -g @earendil-works/pi-coding-agent || {
    echo "global npm install failed (permissions?). Try: npm config set prefix ~/.local && export PATH=~/.local/bin:\$PATH, then re-run" >&2; exit 1; }
fi
echo "pi: $(command -v pi) ($(pi --version 2>/dev/null | head -1))"

mkdir -p "$PI_DIR/extensions"
python3 - "$PI_DIR" "$HERE/models.fragment.json" "$BASE_URL" "$KEY_ENV" "${SET_DEFAULT:-0}" <<'PY'
import json, os, shutil, sys, time
pi_dir, frag_path, base_url, key_env, set_default = sys.argv[1:6]
stamp = time.strftime("%Y%m%d-%H%M%S")
PROVIDER, MODEL = "dsv41-sparks", "deepseek-v4.1-flash"

def load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        text = f.read().strip()
    return json.loads(text) if text else {}

def save(path, data):
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak-dsv41-sparks-{stamp}")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2); f.write("\n")
    os.replace(tmp, path)

prov = json.load(open(frag_path))["providers"][PROVIDER]
prov["baseUrl"] = base_url
prov["apiKey"] = "$" + key_env

mpath = os.path.join(pi_dir, "models.json")
models = load(mpath)
action = "replaced" if PROVIDER in models.get("providers", {}) else "added"
models.setdefault("providers", {})[PROVIDER] = prov
save(mpath, models)
print(f"{mpath}: {action} provider {PROVIDER} -> {base_url}")

spath = os.path.join(pi_dir, "settings.json")
settings = load(spath); changed = False
ref = f"{PROVIDER}/{MODEL}"
if isinstance(settings.get("enabledModels"), list) and ref not in settings["enabledModels"]:
    settings["enabledModels"].append(ref); changed = True
if set_default == "1":
    settings["defaultProvider"], settings["defaultModel"] = PROVIDER, MODEL; changed = True
if changed:
    save(spath, settings); print(f"{spath}: updated (enabledModels/default)")
PY
cp "$HERE/dsv41-sparks.ts" "$PI_DIR/extensions/dsv41-sparks.ts"
echo "$PI_DIR/extensions/dsv41-sparks.ts: installed"

if [ -z "${!KEY_ENV:-}" ]; then
  echo
  echo "Set the API key before starting Pi (copy it from the head Spark):"
  echo "  export $KEY_ENV=\$(ssh <head-spark> cat ~/.dsv41-two-sparks/api_key)"
fi
echo "Start:  pi --model dsv41-sparks/deepseek-v4.1-flash"
echo "Inside Pi: /model dsv41-sparks/deepseek-v4.1-flash, /thinking to pick the reasoning level."
