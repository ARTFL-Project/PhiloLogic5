#!/bin/bash
# Set up philologic5-webui-loader as a service (systemd), which the web server of the databases serves on their host,
# at https://<their host>/philologic5-webui-loader/:
#   sudo extras/webui_loader/setup_service.sh        (install.sh runs it)
# With webui_loader = False in the global config, it stops and disables the service instead.
# It creates the account of the service in the group of database_root and its state directory, adds its settings to
# the global config (webui_loader_* keys, those it doesn't set yet), installs the systemd unit and starts the service,
# and says what to add to the web server if it doesn't serve the UI yet. Variables: WEBUI_USER (philologic), DRY_RUN=1
# to show what it would do. See docs/webui_loader.md.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${WEBUI_PYTHON:-/var/lib/philologic5/philologic_env/bin/python3}
GLOBAL_CONFIG=${PHILOLOGIC_CONFIG:-/etc/philologic/philologic5.cfg}
STATE_DIR=/var/lib/philologic5-webui-loader
SERVICE_USER=${WEBUI_USER:-philologic}
UNIT=/etc/systemd/system/philologic5-webui-loader.service

run() {
    if [ -n "$DRY_RUN" ]; then echo "+ $*"; else "$@"; fi
}

# write_file path mode owner:group (content on stdin)
write_file() {
    if [ -n "$DRY_RUN" ]; then
        echo "+ write $1 ($2, $3):"
        sed 's/^/    /'
    else
        install -m "$2" -o "${3%%:*}" -g "${3##*:}" /dev/stdin "$1"
    fi
}

if [ -z "$DRY_RUN" ] && [ "$(id -u)" != 0 ]; then
    echo "Run as root (sudo)." >&2
    exit 1
fi
if [ ! -d /run/systemd/system ]; then
    echo "systemd is not running here: the service can't be installed (start it with philologic5-webui-loader serve)." >&2
    exit 1
fi

# Whether the UI is on, database_root, the path of the UI, and where the web server passes requests on to
VALUES=$("$PYTHON" -c "
import sys
from philologic.webui_loader.settings import SettingsError, read_global_config, service_settings
try:
    values = read_global_config(sys.argv[1])
except SettingsError as error:
    print('UNSET'); print(error)
else:
    if not values['enabled']:
        print('OFF')
    else:
        settings = service_settings(sys.argv[1], check=False)
        host, port = settings.bind.rsplit(':', 1)
        print('ON'); print(settings.database_root); print(settings.url_prefix)
        print(f\"http://{'127.0.0.1' if host in ('', '0.0.0.0') else host}:{port}\")
" "$GLOBAL_CONFIG")
{ read -r STATE; read -r DATABASE_ROOT; read -r URL_PATH; read -r TARGET; } <<< "$VALUES" || true
if [ "$STATE" = OFF ]; then
    echo "philologic5-webui-loader is turned off (webui_loader = False in $GLOBAL_CONFIG): not set up."
    if [ -f "$UNIT" ]; then
        run systemctl disable --now philologic5-webui-loader
    fi
    exit 0
fi
if [ "$STATE" != ON ]; then
    echo "$DATABASE_ROOT"  # the message of the error
    echo "The web UI loader isn't set up: once $GLOBAL_CONFIG is right, run sudo $SCRIPT_DIR/setup_service.sh"
    exit 0
fi
# The UI takes its address from the requests: this one, from the name of the machine, is only to check and say it
HOST_NAME=$(hostname -A 2> /dev/null | awk '{print $1}')
[ -n "$HOST_NAME" ] || HOST_NAME=$(hostname -f 2> /dev/null || hostname)
PUBLIC_URL="https://$HOST_NAME$URL_PATH"
GROUP=$(stat -c %G "$DATABASE_ROOT")
echo "database_root: $DATABASE_ROOT (group $GROUP), the UI at $PUBLIC_URL/"

# The account of the service, in the group of database_root: it runs the loads and owns the databases they create
if id "$SERVICE_USER" > /dev/null 2>&1; then
    run usermod -a -G "$GROUP" "$SERVICE_USER"
else
    run useradd --system --home-dir "$STATE_DIR" --create-home --gid "$GROUP" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
# New databases belong to the group, so that its members can still manage them from a shell
run chmod g+ws "$DATABASE_ROOT"

run install -d -m 750 -o "$SERVICE_USER" -g "$GROUP" "$STATE_DIR" "$STATE_DIR/uploads" "$STATE_DIR/numba_cache"

# The settings of the service, added to the global config: only those it doesn't set yet
SWITCH=""
if ! grep -q "^webui_loader[[:space:]]*=" "$GLOBAL_CONFIG"; then
    SWITCH="# philologic5-webui-loader, the web UI to load databases: False turns it off"$'\n'"webui_loader = True"$'\n'
fi
ADDED=""
add_setting() {
    if grep -q "^$1[[:space:]]*=" "$GLOBAL_CONFIG"; then
        echo "Keeping $1 of $GLOBAL_CONFIG"
    else
        ADDED+="$1 = $2"$'\n'
    fi
}
add_setting webui_loader_state_dir "\"$STATE_DIR\""
add_setting webui_loader_allowed_roots "[]"
add_setting webui_loader_upload_dir "\"$STATE_DIR/uploads\""
add_setting webui_loader_upload_quota "21474836480"
if [ -n "$SWITCH$ADDED" ]; then
    {
        echo ""
        printf "%s" "$SWITCH"
        if [ -n "$ADDED" ]; then
            echo "# Its service: the directories of corpora it loads from, uploads (see extras/webui_loader/settings.cfg)"
            printf "%s" "$ADDED"
        fi
    } | if [ -n "$DRY_RUN" ]; then echo "+ append to $GLOBAL_CONFIG:"; sed 's/^/    /'; else tee -a "$GLOBAL_CONFIG" > /dev/null; fi
fi

# The unit
sed -e "s|^User=.*|User=$SERVICE_USER|" -e "s|^Group=.*|Group=$GROUP|" "$SCRIPT_DIR/philologic5-webui-loader.service" |
    write_file "$UNIT" 644 root:root
run systemctl daemon-reload

echo ""
echo "## philologic5-webui-loader ##"
PROBLEM=$("$PYTHON" -c "
import sys
from philologic.webui_loader.settings import SettingsError, service_settings
try:
    service_settings(sys.argv[1])
except SettingsError as error:
    print(error)
" "$GLOBAL_CONFIG")
if [ -n "$PROBLEM" ] && [ -z "$DRY_RUN" ]; then
    echo "Not started: $PROBLEM"
    echo "Fix it, then run sudo $SCRIPT_DIR/setup_service.sh again."
    exit 0
fi
run systemctl enable philologic5-webui-loader
run systemctl restart philologic5-webui-loader
echo "Running (check it with: sudo systemctl status philologic5-webui-loader)"

# Whether the web server serves it yet
SERVED=""
if [ -z "$DRY_RUN" ] && command -v curl > /dev/null; then
    # From the web server of this machine, under its name (whatever its certificate and DNS)
    SERVED=$(curl -sk --resolve "$HOST_NAME:443:127.0.0.1" --max-time 5 --retry 5 --retry-delay 1 "$PUBLIC_URL/api/session" |
        grep -o '"mode": "service"' || true)
fi
if [ -n "$SERVED" ]; then
    echo "Your web server serves it at $PUBLIC_URL/"
else
    echo "Have the web server of the databases serve it at $PUBLIC_URL/: add, where it serves them over HTTPS,"
    echo "=== Apache === (sudo a2enmod proxy proxy_http headers), in its <VirtualHost *:443>:"
    echo "    <Location \"$URL_PATH\">"
    echo "        ProxyPass \"$TARGET$URL_PATH\" timeout=300"
    echo "        ProxyPassReverse \"$TARGET$URL_PATH\""
    echo "        RequestHeader set X-Forwarded-Proto \"https\""
    echo "    </Location>"
    echo "=== Nginx === in its server block, see $SCRIPT_DIR/nginx.conf"
    echo "then reload the web server."
fi
echo "Then:"
echo "  - put the directories of your corpora in webui_loader_allowed_roots, in $GLOBAL_CONFIG (then restart the service)"
echo "  - create the first admin: sudo -u $SERVICE_USER philologic5-webui-loader users add YOUR_NAME --admin"
