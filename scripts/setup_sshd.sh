#!/bin/sh
# Setup and start sshd on OpenHarmony robot boards.
#
# Usage:
#   sh scripts/setup_sshd.sh          # setup, start, and enable autostart
#   sh scripts/setup_sshd.sh status   # check status only
#   sh scripts/setup_sshd.sh start    # start only (skip setup)
#
# This script remounts / as rw, copies sshd config/libexec from the
# system sysdeps directory, generates host keys, starts sshd, and
# installs an init config so sshd auto-starts on boot.
#
# On BQ3588HM:  sysdeps at /data/out
# On RoboPi:    sysdeps at /sys_prod/robot/out

set -eu
# pipefail may not be available in all POSIX shells (e.g. dash), but
# busybox ash on OpenHarmony supports it; silently ignore if unsupported.
set -o pipefail 2>/dev/null || true

# --- locate sysdeps --------------------------------------------------------

SYSDEPS=""
for candidate in \
    /sys_prod/robot/out \
    /data/out \
    /data/install/out; do
    if [ -f "${candidate}/sbin/sshd" ]; then
        SYSDEPS="${candidate}"
        break
    fi
done

if [ -z "${SYSDEPS}" ]; then
    echo "ERROR: cannot find sshd binary"
    echo "  Checked: /sys_prod/robot/out, /data/out, /data/install/out"
    exit 1
fi

SSHD_BIN="${SYSDEPS}/sbin/sshd"
SSHD_CONFIG="${SYSDEPS}/etc/sshd_config"

echo "SYSDEPS:    ${SYSDEPS}"
echo "SSHD_BIN:   ${SSHD_BIN}"
echo "SSHD_CONFIG:${SSHD_CONFIG}"

# --- helper: check if port 22 is listening ---------------------------------

port22_listening() {
    netstat -tlnp 2>/dev/null | grep ":22 " >/dev/null 2>&1
}

# --- status only -----------------------------------------------------------

if [ "${1:-}" = "status" ]; then
    if port22_listening; then
        echo "sshd: running (port 22 listening)"
        netstat -tlnp 2>/dev/null | grep ":22 "
    else
        echo "sshd: not running"
    fi
    echo ""
    echo "autostart:"
    for f in /vendor/etc/init/init.sshd.cfg /system/etc/init/init.sshd.cfg; do
        if [ -f "$f" ]; then
            echo "  enabled ($f)"
            exit 0
        fi
    done
    echo "  not configured"
    exit 0
fi

# --- start only (skip setup) -----------------------------------------------

if [ "${1:-}" = "start" ]; then
    if port22_listening; then
        echo "sshd is already running"
        exit 0
    fi
    if [ ! -f /etc/sshd_config ]; then
        echo "ERROR: /etc/sshd_config not found, run full setup first"
        exit 1
    fi
    echo "Starting sshd..."
    "${SSHD_BIN}" -f /etc/sshd_config
    for i in 1 2 3 4 5; do
        if port22_listening; then
            echo "sshd started successfully"
            exit 0
        fi
        sleep 1
    done
    echo "ERROR: sshd failed to start (port 22 not listening after 5s)"
    exit 1
fi

# --- check if already running ----------------------------------------------

if port22_listening; then
    echo "sshd is already running (port 22 listening)"
    # still fall through to ensure autostart is configured
else
    # not running, proceed with full setup

# --- remount root as rw ----------------------------------------------------

echo "Remounting / as rw..."
mount -o remount,rw /

# --- create required directories -------------------------------------------

echo "Creating directories..."
mkdir -p /var/empty /var/run /root/.ssh /libexec
chmod 0555 /var/empty

# --- copy sshd config and libexec to root fs (first time only) -------------

if [ ! -f "/etc/sshd_config" ]; then
    echo "Copying sshd config to /etc..."
    cp -r "${SYSDEPS}"/etc/* /etc/ || true
    cp -r "${SYSDEPS}"/libexec/* /libexec/ || true

    # fix root home and shell (match any GECOS field, not just empty)
    sed -i 's|^root:\([^:]*\):0:0:[^:]*:[^:]*:/bin/false|root:\1:0:0::/root:/bin/sh|' /etc/passwd \
        || echo "WARNING: failed to fix root shell in /etc/passwd"

    # permit root login with password (match both commented and uncommented)
    sed -i 's|^#*PermitRootLogin .*|PermitRootLogin yes|' /etc/sshd_config \
        || echo "WARNING: failed to set PermitRootLogin"
    sed -i 's|^#*PasswordAuthentication .*|PasswordAuthentication yes|' /etc/sshd_config \
        || echo "WARNING: failed to set PasswordAuthentication"
else
    echo "/etc/sshd_config already exists, skipping copy"
fi

# --- generate host keys (first time only) ----------------------------------

if [ ! -f "/etc/ssh_host_rsa_key" ]; then
    echo "Generating SSH host keys..."
    export PATH="${SYSDEPS}/bin:${SYSDEPS}/sbin:${PATH}"
    ssh-keygen -A || true
else
    echo "Host keys already exist"
fi

# --- start sshd ------------------------------------------------------------

    echo "Starting sshd..."
    "${SSHD_BIN}" -f /etc/sshd_config
    # wait for port 22 (up to 5 seconds, retry on slow boards)
    for i in 1 2 3 4 5; do
        if port22_listening; then
            break
        fi
        sleep 1
    done
fi

# --- verify ----------------------------------------------------------------

if port22_listening; then
    echo "sshd started successfully"
    netstat -tlnp 2>/dev/null | grep ":22 "
    echo ""
    echo "You can now SSH to this board:"
    echo "  ssh root@<board-ip>"
    echo ""
    echo "To set root password:  passwd root"
else
    echo "ERROR: sshd failed to start (port 22 not listening)"
    exit 1
fi

# --- enable autostart on boot -----------------------------------------------

echo ""
echo "Configuring sshd autostart..."

# Try vendor partition first, then system
AUTOSTART_DIR=""
for d in /vendor/etc/init /system/etc/init; do
    if [ -d "$d" ]; then
        AUTOSTART_DIR="$d"
        break
    fi
done

if [ -n "${AUTOSTART_DIR}" ]; then
    # Extract mount point: /vendor/etc/init -> /vendor, /system/etc/init -> /system
    mount_pt="$(echo "${AUTOSTART_DIR}" | cut -d/ -f1-2)"
    mount -o remount,rw "${mount_pt}" 2>/dev/null || mount -o remount,rw / 2>/dev/null || true

    AUTOSTART_FILE="${AUTOSTART_DIR}/init.sshd.cfg"
    if [ -f "${AUTOSTART_FILE}" ]; then
        echo "Autostart already configured (${AUTOSTART_FILE})"
    else
        # Use unquoted heredoc so ${SSHD_BIN} is expanded to the detected path
        cat > "${AUTOSTART_FILE}" << CFGEOF
{
    "jobs" : [{
            "name" : "post-init",
            "cmds" : [
                "start sshd"
            ]
        }
    ],
    "services" : [{
            "name" : "sshd",
            "path" : ["${SSHD_BIN}", "-f", "/etc/sshd_config"],
            "uid" : "root",
            "gid" : ["root"],
            "once" : 1,
            "importance" : 0,
            "caps" : [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 21, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 200, 201, 210, 1003, 1004, 1007, 1014, 1018, 1021, 1032, 1065]
        }
    ]
}
CFGEOF
        echo "Autostart configured: ${AUTOSTART_FILE}"
    fi
else
    echo "WARNING: no init directory found, autostart not configured"
    echo "  To start manually on boot, add to a startup script:"
    echo "    ${SSHD_BIN} -f /etc/sshd_config"
fi
