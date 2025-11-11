# Raspberry Pi 4/5 Deployment Guide

This guide walks through installing **Ai:oS** on Raspberry Pi 4 or Raspberry Pi 5 hardware.  
It targets Raspberry Pi OS (64‑bit) “bookworm” or any modern aarch64 Linux distribution.

## 1. Hardware & OS prerequisites

- Raspberry Pi 4 (4 GB+) or Raspberry Pi 5 (8 GB recommended).
- 64‑bit Raspberry Pi OS Lite/Desktop (other aarch64 Debian derivatives work, but commands below use `apt`).
- Reliable 32 GB+ microSD or NVMe boot. For heavier workloads prefer SSD/NVMe.
- Internet access for first-time package installation.

### Recommended tuning

1. Update firmware & OS:
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```
2. Expand swap to at least 2 GB (4 GB for Whisper/Torch):
   ```bash
   sudo dphys-swapfile swapoff
   echo CONF_SWAPSIZE=4096 | sudo tee /etc/dphys-swapfile
   sudo dphys-swapfile setup && sudo dphys-swapfile swapon
   ```
3. Enable 64‑bit userland (for Pi 4) via `raspi-config` if not already, then reboot.

## 2. Clone or copy Ai:oS

On the Pi:

```bash
mkdir -p ~/src && cd ~/src
# If you already have the tree elsewhere, copy it here instead of cloning.
git clone <your-aios-repo-url> aios
cd aios
```

If you transferred the repository by other means, ensure the new files from this update are present:

- `INSTALL_AIOS_RPI.sh`
- `requirements-rpi.txt`
- `docs/raspberry-pi-deployment.md`

## 3. Run the Raspberry Pi installer

The installer bootstraps system packages, creates a Python virtual environment, and drops a helper launcher.

```bash
chmod +x INSTALL_AIOS_RPI.sh
./INSTALL_AIOS_RPI.sh
```

Installer behaviour:
- Installs core system packages (`python3-venv`, `portaudio19-dev`, `ffmpeg`, etc.).
- Creates (or reuses) `.venv/` in the project root, then installs packages from `requirements-rpi.txt`.
- Generates `aios-rpi.sh` – a wrapper that activates the virtualenv and launches `aios_shell.py`.

### Optional flags

| Flag | Purpose |
| ---- | ------- |
| `--with-torch` | Attempts to install PyTorch for aarch64 via the nightly CPU wheel. Requires 64‑bit OS and adequate swap. |
| `--prefix=/opt/aios` | Deploys the project to a different directory (copies files there). |
| `--skip-apt` | Skips `apt` package installation (useful on air-gapped or pre-provisioned images). |

Environment alternatives:

```bash
AIOS_RPI_WITH_TORCH=1 ./INSTALL_AIOS_RPI.sh    # same as --with-torch
AIOS_RPI_PREFIX=/opt/aios ./INSTALL_AIOS_RPI.sh
```

## 4. Launch Ai:oS

Inside the install directory:

```bash
# One-time session (activates virtualenv manually)
source .venv/bin/activate
python aios_shell.py boot

# or use the helper
./aios-rpi.sh boot
```

The helper accepts any command supported by `aios_shell.py`:

```bash
./aios-rpi.sh status
./aios-rpi.sh prompt "scan the LAN and summarise findings"
```

### Running as a service (optional)

Create `/etc/systemd/system/aios.service`:

```ini
[Unit]
Description=Ai:oS Raspberry Pi Runtime
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/src/aios
ExecStart=/home/pi/src/aios/aios-rpi.sh boot
Restart=on-failure
Environment=AIOS_PROFILE=rpi

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aios.service
```

## 5. Updating

```bash
cd /home/pi/src/aios
git pull
./INSTALL_AIOS_RPI.sh --skip-apt
```

The script is idempotent: it refreshes the virtualenv and launcher without overwriting configuration files outside `.venv/`.

## 6. Feature notes & limitations

- **PyTorch / deep learning** – Only installed when `--with-torch` is supplied. Expect long compile times if fallback source build triggers. Ensure ≥4 GB swap.
- **OpenAI Whisper & ElevenLabs** – Not included in `requirements-rpi.txt` because upstream wheels are x86_64-only. For speech workloads use lightweight alternatives (e.g. Vosk) or offload to another host.
- **Numba** – Not available for ARM; features that require it remain disabled on Pi builds.
- **Quantum/backends** – Qiskit CPU simulators run but are slow; keep circuits small or offload to a desktop/cloud instance.
- **GPU acceleration** – Raspberry Pi GPU drivers do not accelerate PyTorch by default. Use CPU inference or network a more capable accelerator.

## 7. Troubleshooting

| Symptom | Resolution |
| ------- | ---------- |
| `pip` runs out of memory installing numpy/scipy | Increase swap as shown above; ensure `libatlas-base-dev` is installed; rerun installer. |
| PyTorch wheel download fails | Rerun `AIOS_RPI_WITH_TORCH=1 ./INSTALL_AIOS_RPI.sh` after confirming network access; consider using an offline wheel cache. |
| Audio capture errors | Confirm `portaudio19-dev` is installed and the executing user is in the `audio` group (`sudo usermod -aG audio pi`). |
| mitmproxy/scapy permissions | Run under sudo or grant necessary capabilities (e.g. `sudo setcap cap_net_raw+eip $(readlink -f $(which python3))`). |

For additional questions, review the main `README.md` and system architecture docs in `docs/`.
