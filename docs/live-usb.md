# AgentaOS Live USB Builder

This folder ships tooling for creating a reproducible live image that boots AgentaOS straight into the runtime without requiring an ISO download. The process relies on Debian's `live-build` so the result is an `iso-hybrid` image that can be flashed to a USB drive directly.

## Prerequisites

- Docker (recommended) **or** a Debian/Ubuntu host with `live-build`, `debootstrap`, `squashfs-tools`, and `xorriso` installed.
- At least 12 GB of free disk space for the build artefacts.

## Building with Docker

```bash
# From the repository root
docker build -t agentaos-live build

docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  agentaos-live
```

The container executes `build/build-image.sh`, which:

1. Syncs the repo into `build/live-build/config/includes.chroot/opt/agentaos`.
2. Runs `lb config` with the Debian bookworm presets.
3. Generates `dist/agentaos.img` (an `iso-hybrid` image).

## Building on the host

```bash
sudo apt-get update
sudo apt-get install live-build debootstrap squashfs-tools xorriso syslinux-common rsync

bash build/build-image.sh
```

> The script uses `rsync` with `--delete`; do not point `ROOT_DIR` at a path that contains uncommitted work you are unwilling to mirror inside the chroot.

## Flashing to USB

Once `dist/agentaos.img` exists:

```bash
sudo dd if=dist/agentaos.img of=/dev/sdX bs=4M status=progress oflag=sync
```

Replace `/dev/sdX` with the target device (not a partition). After flashing, create an optional third partition for encrypted persistence:

1. Partition layout suggestion:
   - **Partition 1** – EFI System Partition (~512 MB, FAT32) populated by live-build.
   - **Partition 2** – Read-only squashfs (auto-generated).
   - **Partition 3** – Optional LUKS volume for exporting artefacts.
2. Initialize the persistence container when needed:

   ```bash
   sudo cryptsetup luksFormat /dev/sdX3
   sudo cryptsetup open /dev/sdX3 agentaos-persist
   sudo mkfs.ext4 /dev/mapper/agentaos-persist
   ```

Mount the encrypted volume on-demand inside AgentaOS before exporting session data so the default workflow remains stateless.

## Customising the Image

- Edit `build/live-build/config/package-lists/agentaos.list.chroot` to add Debian packages.
- Drop additional files under `build/live-build/config/includes.chroot` (they land at the same path inside the live filesystem).
- Adjust the systemd unit (`build/live-build/config/includes.chroot/usr/lib/systemd/system/agentaos.service`) to change how the runtime boots.

Run `bash build/build-image.sh` again after making changes.
