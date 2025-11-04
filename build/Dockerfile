FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      live-build \
      debootstrap \
      syslinux-common \
      squashfs-tools \
      xorriso \
      dosfstools \
      rsync \
      git \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# The build script expects the repository to be mounted at /workspace.
CMD ["bash", "build/build-image.sh"]
