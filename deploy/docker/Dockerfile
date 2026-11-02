# Build on Thor after checking its host driver. The default is NVIDIA's ARM64 CUDA 13.2 image.
# The GPU driver/kernel remain on the host. All Python, SDK and DDS dependencies live here.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3@sha256:9b1b9d6477afce05a5e941a3711093366b43de1b4eb7c1963251ade6a07c34fe
FROM ${BASE_IMAGE}
USER root
ARG BASE_IMAGE
ARG SOURCE_COMMIT=unknown
ARG CYCLONEDDS_COMMIT=2cdd114cbd18340c606573b4cc8dc20cc161ec5a
ARG UNITREE_SDK_COMMIT=814556d15970dd2ecf1c9984e845ca02ab07e206
LABEL org.opencontainers.image.source="https://github.com/Manouselis/supermimic_vibed_fork" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.supermimic.role="receive-only"

ENV DEBIAN_FRONTEND=noninteractive \
    CYCLONEDDS_HOME=/opt/cyclonedds \
    CMAKE_PREFIX_PATH=/opt/cyclonedds \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_CACHE_PATH=/tmp/cuda-cache \
    XDG_CACHE_HOME=/tmp/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates python3-dev pkg-config libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN git init /tmp/cyclonedds \
    && git -C /tmp/cyclonedds remote add origin https://github.com/eclipse-cyclonedds/cyclonedds.git \
    && git -C /tmp/cyclonedds fetch --depth 1 origin ${CYCLONEDDS_COMMIT} \
    && git -C /tmp/cyclonedds checkout --detach FETCH_HEAD \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
       -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DCMAKE_INSTALL_LIBDIR=lib \
       -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_DDSPERF=OFF -DENABLE_SECURITY=OFF \
    && cmake --build /tmp/cyclonedds/build --parallel 4 \
    && cmake --install /tmp/cyclonedds/build \
    && printf '/opt/cyclonedds/lib\n' > /etc/ld.so.conf.d/supermimic-cyclonedds.conf && ldconfig \
    && rm -rf /tmp/cyclonedds

COPY deploy/requirements.txt /tmp/deployment-requirements.txt
RUN python3 -c 'import torch; print(torch.__version__, file=open("/tmp/torch-version-before", "w"))' \
    && git init /opt/unitree_sdk2_python \
    && git -C /opt/unitree_sdk2_python remote add origin https://github.com/unitreerobotics/unitree_sdk2_python.git \
    && git -C /opt/unitree_sdk2_python fetch --depth 1 origin ${UNITREE_SDK_COMMIT} \
    && git -C /opt/unitree_sdk2_python checkout --detach FETCH_HEAD \
    && python3 -m pip install --no-cache-dir -r /tmp/deployment-requirements.txt -e /opt/unitree_sdk2_python \
    && python3 -c 'import torch; assert torch.__version__ == open("/tmp/torch-version-before").read().strip(), "The base Torch installation changed"' \
    && python3 -c 'from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_; from unitree_sdk2py.utils.crc import CRC; from unitree_sdk2py.core.channel import ChannelSubscriber; CRC(); print("SDK receive-side imports OK")'

# Imports alone missed a configuration crash with native DDS 0.10.2. Exercise the pinned
# maintenance release on isolated loopback during the build. This never creates a publisher.
RUN python3 -c 'from unitree_sdk2py.core.channel import ChannelFactoryInitialize; ChannelFactoryInitialize(173, "lo"); print("DDS loopback initialization OK")'

WORKDIR /app
COPY supermimic/ /app/supermimic/
COPY deploy/ /app/deploy/
COPY docs/RELEASE_LICENSES.md /app/docs/RELEASE_LICENSES.md
COPY tools/test_deployment.py tools/test_deployment_docker.py tools/test_policy_release.py /app/tools/
RUN python3 -c 'import deploy.runtime, torch, sys; assert "isaaclab" not in sys.modules; assert float(torch.ones(4) @ torch.ones(4)) == 4'
ENTRYPOINT ["python3", "-m", "deploy.run"]
CMD ["inspect"]
