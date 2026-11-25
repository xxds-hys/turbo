# Build Turbo from source on Ubuntu 20.04.
FROM ubuntu:20.04 AS environment
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential python3 python3-dev python3-distutils pkg-config \
       openmpi-bin libopenmpi-dev libboost-all-dev libgsl-dev \
       libsqlite3-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

FROM environment AS compiled
WORKDIR /opt/turbo
COPY ns-3.33/ ns-3.33/
COPY file-in-ctr/ file-in-ctr/
COPY scripts/turbo.py scripts/turbo.py
ARG BUILD_JOBS=2
RUN --network=none python3 scripts/turbo.py build --configure --jobs "${BUILD_JOBS}"
COPY . /opt/turbo/
ENTRYPOINT ["python3", "/opt/turbo/scripts/turbo.py"]
CMD ["--help"]
