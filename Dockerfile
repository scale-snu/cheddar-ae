FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# https://apt.kitware.com/
RUN apt-get update && \
    apt-get install -y ca-certificates gpg wget && \
    test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
    apt-get install -y kitware-archive-keyring && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy-rc main' | tee -a /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake=3.25.2-0kitware1ubuntu22.04.1 cmake-data=3.25.2-0kitware1ubuntu22.04.1 && \
    apt-mark hold cmake && \
    apt-get install -y zlib1g-dev

RUN apt-get update && apt-get install -y \
    build-essential git libtommath-dev python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install pandas seaborn matplotlib

WORKDIR /cheddar
COPY libcheddar_base.so.gz /cheddar/lib/base/libcheddar_base.so.gz
COPY libcheddar_parfuse.so.gz /cheddar/lib/parfuse/libcheddar_parfuse.so.gz
COPY libcheddar_seqfuse.so.gz /cheddar/lib/seqfuse/libcheddar_seqfuse.so.gz

RUN gunzip /cheddar/lib/base/libcheddar_base.so.gz && \
    gunzip /cheddar/lib/parfuse/libcheddar_parfuse.so.gz && \
    gunzip /cheddar/lib/seqfuse/libcheddar_seqfuse.so.gz && \
    chmod 755 /cheddar/lib/base/libcheddar_base.so && \
    chmod 755 /cheddar/lib/parfuse/libcheddar_parfuse.so && \
    chmod 755 /cheddar/lib/seqfuse/libcheddar_seqfuse.so

RUN ln -s /cheddar/lib/base/libcheddar_base.so /cheddar/lib/libcheddar.so


COPY include /cheddar/include
COPY parameters /cheddar/unittest/parameters
COPY unittest /cheddar/unittest

COPY scripts/Experiment1.py /cheddar/
COPY scripts/Experiment2.py /cheddar/
COPY scripts/Experiment3.py /cheddar/
COPY scripts/Experiment4.py /cheddar/
COPY scripts/Experiment4-1.py /cheddar/
COPY MNIST_data /cheddar/unittest/MNIST_data
COPY resnet20_fused /cheddar/unittest/resnet20_fused

ENV LD_LIBRARY_PATH=/cheddar/lib/:$LD_LIBRARY_PATH

RUN mkdir -p /cheddar/unittest/build

WORKDIR /cheddar/unittest/build

RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

WORKDIR /cheddar

CMD ["/bin/bash"]
