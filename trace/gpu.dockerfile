FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set up time zone.
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak \
&& echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse" >> /etc/apt/sources.list \
&& echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse" >>/etc/apt/sources.list \
&& echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse" >>/etc/apt/sources.list \
&& echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >>/etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl \
      make \
      wget \
      unzip \
      bash \
      jq \
      libcomerr2 \
      libssl1.0-dev \
      libasl-dev \
      libsasl2-dev \
      pkg-config \
      libsystemd-dev \
      zlib1g-dev

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

RUN conda install -y torchaudio torchdata torchtext torchvision -c pytorch

COPY audio.py /workspace/
COPY lm.py /workspace/
COPY mnist.py /workspace/
COPY tools.py /workspace/
COPY cache.py /workspace/

# fix
RUN apt-get install gawk bison -y
RUN wget https://mirrors.ustc.edu.cn/gnu/glibc/glibc-2.29.tar.gz
RUN tar -zxvf glibc-2.29.tar.gz && cd glibc-2.29 && mkdir build && cd build && ../configure --prefix=/usr/local --disable-sanity-checks && make -j8 && make install
RUN cd /lib/x86_64-linux-gnu && cp /usr/local/lib/libm-2.29.so /lib/x86_64-linux-gnu/ && ln -sf libm-2.29.so libm.so.6

# cache everything
RUN cd /workspace && python cache.py
