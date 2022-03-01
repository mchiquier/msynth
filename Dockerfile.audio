# Pulseaudio
#
# docker run -d \
#	-v /etc/localtime:/etc/localtime:ro \
#	--device /dev/snd \
#	--name pulseaudio \
#	-p 4713:4713 \
#	-v /var/run/dbus:/var/run/dbus \
#	-v /etc/machine-id:/etc/machine-id \
#	jess/pulseaudio
#
FROM ubuntu:18.04
LABEL maintainer "Jessie Frazelle <jess@linux.com>"

RUN apt-get update && apt-get install -y \
	alsa-utils \
	libasound2 \
	libasound2-plugins \
	pulseaudio \
	pulseaudio-utils \
    curl \
    gnupg \
	--no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*
        
# LV2 plugin installs
RUN apt-get update && apt-get install apt-utils -y && apt-get install pkg-config -y && apt-get install wget -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install ubuntustudio-audio-plugins -y && apt-get install libsndfile-dev -y \
    && wget https://sourceforge.net/projects/lsp-plugins/files/lsp-plugins/1.1.19/Linux-x86_64/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz -P /home/code-base/ \
    && tar -C /home/code-base/ -xvf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz \
    && cp -rf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64/usr/local/lib/lv2/lsp-plugins.lv2 /usr/lib/lv2/ \
    && rm -rf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz \
    && rm -rf /home/code-base/lsp-plugins-lv2-1.1.19-Linux-x86_64 \
    && apt-get install dh-autoreconf -y \
    && apt-get install meson -y

# Install lilv from source and other tools
RUN apt-get update && apt-get install lv2proc -y \
    && apt-get install lilv-utils -y \
    && apt-get install lv2-dev -y \
    && apt-get install liblilv-dev -y \
    && apt-get install audacity -y
    
# Install REAPER
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    && wget https://www.reaper.fm/files/6.x/reaper649_linux_x86_64.tar.xz \
    && tar -xf reaper649_linux_x86_64.tar.xz \
    && cd reaper_linux_x86_64 \
    && ./install-reaper.sh --install /opt --integrate-desktop --usr-local-bin-symlink --quiet
    
####### miniconda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    && wget https://github.com/falkTX/Carla/releases/download/v2.2.0/Carla_2.2.0-linux64.tar.xz \
    && tar -xf Carla_2.2.0-linux64.tar.xz
    
# See https://github.com/jessfraz/dockerfiles/issues/253#issuecomment-313995830
RUN apt-get update && apt-get install -y mesa-utils libgl1-mesa-glx


ENV HOME /home/pulseaudio
RUN useradd --create-home --home-dir $HOME pulseaudio \
	&& usermod -aG audio,pulse,pulse-access pulseaudio \
	&& chown -R pulseaudio:pulseaudio $HOME

WORKDIR $HOME
USER pulseaudio

COPY docker/default.pa /etc/pulse/default.pa
COPY docker/client.conf /etc/pulse/client.conf
COPY docker/daemon.conf /etc/pulse/daemon.conf

ENTRYPOINT [ "pulseaudio" ]
CMD [ "--log-level=4", "--log-target=stderr", "-v" ]
