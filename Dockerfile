FROM ubuntu:20.04
LABEL maintainer "Nick Bryan <nibryan@adobe.com>"

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
    && wget https://sourceforge.net/projects/lsp-plugins/files/lsp-plugins/1.1.19/Linux-x86_64/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz -P /home/temp/ \
    && tar -C /home/temp/ -xvf /home/temp/lsp-plugins-lv2-1.1.19-Linux-x86_64.tar.gz \
    && cp -rf /home/temp/lsp-plugins-lv2-1.1.19-Linux-x86_64/usr/local/lib/lv2/lsp-plugins.lv2 /usr/lib/lv2/ \
    && rm -rf /home/temp \
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
    && ./install-reaper.sh --install /opt --integrate-desktop --usr-local-bin-symlink --quiet \
    && rm -rf reaper_linux_x86_64 \
    && rm -rf reaper649_linux_x86_64.tar.xz
    
####### miniconda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc

# Install Carla
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
#     && wget https://github.com/falkTX/Carla/releases/download/v2.2.0/Carla_2.2.0-linux64.tar.xz \
#     && tar -xf Carla_2.2.0-linux64.tar.xz
    
# Attempt to install OpenGL stuff
# See https://github.com/jessfraz/dockerfiles/issues/253#issuecomment-313995830
RUN apt-get update && apt-get install -y mesa-utils libgl1-mesa-glx


# Supervisor setup
RUN apt-get update && apt-get install -y supervisor openssh-client
RUN mkdir -p /var/log/supervisord
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# Upgrade pip
RUN /bin/bash -c "python -m pip install --upgrade pip"

# Install the JupyterLab IDE
RUN /bin/bash -c "pip install jupyterlab"

# ####### INSTALL CODE SERVER 
# # via https://github.com/cdr/code-server/issues/2341#issuecomment-740892890
# RUN /bin/bash -c "curl -fL https://github.com/cdr/code-server/releases/download/v3.8.0/code-server-3.8.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xz"
# RUN /bin/bash -c "mv /usr/local/bin/code-server-3.8.0-linux-amd64 /usr/local/bin/code-server-3.8.0"
# RUN /bin/bash -c "ln -s /usr/local/bin/code-server-3.8.0/bin/code-server /usr/local/bin/code-server"
# # Install Python extension 
# RUN /bin/bash -c "wget https://github.com/microsoft/vscode-python/releases/download/2020.10.332292344/ms-python-release.vsix \
#  		&& code-server --install-extension ./ms-python-release.vsix || true"
# # Install C++ extension
# RUN /bin/bash -c "wget https://github.com/microsoft/vscode-cpptools/releases/download/1.1.3/cpptools-linux.vsix  \
# 		&& code-server --install-extension ./cpptools-linux.vsix || true"
# # Set VS Code password to None
# #RUN /bin/bash -c "sed -i.bak 's/auth: password/auth: none/' ~/.config/code-server/config.yaml"
# COPY docker/code-server-config.yaml /root/.config/code-server/config.yaml
# # Fix broken python plugin # https://github.com/cdr/code-server/issues/2341
# RUN /bin/bash -c "mkdir -p ~/.local/share/code-server/ && mkdir -p ~/.local/share/code-server/User"
# COPY docker/settings.json /root/.local/share/code-server/User/settings.json 
# ####### DONE INSTALL CODE SERVER 

ENV HOME /home
RUN usermod -aG audio,pulse,pulse-access root \
	&& chown -R root:root $HOME
WORKDIR $HOME

COPY docker/default.pa /etc/pulse/default.pa
COPY docker/client.conf /etc/pulse/client.conf
COPY docker/daemon.conf /etc/pulse/daemon.conf

EXPOSE 8080 8888 8887 443

ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
