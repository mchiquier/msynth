## msynth

We are using a python package. Therefore, please run : pip install -e msynth , in the directory that is the parent of the msynth folder. 



## Developer


### Install pip package 

```
# Create a clean conda env
conda create -n msynth python=3.7
conda activate msynth

cd <msynth git repo>

# Install package
pip install -e .

```


## Run Linux Audio on MacOS

* Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
* Install X11 for macOS via [X Quartz](https://github.com/XQuartz/XQuartz/releases/download/XQuartz-2.8.1/XQuartz-2.8.1.dmg)
* Install PulseAudio via homebrew, start the PulseAudio deamon, and verify status:

```
# Install
brew install pulseaudio

# Start daemon
pulseaudio --load=module-native-protocol-tcp --exit-idle-time=-1 --daemon

# Check pulse audio status
pulseaudio --check -v
```

* Configure PulseAudio output via:

```
# List default outputs for pulse audio
pacmd list-sinks

# Set the default output
pacmd set-default-sink <index of sink>
```

* Try to play an audio file via:

```
paplay -p <testfile>.wav
```

* Build the docker audio image with tag name lv2

```
cd <git_repo>
docker build --file Dockerfile.audio -t lv2 .
```

* Start Xquartz

```
open -a Xquartz
```

* Configure Xquartz to "Allow connections from network clients"
* Attempt to allow OpenGL X11 forwarding (doesn't work for me yet)

```
defaults write `quartz-wm --help | awk '/default:.*X11/ { gsub(/\)/, "", $2); print $2}'` enable_iglx -bool true
```
* Whitelist localhost for X11 network connections via paste below on command line

```
/opt/X11/bin/xhost + "127.0.0.1"
```

* Run (and ssh into) your image  

```
# Basic
docker run -it -e PULSE_SERVER=docker.for.mac.localhost -e DISPLAY=host.docker.internal:0 -v ~/.config/pulse:/home/pulseaudio/.config/pulse -v ~/Desktop:/opt/Desktop --entrypoint /bin/bash --rm -u 0 lv2 
```

* Run and ssh into your image and do anything you want

```
docker run -it -e PULSE_SERVER=docker.for.mac.localhost \
	-e DISPLAY=host.docker.internal:0  \
	-v ~/.config/pulse:/home/pulseaudio/.config/pulse \
	-v ~/:/opt/mac --entrypoint /bin/bash -u 0 lv2 
```

* Try running reaper within your Linux container on mac via

```
reaper
```

* Create a container called "audacity" that runs audacity directly  

```
docker run -it -e PULSE_SERVER=docker.for.mac.localhost \
	-e DISPLAY=host.docker.internal:0  \
	-v ~/.config/pulse:/home/pulseaudio/.config/pulse \
	-v ~/:/opt/mac --entrypoint audacity -u 0 --name audacity lv2 
```

* Start your container again

```
docker start audacity
```
 
<!--* Find, restart, and ssh into your container

```
# Find your container id via:
docker ps -a

# Restart
docker restart <container_id>

# SSH
docker exec -it <container_id> /bin/bash
```
-->
