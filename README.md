## msynth

We are using a python package. Therefore, please run : pip install -e msynth , in the directory that is the parent of the msynth folder. 

To run a sweep, run: 

wandb sweep sweep.yaml 

This will return a sweep id. 

wandb agent [your wandb username] / [the project] / [the id given to you by the first command above]

For me thats: wandb agent chiquita/msynth-tests/[id]

## Developer


### Install pip package 

```
# Create a clean conda env
conda create -n msynth python=3.7 -y
conda activate msynth

cd <msynth git repo>

# Install package
pip install -e .

```



## Linux Development on MacOS

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

* Start Xquartz

```
open -a Xquartz
```

* Configure Xquartz to "Allow connections from network clients"

<!-- * Attempt to allow OpenGL X11 forwarding (doesn't work for me yet)

```
defaults write `quartz-wm --help | awk '/default:.*X11/ { gsub(/\)/, "", $2); print $2}'` enable_iglx -bool true
``` -->

* Whitelist localhost for X11 network connections via paste below on command line

```
/opt/X11/bin/xhost + "127.0.0.1"
```

* Build the docker image into a container
docker-compose up --build -d

* Specify a shared data directory between your local machine and docker to share data
```
export MSYNTH=<path/for/shared/data>
```

* Run an existing image
```
docker-compose up 
```

* Open your IDE of choice
	* For Jupyter, open a web browser at http://127.0.0.1:8888 (password is dsp). 
	* For VS Code, install the Docker extension, then attach to the msyth container and use VS Code.
	* For command line SSH access, open a second terminal, find the running container id, and enter it
		```
		docker container ls
		docker exec -it <CONTAINER ID> bash
		```
* Run Linux applications with audio and graphics forwarding.

Enter your container via the SSH access instructions above and launch. Try running reaper within your Linux container on mac via

```
audacity
reaper
```
