# CollisionWarningService

The algorithm detects and tracks objects in video using SORT algorithm. For all objects, their projection to road 
plane is calculated (i.e. camera calibration is necessary). Location of objects on the road plane is filtered by 
Kalman Filter - which gives us the ability to predict future movement of objects. If the future path of an object 
strikes warning zone, alarm event is emitted. The event contains detailed description of the offensive behaviour, like 
location on screen and in the world, relative speed and direction of object, and time of entering the warning zone.

![Example](/data/example.gif)

## Requirements

There are few basic requirements for the algorithm itself
* `numpy`
* `pyyaml`
* `opencv-python` or  `py-opencv` if you use conda
* `pillow`
* `shapely`
* `filterpy`
* `pytorch`

Additional packages are required if you want to use the service as a Network Application within 5G-Era framework/
* `era_5g_interface`
* `era_5g_client`
* `simple-websocket`
* `python-socketio`
* `flask`


## Installation

## Getting started - standalone example

As an example, we use the video posted by u/Big-Replacement-7684 in 
[r/IdiotsInCars](https://www.reddit.com/r/IdiotsInCars/comments/10vfg5d/if_you_arent_going_to_yield_to_oncoming_traffic) 
showing typical dangerous situation that might result in car crash.


```bash
# This will load configurations for video3.mp4 and show visualization.
> python fcw_example.py
```

Relevant configurations are in `videos/video3.yaml` - camera config, and `config/config.yaml` algorithm settings.

## Running with your videos

### Calibrate camera

### Setup algorithm parameters

### Run the example

## Network Application for 5G-ERA

### Run FCW service/NetApp

#### Docker

The FCW service can be started in docker ([docker/fcw_service.Dockerfile](docker/fcw_service.Dockerfile)), 
for example in this way, where the GPU of the host computer is used and 
TCP port 5897 is mapped to the host network.
```bash
docker build -f fcw_service.Dockerfile -t but5gera/fcw_service:0.1.0 . \
  && docker run -p 5897:5897 --network host --gpus all but5gera/fcw_service:0.1.0 
```

#### Local startup

The FCW Service can also be run locally using [fcw/service/interface.py](fcw/service/interface.py), 
but all necessary dependencies must be installed and the NETAPP_PORT environment 
variable should be set (default is 5896):
```bash
set NETAPP_PORT=5897
```
or on Linux:
```bash
export NETAPP_PORT=5897
```

Requirements:
- `git`
- `python3.8` or later
- `ffmpeg`
- `CUDA`
- `poetry`

At now, FCW Service package collision-warning-service contains both server and client (examples) parts. This package depends on
- `era-5g-interface>=0.4.1`
- `era-5g-client>=0.4.1`

Clone this repository somewhere first:

```bash
git clone https://github.com/5G-ERA/CollisionWarningService.git
cd CollisionWarningService
```

Installation of collision-warning-service:

```bash
cd ..
poetry install
```

Run FCW service:

```bash
poetry run fcw_service
```

For CUDA accelerated version, on Windows may be needed e.g.:
```bash
pip3 install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
It depends on the version of CUDA on the system [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

## Run client

Set NETAPP_PORT environment variable (default is http://localhost:5896):
```bash
set NETAPP_ADDRESS=http://localhost:5897
```
or on Linux:
```bash
export NETAPP_ADDRESS=http://localhost:5897
```

Run FCW python simple client example:

```bash
poetry run fcw_client_python_simple -c config/config.yaml --camera videos/video3.yaml videos/video3.mp4
```

or run simple client with rtsp stream (yaml files are not compatible with tshi rtsp stream, it is for example only):

```bash
poetry run fcw_client_python_simple -c config/config.yaml --camera videos/video3.yaml rtsp://root:upgm_c4m3r4@upgm-ipkam5.fit.vutbr.cz/axis-media/media.amp
```

or run advanced client:

```bash
poetry run fcw_client_python -c config/config.yaml --camera videos/video3.yaml videos/video3.mp4
```

## Notes

We use slightly modified version of SORT tracker from [abewley](https://github.com/abewley/sort) gitub repository.

