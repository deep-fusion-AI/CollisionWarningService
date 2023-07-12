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
* TODO


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

The FCW service can be started in docker, e.g.The FCL service can be run in docker 
([docker/fcw_service.Dockerfile](docker/fcw_service.Dockerfile)), for example in this way, 
where the GPU of the host computer is used and TCP port 5897 is mapped to the host network.
```bash
docker build -f fcw_service.Dockerfile -t but5gera/fcw_service . \
  && docker run -p 5897:5897 --network host --gpus all but5gera/fcw_service 
```

#### Local startup

The FCW Service can also be run locally using [fcw/service/interface.py](fcw/service/interface.py), 
but all necessary dependencies must be installed in the used python environment
and the NETAPP_PORT environment variable should be set (default is 5896).

Requirements:
- `git`
- `python3.8` or later
- `ffmpeg`
- `CUDA`
- `poetry`

At now, FCW Service package collision-warning-service contains both client and server parts. This package depends on
- `era_5g_object_detection_common`
- `era_5g_object_detection_standalone`
- `era-5g-interface`
- `era-5g-client`

For proper functioning, it is not yet possible (will be after the pip releases of the updated compatible packages) to install all current packages via pip, and they can be installed 
e.g. like this (The order of installation is important, otherwise incompatible versions from pip may be installed):
```bash
git clone https://github.com/klepo/Reference-NetApp.git
cd Reference-NetApp/src/python/era_5g_object_detection_common
pip3 install -r requirements.txt
pip3 install .
```
```bash
cd ..
cd era_5g_object_detection_standalone
pip3 install -r requirements.txt
pip3 install . 
```
```bash
cd ../../../..
git clone https://github.com/klepo/era-5g-interface.git
cd era-5g-interface
pip3 install -r requirements.txt
pip3 install -e .
```
Editable install mode `-e` is mode is needed due to `BUILD` file name collision and `build` folder creation on Windows.

Installation of collision-warning-service:

```bash
cd ..
poetry install
```

Run FCW service

```bash
poetry run fcw_service
```

## Run client

TODO

## Notes

We use slightly modified version of SORT tracker from [abewley](https://github.com/abewley/sort) gitub repository.

