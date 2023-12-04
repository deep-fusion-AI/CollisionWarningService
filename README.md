# Forward Collision Warning Service

The algorithm detects and tracks objects in video using SORT algorithm. For all objects, their projection to road 
plane is calculated (i.e. camera calibration is necessary). Location of objects on the road plane is filtered by 
Kalman Filter - which gives us the ability to predict future movement of objects. If the future path of an object 
strikes warning zone, alarm event is emitted. The event contains detailed description of the offensive behaviour, like 
location on screen and in the world, relative speed and direction of object, and time of entering the warning zone.

![Example](/data/example.gif)

## Requirements

There are few basic requirements for the algorithm itself:
* `git`
* `python3.8` or later
* `numpy`
* `pyyaml`
* `opencv-python` or  `py-opencv` if you use conda
* `pillow`
* `shapely`
* `filterpy`
* `pytorch`

Additional packages are required if you want to use the service as a Network Application within 5G-Era framework:
* `era-5g-interface`
* `era-5g-client`
* `simple-websocket`
* `websocket-client`
* `python-socketio`
* `flask`

System packages:
* `ffmpeg`
* `CUDA`

## Quick start

Run FCW service in docker:
```bash
docker run -p 5896:5896 --gpus all but5gera/fcw_service:0.8.0 
```
Donwload sample files:\
https://raw.githubusercontent.com/5G-ERA/CollisionWarningService/main/config/video3.mp4 \
https://raw.githubusercontent.com/5G-ERA/CollisionWarningService/main/config/video3.yaml \
https://raw.githubusercontent.com/5G-ERA/CollisionWarningService/main/config/config.yaml 

Install minimal client package:
```bash
python3 -m venv myvenv
myvenv\Scripts\activate
pip install fcw-client
```
Run client example:
```bash
fcw_client_python_simple -c config.yaml --camera video3.yaml video3.mp4
```

## Complete installation

Create python virtual environment, e.g.:
```bash
python3 -m venv myvenv
myvenv\Scripts\activate
```
and install fcw packages:
```bash
pip install fcw-core fcw-client fcw-service
```

For CUDA accelerated version, on Windows may be needed e.g.:
```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
It depends on the version of CUDA on the system [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

## Getting started - standalone example

Because of the sample files clone this repository somewhere first:

```bash
git clone https://github.com/5G-ERA/CollisionWarningService.git
cd CollisionWarningService
```

As an example, we use the video posted by u/Big-Replacement-7684 in 
[r/IdiotsInCars](https://www.reddit.com/r/IdiotsInCars/comments/10vfg5d/if_you_arent_going_to_yield_to_oncoming_traffic) 
showing typical dangerous situation that might result in car crash.

```bash
cd fcw-core/fcw_core
fcw_example --viz -t 150 -c ../../config/config.yaml --camera ../../videos/video3.yaml ../../videos/video3.mp4
```
Relevant configurations are in `videos/video3.yaml` - camera config, and `config/config.yaml` algorithm settings.

## Network Application for 5G-ERA

### Run FCW service / 5G-ERA Network Application

#### Run in Docker

The FCW service can be started in docker ([docker/fcw_service.Dockerfile](docker/fcw_service.Dockerfile)).
The image can be built:
```bash
cd ..
cd fcw-core/docker 
docker build -f fcw_service.Dockerfile -t but5gera/fcw_service:0.8.0 . 
```
or the image directly from the Docker Hub can be used.
 
The startup can be like this, where the GPU of the host computer is used and 
TCP ports 5896 and 8554 are mapped to the host network.
The port 8554 is RTSP port used for remote visualization.

```bash
docker run -p 5896:5896 -p 8554:8554 --network host --gpus all but5gera/fcw_service:0.8.0 
```

#### Local startup

The FCW service can also be started locally using [fcw-service/interface.py](fcw/service/interface.py), 
but the fcw-service package must be installed and the NETAPP_PORT environment 
variable should be set (default is 5896).
Run FCW service in same virtual environment as standalone example:

```bash
fcw_service
```

## Run client

In other terminal and in same virtual environment, set NETAPP_ADDRESS environment 
variable (default is http://localhost:5896) and run FCW python simple client example:

```bash
fcw_client_python_simple -c config/config.yaml --camera videos/video3.yaml videos/video3.mp4
```

or run simple client with rtsp stream (yaml files are not compatible with this rtsp stream, it is for example only):

```bash
fcw_client_python_simple -c config/config.yaml --camera videos/video3.yaml rtsp://root:upgm_c4m3r4@upgm-ipkam5.fit.vutbr.cz/axis-media/media.amp
```

or run advanced client:

```bash
fcw_client_python -c config/config.yaml --camera videos/video3.yaml videos/video3.mp4
```

## Running remote visualization

The visualisation should be enabled with config arguments during initialization command (`CollisionWorker` created 
with `viz` parameter set to `True` (ZeroMQ port can be also configured)).

If the FCW service has been started, **run RTSP server first**, e.g. https://github.com/bluenviron/mediamtx
on address: rtsp://localhost:8554 (TCP port 8554) and then:

```bash
docker run --rm -it -p 8554:8554 -e MTX_PROTOCOLS=tcp bluenviron/mediamtx:latest-ffmpeg
```

In other terminal run

```bash
cd fcw-service/fcw_service
fcw_visualization
```

or executing 
```bash
cd fcw-service/fcw_service
python3 visualization.py
```

You can view video, e.g. (localhost can be replaced with server address):

```bash
ffplay rtsp://localhost:8554/video
```

## Running with your videos

### Calibrate camera

### Setup algorithm parameters

### Run the example

## Notes

We use slightly modified version of SORT tracker from [abewley](https://github.com/abewley/sort) GitHub repository.

