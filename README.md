# CollisionAvoidanceService

The algortithm detects and tracks objects in video using SORT algorithm. For all objects, their projection to road plane is calculated (i.e. camera calibration is necessary). Location of objects on the road plane is filtered by Kalman Filter - whichgibes us the ability to predict future movement of objects. If the future path of an object strikes warning zone, alarm event is emited. The event contains detailed description of the ofensive behaviour, like location on screen and in the world, relative speed and direction of object, and time of entering the warning zone.

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

As an example, we use the video posted by u/Big-Replacement-7684 in [r/IdiotsInCars](https://www.reddit.com/r/IdiotsInCars/comments/10vfg5d/if_you_arent_going_to_yield_to_oncoming_traffic) showing typical dangerous situation that might result in car crash.


```bash
# This will load configurations for video3.mp4 and show vizualiztion.
> python fcw_example.py
```

Relevant configurations are in `videos/video3.yaml` - camera config, and `config/config.yaml` algorithm settings.

## Running with your videos

### Calibrate camera

### Setup algorithm parameters

### Run the example

## Network Application for 5G-Era

TODO

## Notes

We use slightly modified version of SORT tracker from [abewley](https://github.com/abewley/sort) gitub repository.

