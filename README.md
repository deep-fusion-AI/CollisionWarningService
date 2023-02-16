# CollisionAvoidanceService

The algortithm detects and tracks objects in video using SORT algorithm. For all objects, their projection to road plane is calculated (i.e. camera calibration is necessary). Location of objects on the road plane is filtered by Kalman Filter - whichgibes us the ability to predict future movement of objects. If the future path of an object strikes warning zone, alarm event is emited. The event contains detailed description of the ofensive behaviour, like location on screen and in the world, relative speed and direction of object, and time of entering the warning zone.

## Inputs

* Video stream with known and stable FPS
* Intrinsic and extrinsic camera calibration
* Configuration parameters of the FCW algorithm

Example configurations can be found in `config` directory.

## The algorithm description

TODO

## Network Application for 5G-Era

TODO

## Notes

We use slightly modified version of SORT tracker from [abewley](https://github.com/abewley/sort) gitub repository.

## Next steps

* [ ] Horizon in config must be given in distorted coordinates (not undistorted)
* [ ] Translation vector must be given as full `xyz` not just `z`
* [ ] Loading algorithm configuration from `yaml` file
* [ ] Detector initialization from config
* [ ] Pass list of classes to detector (now its fixed)
* [ ] Initi SORT tracker from config
* [ ] Refactor to make a python package from core algorithm
* [ ] Update README to match configuration formats, and add description of the algorithm
* [ ] Wrap to 5G-Era interface
* [ ] Test and deploy