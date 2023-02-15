# CollisionAvoidanceService


# Forward collision warning

The algortithm detects and tracks objects in video using SORT algorithm. For all objects, their projection to road plane is calculated (i.e. camera calibration is necessary). Location of objects on the road plane is filtered by Kalman Filter - whichgibes us the ability to predict future movement of objects. If the future path of an object strikes warning zone, alarm event is emited. The event contains detailed description of the ofensive behaviour, like location on screen and in the world, relative speed and direction of object, and time of entering the warning zone.

## Inputs

* Video stream with knwon and stable FPS
* Intrinsic and extrinsic camera calibration
* Configuration parameters of the FCW algorithm


This is an example of camera calibration config file

```yaml
# Width and height of the image
image_size: [2064, 1544]

# 3x3 matrix K measured with OpenCV calibration procedure
K: [
  [1240,  0.0, 988],
  [ 0.0, 1237, 762],
  [ 0.0,  0.0, 1.0]
]

# Fisheye distortion coeficients
D: [-0.069,0,0,0]

# Extrinsic parameters estimated by user
# Horizon line is given as (x1, y1, x2, y2) - two points on horizon line
# with the first point (x1,y1) pointing towards vanishing point in the
# direction of vehicle movement.
horizon: [2018, 166, 0, 166]

# View direction can be '+x' (forward facing camera) or '-x' (backward facing camera)
view_direction: "+x"

# Height of camera above the ground in meters
height: 1.7
```

The parameters of algorithm are specified in `YAML` file.

```yaml
# Object path prediction to the future
prediction_time: 1  # [s]
prediction_interval: 0.1  # [s]

warning_zone_length: 15  # [m]
warning_zone_width: 3 # [m]
```



