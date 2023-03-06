"""
"""

from math import ceil
from typing import Iterable, List

import numpy as np
import sort
from collision import PointWorldObject, ForwardCollisionGuard, ObjectStatus
from geometry import Camera
from more_itertools import windowed
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split

# Font for OSD
_font = ImageFont.truetype("../data/UbuntuMono-R.ttf", 14, encoding="unic")



# def segmentize(p: LineString, max_dist=10):
#     pts = []
#     for a, b in windowed(p.coords, n=2):
#         seg = LineString([a, b])
#         f = np.linspace(0, seg.length, ceil(seg.length / max_dist), endpoint=False)
#         _pts = [seg.interpolate(x) for x in f]
#         pts.extend(_pts)
#     return LineString(pts)


def draw_horizon(size: tuple, cam: Camera, **kwargs):
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)
    x = list(cam.horizon.coords)
    draw.line(x, **kwargs)
    return image

def compose_layers(base: Image.Image, *layers: Iterable[tuple]):
    for l, dest in layers:
        base.alpha_composite(l, dest or (0,0))


def draw_image_trackers(
        size: tuple,
        trackers: List[sort.KalmanBoxTracker],
    ):
    """

    """
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)

    for t in trackers:
        x1, y1, x2, y2 = t.get_state()[0]
        color = (0, 255, 0, 64)
        outline = (0, 255, 0, 128)
        if t.age < 3 or t.hit_streak == 0:  # TODO: call it.is_reliable()
            color = (255,255,0,32)
            outline = None        
        draw.rectangle((x1, y1, x2, y2), fill=color, outline=outline)

    return image


def draw_world_objects(
        size: tuple,
        camera: Camera,
        objects: Iterable[PointWorldObject]
    ):
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)

    # for o in objects:
    #     x1, y1, x2, y2 = tracked_objects[tid].get_state()[0]
    #     # objects_draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    #     objects_draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 64))
    #     dist = Point(o.location).distance(guard.vehicle_zone)
    #     info = f"{dist:.1f} m"
    #     objects_draw.text(
    #         (0.5 * (x1 + x2), 0.5 * (y1 + y2)), info, align="center", font=font,
    #         stroke_fill=(255, 255, 255), stroke_width=1, fill=(0, 0, 0)
    #         )
        
    for o in objects:
        
        X = np.atleast_2d([o.kf.x[0,0],o.kf.x[3,0],0])
        scr_loc, _ = camera.project_points(X)
        if scr_loc.size > 0:
            x,y = scr_loc[0]
            draw.line([(x-10,y),(x+10,y)], fill=(255,255,0,128), width=3)
            draw.line([(x,y-10),(x,y+10)], fill=(255,255,0,128), width=3)

        X = np.array(o.future_path(5).coords)
        n = X.shape[0]
        X = np.hstack([X, np.zeros((n,1))])
        scr_loc, _ = camera.project_points(X,near=5)
        scr_loc = list(map(tuple, scr_loc))
        draw.line(scr_loc, fill=(0,255,0,255), width=1)

    return image


def draw_danger_zone(size: tuple, camera: Camera, zone: Polygon):
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)

    front = Polygon([
        [ 1,-10],
        [ 1, 10],
        [50, 10],
        [50,-101],
    ])

    X = np.array(zone.intersection(front).boundary.coords)
    n = X.shape[0]
    X = np.hstack([X, np.zeros((n,1))])
    scr_loc, _ = camera.project_points(X,near=-100)
    scr_loc = list(map(tuple, scr_loc))
    draw.polygon(scr_loc, fill=(255,255,0,32), outline=(255,255,0,128))

    return image




def draw_world_coordinate_system(size: tuple, camera: Camera):
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)

    def draw_line_string(ls: LineString, accuracy:float=0.01, **kwargs):
        n = int(np.ceil(1/accuracy))
        coords = [
            ls.interpolate(t, normalized=True).xy for t in np.linspace(0,1,n)
        ]
        X = np.hstack(coords)
        X = np.vstack([X, np.zeros((1,n)) ])
        x,_ = camera.project_points(X.T, near=1)
        draw.line(list(map(tuple, x)), **kwargs)


    for x in np.linspace(-20, 20, 41):
        draw_line_string(LineString( [(x, -20),(x, 20)] ), fill=(255,0,0) )

    for y in np.linspace(-20, 20, 41):
        draw_line_string(LineString( [(-20, y),(20, y)] ), fill=(0,255,0) )

    return image


def tracking_info(
        size: tuple,
        object_status: List[ObjectStatus]
    ):
    image = Image.new("RGBA", size, color=(0,0,0,255))
    draw = ImageDraw.Draw(image)

    info_text = f"Tracking {len(object_status)}"
    draw.text((8, 0), info_text, fill=(255, 255, 255), font=_font)

    caution_status = any(
        s.crosses_danger_zone for s in object_status
    )

    warning_status = any(
        s.is_in_danger_zone for s in object_status
    )

    danger_status = any(
        s.time_to_collision > 0 for s in object_status if s.time_to_collision is not None
    )

    caution_color = (255,255,0) if caution_status else (64,64,64)
    draw.text((160, 0), "CAUTION", fill=caution_color, font=_font, align="left")
    
    warning_color = (255,255,0) if warning_status else (64,64,64)
    draw.text((260, 0), "WARNING", fill=warning_color, font=_font, align="left")

    danger_color = (255,0,0) if danger_status else (64,64,64)
    draw.text((360, 0), "DANGER", fill=danger_color, font=_font, align="left")

    if danger_status:
        ttc = min(
            s.time_to_collision for s in object_status if s.time_to_collision is not None
        )
        draw.text((460, 0), f"ttc = {ttc:0.1f} s", fill=(255, 0, 0), font=_font, align="left")

    return image


def cog_logo(size: tuple = (256, 256)):
    """
    Cognitechna logo image
    """
    logo = Image.open("../data/cog_logo.png").convert("RGBA")  # FIXME location data in the package not relative to `pwd`
    w,h = logo.size
    cx, cy = w / 2, h / 2
    sz = 155
    box = cx-sz, cy-sz, cx+sz, cy+sz
    logo = logo.resize(size, box=box, reducing_gap=True, resample=Image.LANCZOS)

    bg = Image.new("RGBA", size, (255,255,255))
    bg.alpha_composite(logo)

    drw = ImageDraw.ImageDraw(bg)
    drw.rectangle((0, 0, size[0]-1, size[1]-1), fill=None, outline=(0,0,0,255), width=1)

    return bg


def vehicle_marker_image(scale: int = 1):
    marker_image = Image.open("../data/marker.png")
    w,h = marker_image.size
    return marker_image.resize((w*scale, h*scale), Image.NEAREST), (7*scale, 0)
 

def mark_vehicles(size: tuple, objects:Iterable[PointWorldObject], camera: Camera, marker: Image, anchor:tuple = (0,0)):
    image = Image.new("RGBA", size, color=(0,0,0,0))
    ax,ay = anchor
    # loc (N,2) xy
    for o in objects:
        X = np.atleast_2d([o.kf.x[0,0],o.kf.x[3,0],0])
        scr_loc, _ = camera.project_points(X, near=1, to_rectified=False)
        if scr_loc.shape[0] > 0:
            x,y = scr_loc[0]
            image.paste(marker, (int(x-ax), int(y-ay)))
    return image