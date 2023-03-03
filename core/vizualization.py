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

        X = np.array(o.future_path().coords)
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


# def draw_tracked_objects(d: ImageDraw.ImageDraw, tracked_objects: dict):
#     for tid, t in tracked_objects.items():
#         x1, y1, x2, y2 = t.get_state()[0]
#         color = (0, 255, 0, 64)
#         d.rectangle((x1, y1, x2, y2), fill=color, outline=None, width=0.5)
#         # label = f"track {tid}"
#         # _, _, tw, th = font.getbbox(label, stroke_width=1)
#         # tw, th = tw + 4, th + 4
#         # d.rectangle((x1, y1 - th, x1 + tw, y1), fill=(0, 0, 0))
#         # d.text((x1 + 3, y1 - th + 2), label, fill=(255, 255, 255), font=font, stroke_width=0)



#   # Visualization
#         base = Image.fromarray(img_undistorted[..., ::-1], "RGB").convert("RGBA")
#         objects_image = Image.new("RGBA", base.size)
#         osd_image = Image.new("RGBA", base.size)

#         osd_draw = ImageDraw.Draw(osd_image)
#         draw_horizon(osd_draw, camera, fill=(255, 255, 0, 128), width=1)

#         objects_draw = ImageDraw.Draw(objects_image)

#        
#         draw_tracked_objects(objects_draw, tracked_objects)

#         for tid, o in dangerous_objects.items():
#             x1, y1, x2, y2 = tracked_objects[tid].get_state()[0]
#             # objects_draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
#             objects_draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 64))
#             dist = Point(o.location).distance(guard.vehicle_zone)
#             info = f"{dist:.1f} m"
#             objects_draw.text(
#                 (0.5 * (x1 + x2), 0.5 * (y1 + y2)), info, align="center", font=font,
#                 stroke_fill=(255, 255, 255), stroke_width=1, fill=(0, 0, 0)
#                 )
            
#         for tid, o in guard.objects.items():
#             X = np.atleast_2d([o.kf.x[0,0],o.kf.x[3,0],0])
#             scr_loc, _ = camera.project_points(X)
#             if scr_loc.size > 0:
#                 x,y = scr_loc[0]
#                 objects_draw.line([(x-10,y),(x+10,y)], fill=(255,255,0,128), width=3)
#                 objects_draw.line([(x,y-10),(x,y+10)], fill=(255,255,0,128), width=3)

#             X = np.array(o.future_path().coords)
#             n = X.shape[0]
#             X = np.hstack([X, np.zeros((n,1))])
#             scr_loc, _ = camera.project_points(X,near=5)
#             scr_loc = list(map(tuple, scr_loc))
#             objects_draw.line(scr_loc, fill=(0,255,0,255), width=1)


            

#         display = Image.alpha_composite(objects_image, osd_image)
#         out = Image.alpha_composite(base, display).convert("RGB")

#         cv_image = np.array(out)[..., ::-1]
#         cv2.imshow("FCW", cv_image)
#         cv2.waitKey(1)