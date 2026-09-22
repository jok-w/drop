"""Local contrast measurements for compact targets distinct from their background.

The foreground polarity and intensity threshold are learned from the manual ROI.
This is not a semantic detector; ambiguous matches are deliberately withheld.
"""
import math

import cv2
import numpy as np


class ContrastLocator:
    def __init__(self, frame, box):
        self.enabled = False
        x, y, w, h = box
        self.initial_size = max(w, h)
        margin = max(4, round(max(w, h) * .3))
        x0, y0 = max(0, x-margin), max(0, y-margin)
        x1, y1 = min(frame.shape[1], x+w+margin), min(frame.shape[0], y+h+margin)
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        roi = gray[y-y0:y-y0+h, x-x0:x-x0+w]
        border = np.ones(gray.shape, dtype=bool)
        border[y-y0:y-y0+h, x-x0:x-x0+w] = False
        if not border.any():
            return
        background = float(np.median(gray[border]))
        low, high = np.percentile(roi, [20,80])
        self.dark = background-low >= high-background
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.threshold = threshold
        self.kernel = np.ones((3,3), dtype=np.uint8)
        labels, stats = self.components(gray)
        choices = []
        for label, stat in enumerate(stats[1:], 1):
            bx, by, bw, bh, area = map(int, stat)
            if area < max(12, w*h*.08) or area > w*h*1.5:
                continue
            ix0, iy0 = max(bx,x-x0), max(by,y-y0)
            ix1, iy1 = min(bx+bw,x-x0+w), min(by+bh,y-y0+h)
            overlap = max(0,ix1-ix0)*max(0,iy1-iy0)
            if overlap / max(1,bw*bh) < .75 or not .25 < bw/bh < 4:
                continue
            choices.append((overlap, label, stat))
        if not choices:
            return
        _, label, stat = max(choices, key=lambda item:item[0])
        values = gray[labels==label]
        self.foreground = float(np.median(values))
        self.contrast = abs(background-self.foreground)
        if self.contrast < 25:
            return
        self.initial_area = self.last_area = float(stat[4])
        self.enabled = True

    def components(self, gray):
        kind = cv2.THRESH_BINARY_INV if self.dark else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, self.threshold, 255, kind)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        return labels, stats

    def locate(self, frame, prediction, last_box, position_covariance, recovering=False):
        if not self.enabled:
            return None, {"appearance_reason":"unavailable"}
        x,y,w,h = last_box
        last_center = np.array([x+w/2,y+h/2])
        predicted = np.asarray(prediction)
        sigma = math.sqrt(max(0,float(np.linalg.eigvalsh(position_covariance).max())))
        radius = min(self.initial_size*5, max(max(w,h)*2.5, sigma*3))
        lo = np.minimum(last_center,predicted)-radius
        hi = np.maximum(last_center,predicted)+radius
        x0,y0 = max(0,int(lo[0])),max(0,int(lo[1]))
        x1,y1 = min(frame.shape[1],int(hi[0])+1),min(frame.shape[0],int(hi[1])+1)
        if x1<=x0 or y1<=y0:
            return None, {"appearance_reason":"outside_frame"}
        gray = cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        labels,stats = self.components(gray)
        candidates=[]
        for label, stat in enumerate(stats[1:],1):
            bx,by,bw,bh,area=map(int,stat)
            if not max(12,self.last_area*.35,self.initial_area*.06)<=area<=min(self.last_area*2.8,self.initial_area*2):
                continue
            if not .25<=bw/bh<=4 or area/(bw*bh)<.3:
                continue
            if bx==0 or by==0 or bx+bw==gray.shape[1] or by+bh==gray.shape[0]:
                continue
            values=gray[by:by+bh,bx:bx+bw][labels[by:by+bh,bx:bx+bw]==label]
            foreground=float(np.median(values))
            color_error=abs(foreground-self.foreground)
            if color_error>max(20,self.contrast*.4):
                continue
            margin=max(3,round(max(bw,bh)*.2))
            rx0,ry0=max(0,bx-margin),max(0,by-margin)
            rx1,ry1=min(gray.shape[1],bx+bw+margin),min(gray.shape[0],by+bh+margin)
            surround=gray[ry0:ry1,rx0:rx1]
            outside=np.ones(surround.shape,dtype=bool)
            outside[by-ry0:by-ry0+bh,bx-rx0:bx-rx0+bw]=False
            background=float(np.median(surround[outside]))
            contrast=(background-foreground) if self.dark else (foreground-background)
            # Reacquisition needs stronger appearance evidence than continuation;
            # otherwise an expanding search can gradually latch onto scenery.
            minimum_contrast = self.contrast * (.9 if recovering else .75)
            if contrast<max(25,minimum_contrast):
                continue
            center=np.array([x0+bx+bw/2,y0+by+bh/2])
            distance=float(np.linalg.norm(center-predicted))/max(w,h,1)
            if np.linalg.norm(center-last_center)>self.initial_size*3:
                continue
            # A visual quality score, not a calibrated probability.
            cost=distance*.45+abs(math.log(area/self.last_area))*.35+color_error/max(self.contrast,1)
            padding=max(2,round(max(bw,bh)*.12))
            left,top=max(0,x0+bx-padding),max(0,y0+by-padding)
            right,bottom=min(frame.shape[1],x0+bx+bw+padding),min(frame.shape[0],y0+by+bh+padding)
            candidates.append((cost,(left,top,right-left,bottom-top),area,contrast))
        candidates.sort(key=lambda c:c[0])
        if not candidates:
            return None,{"appearance_reason":"no_match","appearance_candidates":0}
        best=candidates[0]
        if len(candidates)>1 and candidates[1][0]-best[0]<.35:
            return None,{"appearance_reason":"ambiguous","appearance_candidates":len(candidates)}
        if best[0]>2.5:
            return None,{"appearance_reason":"weak_match","appearance_candidates":len(candidates)}
        return best[1],dict(appearance_reason="matched",appearance_candidates=len(candidates),
                           appearance_cost=best[0],foreground_area=best[2],local_contrast=best[3])

    def accept(self, details):
        self.last_area=float(details["foreground_area"])
