import cv2
import numpy as np
import argparse
from os import path
import pathlib
import signal
import sys

class WatermarkAdder:
    def __init__(self):
        self.video_path = None
        self.watermark_path = None
        self.output_path = 'output.avi'
        self.watermark_cv_obj = None
        self.watermark_pos_horizontal = None
        self.watermark_pos_vertical = None
        self.watermark_pos_center = False
        self.watermark_offset_horizontal = None
        self.watermark_offset_vertical = None
        self.show_processing = False
        self.watermark_transparency = None
        
    def set_video_path(self, path):
        self.video_path = path

    def set_watermark_path(self, path):
        self.watermark_path = path
        self.watermark_cv_obj = cv2.imread(self.watermark_path)
        self.watermark_cv_obj = self.resize_image(self.watermark_cv_obj, height=50)
        self.watermark_cv_obj = cv2.cvtColor(self.watermark_cv_obj, cv2.COLOR_BGR2BGRA)

    def set_output_path(self, path):
        self.output_path = path
    
    def get_output_path(self):
        return str(pathlib.Path().absolute()) + '/' + self.output_path

    def set_watermark_transparency(self, value):
        self.watermark_transparency = value

    def set_show_processing(self, value):
        self.show_processing = value

    def preform_processing(self):
        if self.video_path is None or self.watermark_path is None:
            if self.video_path:
                print('Configure video path.')
            else:
                print('Configure watermark path.')
            return 1

        if self.watermark_pos_horizontal == None and self.watermark_pos_vertical == None \
                                                    and self.watermark_pos_center == False:
            self.set_watermark_position(horizontal='right', vertical='bottom', center=False)

        self.overlay_watermark()

    def set_watermark_position_from_arg(self, quadrant):
        if quadrant == 0:
            self.set_watermark_position(vertical='top', horizontal='left', center=False)
        elif quadrant == 1:
            self.set_watermark_position(vertical='top', horizontal=None, center=True)
        elif quadrant == 2:
            self.set_watermark_position(vertical='top', horizontal='right', center=False)
        elif quadrant == 3:
            self.set_watermark_position(vertical=None, horizontal='left', center=True)
        elif quadrant == 4:
            self.set_watermark_position(vertical=None, horizontal=None, center=True)
        elif quadrant == 5:
            self.set_watermark_position(vertical=None, horizontal='right', center=True)
        elif quadrant == 6:
            self.set_watermark_position(vertical='bottom', horizontal='left', center=False)
        elif quadrant == 7:
            self.set_watermark_position(vertical='bottom', horizontal=None, center=True)
        elif quadrant == 8:
            self.set_watermark_position(vertical='bottom', horizontal='right', center=False)

    def set_watermark_position(self, vertical=None, horizontal=None, center=None):
        if vertical is not None:
            self.watermark_pos_vertical = vertical
        if horizontal is not None:
            self.watermark_pos_horizontal = horizontal
        if center != None and center != self.watermark_pos_center:
            self.watermark_pos_center = center
        self.watermark_offset_horizontal = None
        self.watermark_offset_vertical = None
        self.watermark_offset_horizontal, self.watermark_offset_vertical = \
                                        self.set_watermark_offsets(self.watermark_pos_vertical,
                                        self.watermark_pos_horizontal, self.watermark_pos_center)

    def get_watermark_position(self):
        return self.watermark_pos_horizontal, self.watermark_pos_vertical, self.watermark_pos_center

    def set_watermark_offsets(self, vertical=None, horizontal=None, center=False):
        temp_cap = cv2.VideoCapture(self.video_path)
        video_w = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_h = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        watermark_h, watermark_w, _ = self.watermark_cv_obj.shape
        offset_w = 0
        offset_h = 0

        if center == True:
            if vertical is None:
                offset_h = video_h/2 - watermark_h/2
            if horizontal is None:
                offset_w = video_w/2 - watermark_w/2

        if vertical == 'top':
            offset_h = 20
        elif vertical == 'bottom':
            offset_h = video_h - watermark_h - 20

        if horizontal == 'left':
            offset_w = 20
        elif horizontal == 'right':
            offset_w = video_w - watermark_w - 20

        temp_cap.release()
        return int(offset_h), int(offset_w)

    def resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w*r), height)
        else:
            r = width / float(w)
            dim = (width, int(h*r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def generate_frame_with_overlay(self, frame):
        frame_h, frame_w, _ = frame.shape
        overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')

        watermark_h, watermark_w, _ = self.watermark_cv_obj.shape
        offset_h, offset_w = self.set_watermark_offsets(self.watermark_pos_vertical, 
                                                        self.watermark_pos_horizontal, 
                                                        self.watermark_pos_center)

        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                if self.watermark_cv_obj[i,j][3] != 0:
                    overlay[i+offset_h, j+offset_w] = self.watermark_cv_obj[i, j]

        cv2.addWeighted(overlay, self.watermark_transparency, frame, 1.0, 0, frame)
        return frame    

    def overlay_watermark(self):
        not_terminated = True
        capture = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter(self.output_path, fourcc, capture.get(cv2.CAP_PROP_FPS), 
            (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:
            ret, frame = capture.read()
            if ret is True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                frame = self.generate_frame_with_overlay(frame)
                if self.show_processing == True:
                    cv2.imshow('frame', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                output.write(frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    not_terminated = False
                    break
            else:
                break

        capture.release()
        output.release()
        cv2.destroyAllWindows()
        return not_terminated

def signal_handler(sig, frame):
    print('terminated.')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    Watermark = WatermarkAdder()

    parser = argparse.ArgumentParser(prog='WatermarkAdder')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-v', '--video', required=True,
        help='load video by path', type=str, metavar='')
    required.add_argument('-w', '--watermark', required=True,
        help='load watermark by path', type=str, metavar='')
    parser.add_argument('-t', '--transparency', required=False,
        help='set transparency: [0.1-1.0]', type=float, default=0.25, metavar='')
    parser.add_argument('-p', '--processing', required=False,
        help='show processing', action='store_true')
    parser.add_argument('-q', '--quadrant', required=False,
        help='quadrant to place the watermark: [0-8]', type=int, default=8, choices=range(0, 9), metavar='')
    args = parser.parse_args()

    if not path.exists(args.video):
        print(args.video, 'does not exist.')
        sys.exit(1)
    if not path.exists(args.watermark):
        print(args.watermark, 'does not exist.')
        sys.exit(1)
    if not 0.1 <= args.transparency <= 1.0:
        print('transparency must be between 0.1 and 1.0.')
        parser.print_help()
        sys.exit(1)

    Watermark.set_video_path(args.video)
    Watermark.set_watermark_path(args.watermark)
    Watermark.set_watermark_transparency(args.transparency)
    Watermark.set_watermark_position_from_arg(args.quadrant)
    Watermark.set_show_processing(args.processing)

    print('overlaying', args.watermark, 'onto', args.video)
    ret = Watermark.overlay_watermark()
    if ret == True:
        print(args.watermark, 'overlayed on', args.video, 'and saved to', Watermark.get_output_path())
    else:
        print('terminated.')

if __name__ == '__main__':
    main()
