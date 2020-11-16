import cv2
import numpy as np

class WatermarkAdder:
    def __init__(self):
        self.video_path = None
        self.watermark_path = None
        self.watermark_cv_obj = None
        self.watermark_pos_horizontal = None
        self.watermark_pos_vertical = None
        self.watermark_pos_center = False
        self.output_path = "./output.avi"
        
        # for testing purposes, will be removed when GUI is created
        self.set_video_path('./driving.mp4')
        self.set_watermark_path('./watermark.png')


        self.show_processing = True
        self.watermark_transparency = 0.5
        self.overlay_watermark()

    def set_video_path(self, path):
        self.video_path = path
    
    def set_watermark_path(self, path):
        self.watermark_path = path

    def set_output_path(self, path):
        self.output_path = path

    def set_watermark_transparency(self, value):
        self.watermark_transparency = value

    def toggle_show_processing(self):
        self.show_processing = not self.show_processing

    def set_watermark_position(self, vertical=None, horizontal=None, center=False):
        if vertical is not None:
            self.watermark_pos_vertical = vertical
        if horizontal is not None:
            self.watermark_pos_horizontal = horizontal
        if center != self.watermark_pos_center:
            self.watermark_pos_center = center

    def set_watermark_offsets(self, vertical=None, horizontal=None, center=False):
        if vertical == None and horizontal == None and center == False:
            return self.set_watermark_offsets(vertical="bottom", horizontal="right")

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

        if vertical == "top":
            offset_h = 20
        elif vertical == "bottom":
            offset_h = video_h - watermark_h - 20
        
        if horizontal == "left":
            offset_w = 20
        elif horizontal == "right":
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
        frame_h, frame_w, frame_c = frame.shape
        overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')

        watermark_h, watermark_w, watermark_c = self.watermark_cv_obj.shape
        offset_h, offset_w = self.set_watermark_offsets()

        for i in range(0, watermark_h):
            for j in range(0, watermark_w):
                if self.watermark_cv_obj[i,j][3] != 0:
                    overlay[i+offset_h, j+offset_w] = self.watermark_cv_obj[i, j]

        cv2.addWeighted(overlay, self.watermark_transparency, frame, 1.0, 0, frame)
        return frame    

    def overlay_watermark(self):
        if self.video_path is None or self.watermark_path is None:
            if self.video_path:
                print("Configure video path.")
            else:
                print("Configure watermark path.")
            return 1

        self.watermark_cv_obj = cv2.imread(self.watermark_path)
        self.watermark_cv_obj = self.resize_image(self.watermark_cv_obj, height=50)
        self.watermark_cv_obj = cv2.cvtColor(self.watermark_cv_obj, cv2.COLOR_BGR2BGRA)

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
                    break
            else:
                break

        capture.release()
        output.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    WatermarkAdder()