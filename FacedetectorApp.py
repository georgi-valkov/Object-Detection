from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.core.image import Image as CoreImage

from Detector import Detector
import cv2
import copy
import os
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from io import BytesIO
from pymediainfo import MediaInfo


class Capture(Image):
    def __init__(self, **kwargs):
        super(Capture, self).__init__(**kwargs)
        self.capture = None
        self.fps = 15
        self.running = False
        self.texture = Texture.create(size=(1920, 1080), colorfmt='bgr')
        # Firing up the detector
        self.detector = Detector(graph='models/frozen_graph.pb', labels='models/face_label_map.pbtxt')
        self.path = ''

    # Before calling this method capture must be set to cv2.VideoCapture object by a button
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Make a copy of the clear frame
            clear_frame = copy.deepcopy(frame)
            # Actual object detection
            frame, scores, num_detections, boxes = self.detector.detect(frame, resizing_factor=4)
            # Convert frame into a texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Add detected faces to the ui
            for i in range(int(num_detections[0])):
                if scores[0][i] > 0.95: # Add faces if they get detected with 95% certainty
                    # Extract faces and put them on the ui
                    height, width, channels = frame.shape
                    # For All detected objects in the picture
                    # Bounding box coordinates
                    ymin = int((boxes[0][i][0] * height))
                    xmin = int((boxes[0][i][1] * width))
                    ymax = int((boxes[0][i][2] * height))
                    xmax = int((boxes[0][i][3] * width))
                    lp_np = clear_frame[ymin:ymax, xmin:xmax]
                    # Make the face a texture
                    lp_buf1 = cv2.flip(lp_np, 0)
                    lp_buf = lp_buf1.tostring()
                    lp_image_texture = Texture.create(size=(lp_np.shape[1], lp_np.shape[0]), colorfmt='bgr')
                    lp_image_texture.blit_buffer(lp_buf, colorfmt='bgr', bufferfmt='ubyte')
                    # Add it to ui
                    image = Image()
                    image.size_hint_x = None
                    image.texture = lp_image_texture
                    self.parent.ids.face_grid.add_widget(image, len(self.parent.ids.face_grid.children))

            self.texture = image_texture

    # Calls update function with a clock
    def start(self):
        Clock.schedule_interval(self.update, 1 / self.fps)

    # Unschedule update function / pause
    def pause(self):
        Clock.unschedule(self.update)


class MainLayout(BoxLayout):
    loadfile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        self.ids.video.path = os.path.join(path, filename[0])

        self.dismiss_popup()


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Facedetector(App):
    main_layout = None

    def build(self):
        self.title = 'Face Detector'
        self.main_layout = MainLayout()
        return self.main_layout

    # Start button
    def on_press_start(self):
        if self.main_layout.ids.vid_toggle.state == 'down' and not self.main_layout.ids.video.running:

            media_info = MediaInfo.parse(self.main_layout.ids.video.path)
            for track in media_info.tracks:
                if track.track_type == 'Video' or track.track_type == 'Audio':

                    self.main_layout.ids.video.capture = cv2.VideoCapture(
                        self.main_layout.ids.video.path)
                    self.main_layout.ids.vid_toggle.disabled = True
                    self.main_layout.ids.cam_toggle.disabled = True
                else:
                    self.main_layout.ids.video.capture = None

        elif self.main_layout.ids.cam_toggle.state == 'down' and not self.main_layout.ids.video.running:
            self.main_layout.ids.video.capture = cv2.VideoCapture(0)
            cam_fps = self.main_layout.ids.video.capture.get(cv2.CAP_PROP_FPS)
            if not cam_fps > 0:
                self.main_layout.ids.video.capture = None
            else:
                self.main_layout.ids.cam_toggle.disabled = True
                self.main_layout.ids.vid_toggle.disabled = True

        if self.main_layout.ids.video.capture is not None:
            if not self.main_layout.ids.video.running:
                self.main_layout.ids.start_button.disabled = True
                self.main_layout.ids.pause_button.disabled = False
                self.main_layout.ids.video.start()
                self.main_layout.ids.video.running = True
            else:
                self.main_layout.ids.start_button.disabled = True
                self.main_layout.ids.pause_button.disabled = False
                self.main_layout.ids.video.start()
        else:
            if self.main_layout.ids.cam_toggle.state == 'down':
                text = 'No camera detected..'
            elif self.main_layout.ids.vid_toggle.state == 'down':
                text = 'No video detected..'

            font = ImageFont.truetype('assets/FFF_Tusj.ttf', 150)
            img = PILImage.new('RGB', (1920, 1080), color=(0, 0, 0))

            draw = ImageDraw.Draw(img)
            draw.text((1920 / 8, 1080 / 2), text, font=font, fill=(255, 255, 255))

            data = BytesIO()
            img.save(data, format='png')
            data.seek(0)
            im  = CoreImage(BytesIO(data.read()), ext='png')
            self.main_layout.ids.video.texture = im.texture

    # Pause button
    def on_press_pause(self):
        self.main_layout.ids.start_button.disabled = False
        self.main_layout.ids.pause_button.disabled = True
        self.main_layout.ids.video.pause()

    # Stop button
    def on_press_stop(self):
        if self.main_layout.ids.video.running:
            self.main_layout.ids.start_button.disabled = False
            self.main_layout.ids.pause_button.disabled = True
            self.main_layout.ids.video.pause()
            self.main_layout.ids.video.capture.release()
            self.main_layout.ids.vid_toggle.disabled = False
            self.main_layout.ids.cam_toggle.disabled = False
            self.main_layout.ids.video.running = False
            self.main_layout.ids.video.texture = Texture.create(size=(1920, 1080), colorfmt='bgr')
            self.main_layout.ids.face_grid.clear_widgets()

    # Close the app properly
    def on_stop(self):
        if self.main_layout.ids.video.capture is not None:
            self.main_layout.ids.video.capture.release()


if __name__=='__main__':

    Window.size = (600, 1000)
    Facedetector().run()
