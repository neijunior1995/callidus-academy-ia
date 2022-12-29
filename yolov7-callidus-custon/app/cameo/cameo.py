import cv2

from app.cameo.managers import CaptureManager, WindowManager


class Cameo(object):
    def __init__(self,model):
        self.model = model
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                self.onImage(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.

        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        escape -> Quit.

        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
            self._captureManager.startWritingVideo(
                    'screencast.avi')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    'screencas.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._captureManager.close_can()
            self._windowManager.destroyWindow()

    def labelDraw(self,image,detect, color_rec = (255, 255, 0), rec_width = 2, font = cv2.FONT_HERSHEY_COMPLEX_SMALL, width_font = 0.5, color_font = (255,0,0)):
        for det in detect:
            x,y,x2,y2 = det[:4]
            cv2.rectangle(image, (int(x) , int(y)), (int(x2) , int(y2)), color_rec, rec_width)
            names = self.model.class_names
            label = "Class {0}, conf {conf:.2f}".format(names[int(det[5])],conf = det[4])
            cv2.putText(image,label,(int(x),int(y)),font,width_font,color_font)

    def onImage(self,frame):
        image_display = frame.copy()
        image_display = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)
        img_size = image_display.shape
	
        detect = self.model.processFrame(image_display)
        print(detect)
        self.labelDraw(frame,detect)

if __name__ == '__main__':
	Cameo().run()
