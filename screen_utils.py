import pyautogui
import cv2
import numpy as np
import threading
import time

class ScreenShareManager:
    def __init__(self, fps=15):
        self.fps = fps
        self.running = False
        self.thread = None
        self.frame = None
        self.lock = threading.Lock()
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_screen)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _capture_screen(self):
        while self.running:
            try:
                # Capture screen
                img = pyautogui.screenshot()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with self.lock:
                    self.frame = frame
                
                time.sleep(1 / self.fps)
            except Exception as e:
                print(f"Screen capture error: {e}")
                break
    
    def get_frame(self):
        with self.lock:
            return self.frame if self.frame is not None else np.zeros((300, 500, 3), dtype=np.uint8)