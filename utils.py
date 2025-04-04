import random
import numpy as np
import cv2
import pygame


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)



class Balloon:
    def __init__(self, x, y, balloon_type, balloon_images):
        self.x = x
        self.y = y
        self.speed = 7
        self.type = balloon_type
        self.image = balloon_images[balloon_type]
        self.balloon_images = balloon_images
        self.rect = self.image.get_rect(topleft=(self.x, self.y))
        self.hits_required = 2 if balloon_type == "regular4" or "regular3" else 1

    def move(self):
        self.y -= self.speed
        self.rect.y = self.y

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def hit(self):
        self.hits_required -= 1
        if self.type == "regular4" and self.hits_required == 1:
            self.type = "regular2"
            self.image = self.balloon_images["regular2"]
            return False
        elif self.type == "regular3" and self.hits_required == 1:
            self.type = "regular5"
            self.image = self.balloon_images["regular5"]
            return False
        return self.hits_required == 0




def set_up_roi(pts_src, pygame):
    cap = cv2.VideoCapture(0)
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_src) < 4:
            pts_src.append((x, y))

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", click_event)

    # Wait for ROI Selection
    while len(pts_src) < 4:
        ret, frame = cap.read()
        if not ret:
            break
        
        for pt in pts_src:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)  # Green dots for selected points
        
        cv2.imshow("Camera", frame)
        pygame.display.flip()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cap.release()
    cv2.destroyAllWindows()
    return pts_src