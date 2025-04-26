import pygame
import random
import numpy as np
import cv2
from findingBalloons import *
from utils import *

pygame.init()
WIDTH, HEIGHT = 1400, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Popper Game")
FONT = pygame.font.SysFont(None, 50)


def display_score(screen, score):
    score_text = FONT.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (24, 24))


BALLOON_SIZE = (238, 285)
BALLOON_SIZE_2 = (220, 285)


def load_transparent_image(path):
    image = pygame.image.load(path).convert_alpha()
    image.set_colorkey(WHITE)
    return image


balloon_images = {
    "regular1": pygame.transform.scale(load_transparent_image("images/regular1.png"), BALLOON_SIZE),
    "regular2": pygame.transform.scale(load_transparent_image("images/regular2.png"), BALLOON_SIZE),
    # "regular6": pygame.transform.scale(load_transparent_image("images/regular6.png"), BALLOON_SIZE_2),
    "regular4": pygame.transform.scale(load_transparent_image("images/regular4.png"), BALLOON_SIZE),
    "regular5": pygame.transform.scale(load_transparent_image("images/regular5.png"), BALLOON_SIZE),
    "energy": pygame.transform.scale(load_transparent_image("images/energy1.png"), BALLOON_SIZE),
    "bomb": pygame.transform.scale(load_transparent_image("images/bomb.png"), BALLOON_SIZE)
}

balloons = []
spawn_timer = 0
score = 0
pts_src = []
camera_number = 0
pts_src = set_up_roi(camera_number, pts_src, pygame)

# Game loop
cap = cv2.VideoCapture(camera_number)
running = True
clock = pygame.time.Clock()

while running:
    ret, frame = cap.read()
    warped_roi = None

    if len(pts_src) == 4:
        pts_dst = np.float32([(0, 0), (0, HEIGHT), (WIDTH, HEIGHT), (WIDTH, 0)])
        M = cv2.getPerspectiveTransform(np.float32(pts_src), pts_dst)
        warped_roi = cv2.warpPerspective(frame, M, (WIDTH, HEIGHT))
        frame_display = cv2.cvtColor(warped_roi, cv2.COLOR_BGR2RGB)
        frame_display = np.transpose(frame_display, (1, 0, 2))
        cv2.imshow("frame", warped_roi)


    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for balloon in balloons[:]:
                if balloon.is_clicked(mouse_pos):
                    popped = balloon.hit()
                    if popped:
                        if balloon.type == "bomb":
                            score -= 1
                        else:
                            score += 1
                        balloons.remove(balloon)

    spawn_timer += 1
    if spawn_timer > 30:
        x_pos = random.randint(119, WIDTH - 237)
        balloon_type = random.choice(list(balloon_images.keys()))
        balloons.append(Balloon(x_pos, HEIGHT, balloon_type, balloon_images))
        spawn_timer = 0

    for balloon in balloons[:]:
        balloon.move()
        balloon.draw(screen)

        if balloon.y < -120:
            balloons.remove(balloon)

    display_score(screen, score)

    frame = pygame.surfarray.array3d(screen)
    frame = np.rot90(frame, k=3)
    frame = np.flip(frame, axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if warped_roi is not None:
        thresh_hold_frame = threshold_frame(warped_roi)
        
        frame_contour, bounding_boxes = find_contours(thresh_hold_frame)
        mapping = detect_ballon(warped_roi, bounding_boxes, BALLOON_SIZE)
        for (x1, y1, x2, y2), label in mapping.items():
            color = (0, 255, 0)
            if label == "Bomb":
                color = (0, 0, 0)
            elif label == "Energy":
                color = (255, 255, 0)

            cv2.rectangle(warped_roi, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(warped_roi, (x1, y1 - 20), (x1 + 80, y1), color, -1)
            cv2.putText(warped_roi, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("test", warped_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
