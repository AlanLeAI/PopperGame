import pygame
import random
import numpy as np
import cv2
from findingBalloons import *

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Popper Game - Balloon Animation")

WHITE = (255, 255, 255)

def load_transparent_image(path):
    image = pygame.image.load(path).convert_alpha() 
    image.set_colorkey(WHITE) 
    return image

# Load balloon images (Replace with actual image paths)
regular1_balloon_img = load_transparent_image("images/regular1.png")
regular2_balloon_img = load_transparent_image("images/regular2.png")
regular3_balloon_img = load_transparent_image("images/regular3.png")
regular4_balloon_img = load_transparent_image("images/regular4.png")
energy_balloon_img =load_transparent_image("images/energy1.png")
bomb_balloon_img = load_transparent_image("images/bomb.png")

# Scale images to a suitable size
BALLOON_WIDTH, BALLOON_HEIGHT = 100, 120
regular1_balloon_img = pygame.transform.scale(regular1_balloon_img, (BALLOON_WIDTH, BALLOON_HEIGHT))
regular2_balloon_img = pygame.transform.scale(regular2_balloon_img, (BALLOON_WIDTH, BALLOON_HEIGHT))
regular3_balloon_img = pygame.transform.scale(regular3_balloon_img, (BALLOON_WIDTH, BALLOON_HEIGHT))
regular4_balloon_img = pygame.transform.scale(regular4_balloon_img, (BALLOON_WIDTH, BALLOON_HEIGHT))
energy_balloon_img = pygame.transform.scale(energy_balloon_img, (BALLOON_WIDTH, BALLOON_HEIGHT))
bomb_balloon_img = pygame.transform.scale(bomb_balloon_img, (BALLOON_WIDTH, BALLOON_HEIGHT))


# Define balloon class
class Balloon:
    def __init__(self, x, y, balloon_type):
        self.x = x
        self.y = y
        self.speed = 3
        self.balloon_type = balloon_type

        # Assign image based on type
        if balloon_type == "regular1":
            self.image = regular1_balloon_img
        elif balloon_type == "regular2":
            self.image = regular2_balloon_img
        elif balloon_type == "regular3":
            self.image = regular3_balloon_img
        elif balloon_type == "regular4":
            self.image = regular4_balloon_img
        elif balloon_type == "energy":
            self.image = energy_balloon_img
        elif balloon_type == "bomb":
            self.image = bomb_balloon_img

    def move(self):
        self.y -= self.speed

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def is_clicked(self, pos):
        x, y = pos
        return self.x <= x <= self.x + BALLOON_WIDTH and self.y <= y <= self.y + BALLOON_HEIGHT


# Balloon spawn settings
balloons = []
spawn_timer = 0 
score = 0
# Game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    spawn_timer += 1
    if spawn_timer > 50:  # Adjust spawn rate
        x_pos = random.randint(50, WIDTH - 100)
        balloon_type = random.choice(["regular1", "regular2", "regular3", "regular4", "energy", "bomb"])
        balloons.append(Balloon(x_pos, HEIGHT, balloon_type))
        spawn_timer = 0

    for balloon in balloons[:]:
        balloon.move()
        balloon.draw(screen)

        # Remove balloon if it moves off the top of the screen
        if balloon.y < -70:
            balloons.remove(balloon)

    # score_text = font.render(f"Score: {score}", True, RED)
    # screen.blit(score_text, (10, 10))  # Top left
            
    frame = pygame.surfarray.array3d(screen)
    frame = np.rot90(frame, k=3)
    frame = np.flip(frame, axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    thresh_hold_frame = threshold_frame(frame)
    frame_contour, bounding_boxes = find_contours(thresh_hold_frame)
    mapping = {}

    for box in bounding_boxes:
        if len(bounding_boxes) > 0:
            x1,y1 = box[0]
            x2,y2 =box[1]
            # print((x1,y1,x2,y2))
            extract_balloon = frame[y1:y2, x1:x2, :]
            lower_black_bgr = (0, 0, 0)
            upper_black_bgr = (100, 100, 100)
            mask = cv2.inRange(extract_balloon, lower_black_bgr, upper_black_bgr, cv2.THRESH_BINARY_INV)
            # cv2.imshow("test", mask)
            if np.sum(mask/255.0) > 2000:
                mapping[(x1,y1,x2,y2)] = "bomb"
                continue
            elif np.sum(mask/255.0) == 0:
                mapping[(x1,y1,x2,y2)] = "balloon"
                continue

            se_number = get_structure_elements("images/regular3.png")
            mask2 = cv2.erode(mask, se_number)
            if np.sum(mask2) < 50:
                mapping[(x1,y1,x2,y2)] = "balloon_2"
                continue

            se_energy = get_structure_elements("images/energy1.png")
            mask3 = cv2.erode(mask, se_number)
            if np.sum(mask3) < 50:
                mapping[(x1,y1,x2,y2)] = "energy"
                continue
    

    for (x1, y1, x2, y2), label in mapping.items():
        color = (0, 255, 0) 
        if label == "Bomb":
            color = (0, 0, 0) 
        elif label == "Energy":
            color = (255, 255, 0) 

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.rectangle(frame, (x1, y1 - 20), (x1 + 80, y1), color, -1)

        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    

    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    # Refresh display
    pygame.display.flip()
    clock.tick(30)  # Control frame rate

pygame.quit()