import pygame
import random

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

# Balloon spawn settings
balloons = []
spawn_timer = 0  # Timer to control balloon spawn rate

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

    # Refresh display
    pygame.display.flip()
    clock.tick(30)  # Control frame rate

pygame.quit()