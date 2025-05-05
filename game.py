import pygame
import random
from findingBalloons import *
from utils import *

pygame.init()
WIDTH, HEIGHT = 1900, 1050
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
    # "regular1": pygame.transform.scale(load_transparent_image("images/regular1.png"), BALLOON_SIZE),
    # "regular6": pygame.transform.scale(load_transparent_image("images/regular6.png"), BALLOON_SIZE_2),
    "regular2": pygame.transform.scale(load_transparent_image("images/regular2.png"), BALLOON_SIZE),
    "number": pygame.transform.scale(load_transparent_image("images/regular4.png"), BALLOON_SIZE),
    "regular5": pygame.transform.scale(load_transparent_image("images/regular5.png"), BALLOON_SIZE),
    # "energy": pygame.transform.scale(load_transparent_image("images/energy1.png"), BALLOON_SIZE),
    "bomb": pygame.transform.scale(load_transparent_image("images/bomb_1.png"), BALLOON_SIZE)
}

balloons = []
balloon_id = 1
spawn_timer = 0
score = 0
pts_src = []
previous_mapping = {}
pts_src = [(300, 255), (266, 962), (1584, 961), (1564, 279)]
camera_number = 0
# pts_src = set_up_roi(camera_number, pts_src, pygame)
# print(pts_src)

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
        balloons.append(Balloon(balloon_id, x_pos, HEIGHT, balloon_type, balloon_images))
        balloon_id += 1
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

        cv2.imshow("thresh_hold_frame", thresh_hold_frame)
        frame_contour, bounding_boxes = find_contours(thresh_hold_frame)
        mapping = detect_ballon(warped_roi, bounding_boxes, balloons, BALLOON_SIZE)
        for (x1, y1, x2, y2), balloon in mapping.items():
            color = (0, 255, 0)
            font_color = (0, 0, 0)
            if balloon.type == "bomb":
                label = f"{balloon.type}_{balloon.id}"
                color = (0, 0, 255)
                font_color = (255, 255, 255)
            elif balloon.type == f"energy":
                label = f"{balloon.type}_{balloon.id}"
                color = (255, 255, 0)
            elif balloon.type == f"number":
                label = f"{balloon.type}_{balloon.id}"
                color = (255, 0, 0)
                font_color = (255, 255, 255)
            else:
                label = f"{balloon.type}_{balloon.id}"

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width > 300 or box_height > 300:
                continue

            cv2.rectangle(warped_roi, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(warped_roi, (x1, y1 - 25), (x1 + 120, y1), color, -1)
            cv2.putText(warped_roi, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color, 2)

        yellow_obj_detected = detect_yellow_obj(warped_roi)
        collision = detect_collision(yellow_obj_detected, mapping)
        if collision:
            print(f"Type: {collision[1].type}_{collision[1].id}")
            balloon = collision[1]
            popped = balloon.hit()
            if popped:
                if balloon.type == "bomb":
                    score -= 1
                else:
                    score += 1
                balloons.remove(balloon)

        cv2.imshow("test", warped_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
