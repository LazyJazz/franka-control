import pygame
import sys

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Joystick Test")

    # Initialize the joystick
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick connected.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick initialized: {joystick.get_name()}")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Read joystick axes
            axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            print(f"Joystick axes: {axes}")

            # Read joystick buttons
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            print(f"Joystick buttons: {buttons}")

            pygame.time.wait(100)

    except KeyboardInterrupt:
        print("Exiting joystick test.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()