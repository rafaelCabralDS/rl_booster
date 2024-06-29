from typing import Union
import pygame
from Box2D import *
from .world import World
from .constants import *

class TrackPainter:

    def __init__(self, render_mode: str):
        self.screen: Union[pygame.Surface, None] = None
        self.clock = None
        self.render_mode = render_mode

        self.CAMERA_W = VIEWPORT_SIZE
        self.CAMERA_H = VIEWPORT_SIZE
        self.camera = pygame.Rect(0, 0, self.CAMERA_W, self.CAMERA_H)

    def paint(self, world):

        cg_x, cg_y = world.fuselage.worldCenter
        self.camera.center = (cg_x * PIXELS_UNITS_SCALE, cg_y * PIXELS_UNITS_SCALE)
        if cg_y * PIXELS_UNITS_SCALE <= self.CAMERA_H / 2:
            self.camera.center = (self.camera.center[0], self.CAMERA_H / 2)

        if self.render_mode == "human":

            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.CAMERA_W, self.CAMERA_H)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.CAMERA_W, self.CAMERA_H))
        pygame.transform.scale(self.canvas, (PIXELS_UNITS_SCALE, PIXELS_UNITS_SCALE))

        # paint all elements here
        self._paintWorld(world)
        self._paintState(world)

        # booster_cg = calculateCG(world.booster)
        # self.__paintMarker__(*world.fuselage.worldCenter)

        # Update the display
        self.canvas = pygame.transform.flip(self.canvas, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.canvas, (0, 0))
            pygame.event.pump()
            self.clock.tick(RENDER_FPS)
            pygame.display.flip()


        else:  # rgb_Array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def _paintWorld(self, world):
        self._paintSky()
        self.__paintGround__(world.terrain)
        self._paintLaunchPad(world.launch_pad)
        self._paintBooster(world)
        self._paintParticles(world.particles)

    def _paintSky(self):
        pygame.draw.rect(
            self.canvas,
            (135, 206, 235),
            self.canvas.get_rect()
        )

    def __paintGround__(self, ground: b2Body):
        self.__paintBody__(ground, (50, 205, 50))

    def _paintLaunchPad(self, launch_pad):
        self.__paintBody__(launch_pad, (140, 140, 140))

    def _paintBooster(self, world):
        if world.nozzle is not None: self.__paintBody__(world.nozzle, (0, 0, 0))  # main engine
        self.__paintBody__(world.fuselage, (255, 255, 255))  # fuselage

        # legs
        self.__paintBody__(world.legs[0], (50, 50, 50))
        self.__paintBody__(world.legs[1], (50, 50, 50))

        # thrusters
        self.__paintBody__(world.sideThrusters[0], (160, 160, 160))
        self.__paintBody__(world.sideThrusters[1], (160, 160, 160))

    def _paintParticles(self, particles: list[b2Body]):

        for particle in particles:
            particle_power = min(1, max(0, particle.ttl))
            color = (255 * particle_power, 255 * particle_power * 0.2, 255 * (1 - particle_power))
            self.__paintBody__(particle, color)

    def __paintBody__(self, body: b2Body, color):
        for fixture in body.fixtures:
            shape = fixture.shape

            if isinstance(shape, b2EdgeShape):
                vertices = [((body.transform * v * PIXELS_UNITS_SCALE) - self.camera.topleft) for v in shape.vertices]
                pygame.draw.lines(self.canvas, color, closed=False, points=vertices, width=2)
            elif isinstance(shape, b2PolygonShape):
                vertices = [((body.transform * v * PIXELS_UNITS_SCALE) - self.camera.topleft) for v in shape.vertices]
                pygame.draw.polygon(self.canvas, color, vertices)
            elif isinstance(shape, b2CircleShape):
                circle_center = ((body.position * PIXELS_UNITS_SCALE) - self.camera.topleft)
                pygame.draw.circle(self.canvas, color, circle_center, int(shape.radius * PIXELS_UNITS_SCALE))
            else:
                raise Exception(f"The {type(shape)} painter was not implemented yet")

    def __paintMarker__(self, x, y):
        """
        Draws a black '+' sign at the x and y coordinates.
        :param x: normalized x position (0-1)
        :param y: normalized y position (0-1)
        """
        marker_size = 0.1
        pygame.draw.line(self.canvas, start_pos=(x * PIXELS_UNITS_SCALE, (y - marker_size) * PIXELS_UNITS_SCALE),
                         end_pos=(x * PIXELS_UNITS_SCALE, (y + marker_size) * PIXELS_UNITS_SCALE), width=1,
                         color=(255, 0, 0))
        pygame.draw.line(self.canvas, start_pos=((x - marker_size) * PIXELS_UNITS_SCALE, y * PIXELS_UNITS_SCALE),
                         end_pos=((x + marker_size) * PIXELS_UNITS_SCALE, y * PIXELS_UNITS_SCALE), width=1,
                         color=(255, 0, 0))

    def _paintState(self, world: World):
        font = pygame.font.Font(None, 15)  # You can provide a specific font file or use None for the default font

        # Set up the text
        text_lines = [f"V: {round(world.state.Vx,2)}, {round(world.state.Vy,2)} m/s",
                      f"(x,y): {round(world.state.x,2)}, {round(world.state.y,2)} m",
                      f"w: {round(world.state.w,2)} rad/s",
                      f"Angle: {world.state.angle:{.2}} rad"
                      ]
        text_surfaces = [font.render(line, True, (255, 0, 0)) for line in text_lines]
        text_surfaces = [pygame.transform.flip(text, False, True) for text in text_surfaces]

        # Set up the text positions
        text_positions = [(10,
                           self.CAMERA_H * 0.9 - (i * surface.get_height())) for i, surface in
                          enumerate(text_surfaces)]

        for surface, position in zip(text_surfaces, text_positions):
            self.canvas.blit(surface, position)


    def dispose(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
        # Free resources
        self.screen = None
        # self.canvas = None
        self.clock = None



if __name__ == "__main__":
    painter = TrackPainter(render_mode="human")
    from world import *
    from util_methods import load_config

    config = EasyDict()
    config.update(load_config(f"./env.yaml"))
    w = World(seed=43, initial_state=config.initial_condition)
    landed = False
    while not landed:
        painter.paint(w)
        landed = w.step([0, 0, 0])

    # pygame.image.save(painter.screen, "booster_on_ground.png")