import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class InvertedPendulum(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xRef=0.0, randomParameters=False, randomSensor=False, randomActuator=False, render_mode=None):
        super().__init__()

        self.tau = 0.02
        self.tmax = 10.0 # Cada episódio tem 500 timesteps

        self.randomParameters = randomParameters
        self.randomSensor     = randomSensor
        self.randomActuator   = randomActuator
        self.render_mode      = render_mode

        self._set_parameters()

        self.xRef = xRef

        self.theta_threshold = 60*np.pi/180
        self.x_threshold = 0.25     

        high = np.array([self.x_threshold*2, np.finfo(np.float32).max, self.theta_threshold*2, np.finfo(np.float32).max, self.x_threshold*2, ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        self.xRef = 0
        self.nTimesteps = 0
        self.observation = np.zeros(4, dtype=np.float32)
        self.last_action = 0.0
        self.finish = False
        self.disturb = False
        self.disturb_force = 0.0
        self.disturb_steps = 0

        if render_mode == "human":
            pygame.init()

            self.screenWidth = 800
            self.screenHeight = 400

            self.simul = pygame.Rect(50, 100, self.screenWidth, self.screenHeight)
            self.textbox = pygame.Rect(50, self.simul.bottom + 30, 800, 150)

            self.screen = pygame.display.set_mode((1600, 900))
            pygame.display.set_caption('Inverted Pendulum RL')
            self.font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()

            self.cartWidth = 80
            self.cartHeight = 40
            self.pendulumLength = 150
            self.baseLine = 400

            self.c_white = (255, 255, 255)
            self.c_black = (0, 0, 0)
            self.c_cart = (164, 164, 164)
            self.c_pole = (164, 164, 164)
            self.c_ref = (100, 100, 100)

            self.plot_pos = pygame.Rect(900,  30, 650, 380)
            self.plot_vol = pygame.Rect(900, 460, 650, 380)

            self.MAX_HISTORY = 500
            self.hist_x = []
            self.hist_ref = []
            self.hist_theta = []
            self.hist_force = []

    def _set_parameters(self):
        if not self.randomParameters:
            self.g = 9.8
            self.M = 1.0 # massa do carro (kg)
            self.m = 0.1 # massa da haste (kg)
            self.l = 0.5 # metade do comprimento da haste (m)
        else:
            self.g = 9.8 + 0.098 * np.random.randn()
            self.M = 1.0 + 0.1 * np.random.randn()
            self.m = 0.1 + 0.01 * np.random.randn()
            self.l = 0.5 + 0.05 * np.random.randn()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._set_parameters()

        self.xRef = np.random.uniform(-self.x_threshold*0.7, self.x_threshold*0.7)

        self.observation = np.random.uniform(low=-0.02, high=0.02, size=(4,)).astype(np.float32)
        self.nTimesteps  = 0
        self.last_action = 0.0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = self.observation.copy()
        if self.randomSensor:
            obs += 0.01 * np.random.randn(4).astype(np.float32)
        return np.append(obs, self.xRef).astype(np.float32)

    def _noise_sensors(self, obs, noiseVar=0.01):
        return obs + noiseVar * np.random.randn(len(obs)).astype(np.float32)
    
    def limitRef(self):
        self.xRef = np.clip(self.xRef, -self.x_threshold, self.x_threshold)

    def step(self, action):
        force = float(action[0])
        force = np.clip(force, -10.0, 10.0)

        if self.randomActuator:
            force += 0.01 * np.random.randn()

        x1 = self.observation[0] # posição do carro
        x2 = self.observation[1] # velocidade do carro
        x3 = self.observation[2] # ângulo da haste
        x4 = self.observation[3] # velocidade angular da haste

        if self.disturb:
            self.disturb_force = np.random.choice([-1, 1]) * np.random.uniform(1.0, 6.0)
            self.disturb_steps = np.random.randint(10, 20)
            self.disturb = False
 
        if self.disturb_steps > 0:
            force += self.disturb_force
            force = np.clip(force, -20.0, 20.0)
            self.disturb_steps -= 1

        x4dot = (self.g * np.sin(x3) - np.cos(x3) * (force + self.m * self.l * x4**2 * np.sin(x3)) / (self.M + self.m)) / (self.l * (4.0/3.0 - self.m * np.cos(x3)**2 / (self.M + self.m)))
        x2dot = ((force + self.m * self.l * x4**2 * np.sin(x3)) / (self.M + self.m) - self.m * self.l * x4dot * np.cos(x3) / (self.M + self.m))

        self.observation[0] = x1 + self.tau * x2
        self.observation[1] = x2 + self.tau * x2dot
        self.observation[2] = x3 + self.tau * x4
        self.observation[3] = x4 + self.tau * x4dot

        self.last_action = force
        self.nTimesteps += self.tau

        erro_x = self.observation[0] - self.xRef
        erro_theta = self.observation[2]
        vel_carro = self.observation[1]
        vel_angular = self.observation[3]

        reward = (
            -4.0*erro_x**2
            -2.0*erro_theta**2                                                                                  #      
            -0.1*vel_carro**2     
            -0.1*vel_angular**2                                                                                 #
            -0.001*force**2                                                                                     #
        )

        if abs(erro_theta) < 0.05 and abs(erro_x) < 0.05:                                                       #
            reward += 2.0                                                                                       #

        terminated = bool(abs(erro_x) > self.x_threshold or abs(self.observation[2]) > self.theta_threshold)

        if terminated:
            reward -= 5.0

        truncated = bool(self.nTimesteps >= self.tmax)

        if self.render_mode == "human":
            self.hist_x.append(self.observation[0])
            self.hist_ref.append(self.xRef)
            self.hist_theta.append(self.observation[2])
            self.hist_force.append(force)
            if len(self.hist_x) > self.MAX_HISTORY:
                self.hist_x.pop(0)
                self.hist_ref.pop(0)
                # self.hist_theta.pop(0)
                self.hist_force.pop(0)

        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}

    def _draw_plot(self, rect, data, color, y_min, y_max, title, data2=None, color2=None, label=None, label2=None):
        pygame.draw.rect(self.screen, self.c_white, rect)
        pygame.draw.rect(self.screen, self.c_black, rect, 2)

        surf = self.font.render(title, True, self.c_black)
        self.screen.blit(surf, (rect.x + 5, rect.y + 4))

        if y_min < 0 < y_max:
            y_zero = rect.bottom - int((0 - y_min) / (y_max - y_min) * rect.height)
            pygame.draw.line(self.screen, (200, 200, 200), (rect.x, y_zero), (rect.right, y_zero), 1)

        def to_px(val, idx, total):
            px_x = rect.x + int(idx / max(total - 1, 1) * rect.width)
            val_c = np.clip(val, y_min, y_max)
            px_y  = rect.bottom - int((val_c - y_min) / (y_max - y_min) * rect.height)
            return px_x, px_y

        def draw_series(series, col):
            n = len(series)
            if n < 2:
                return
            pts = [to_px(v, i, n) for i, v in enumerate(series)]
            pygame.draw.lines(self.screen, col, False, pts, 2)

        draw_series(data, color)
        if data2 is not None:
            draw_series(data2, color2)

        offset = 0
        for lbl, col in [(label, color), (label2, color2)]:
            if lbl:
                pygame.draw.line(self.screen, col,
                    (rect.x + 10 + offset, rect.y + rect.height - 14),
                    (rect.x + 30 + offset, rect.y + rect.height - 14), 2)
                s = self.font.render(lbl, True, self.c_black)
                self.screen.blit(s, (rect.x + 34 + offset, rect.y + rect.height - 20))
                offset += 120

    def render(self):
        if self.render_mode != "human":
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.finish = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.hist_x     = []
                    self.hist_ref   = []
                    # self.hist_theta = []
                    self.hist_force = []
                    self.reset()

                elif event.key == pygame.K_LEFT:
                    self.xRef -= 0.02
                    self.limitRef()

                elif event.key == pygame.K_RIGHT:
                    self.xRef += 0.02
                    self.limitRef()

                elif event.key == pygame.K_SPACE:
                    self.disturb = True

        self.screen.fill(self.c_white)
        pygame.draw.rect(self.screen, self.c_black, self.simul, width=2)

        scale    = self.simul.width / (2 * self.x_threshold)   # px/m
        sim_cx   = self.simul.left + self.simul.width // 2     # pixel central x
        baseline = self.simul.bottom - 20                      # chão dentro do simul
 
        pygame.draw.line(self.screen, self.c_black, (self.simul.left + 2, baseline), (self.simul.right - 2, baseline), 2)
 
        x_ref_px = int(sim_cx + self.xRef * scale)
        x_ref_px = int(np.clip(x_ref_px, self.simul.left, self.simul.right))
        pygame.draw.line(self.screen, self.c_ref, (x_ref_px, self.simul.top + 2), (x_ref_px, baseline), 2)
 
        x_px = int(sim_cx + self.observation[0] * scale)
        x_px = int(np.clip(x_px, self.simul.left  + self.cartWidth // 2, self.simul.right - self.cartWidth // 2))
 
        cart_top = baseline - self.cartHeight
        cart_rect = pygame.Rect(x_px - self.cartWidth // 2, cart_top, self.cartWidth, self.cartHeight)

        pygame.draw.rect(self.screen, self.c_cart, cart_rect)
        pygame.draw.rect(self.screen, self.c_black, cart_rect, 2)
 
        pivot_x = x_px
        pivot_y = cart_top
 
        theta  = self.observation[2]
        pend_x = pivot_x + int(self.pendulumLength * np.sin(theta))
        pend_y = pivot_y - int(self.pendulumLength * np.cos(theta))
        pygame.draw.line(self.screen, self.c_pole, (pivot_x, pivot_y), (pend_x, pend_y), 6)
        pygame.draw.circle(self.screen, self.c_pole, (pend_x, pend_y), 8)
        
        pygame.draw.circle(self.screen, self.c_black, (pivot_x, pivot_y), 5)

        info = [
            f'xRef:  {self.xRef:.3f} m',
            f'x:     {self.observation[0]:.3f} m',
            f'theta: {np.degrees(self.observation[2]):.2f} deg',
            f'Force: {self.last_action:.2f} N',
        ]

        for idx, text in enumerate(info):
            surf = self.font.render(text, True, self.c_black)
            self.screen.blit(surf, (10, 10 + idx * 22))

        self._draw_plot(
            rect   = self.plot_pos,
            data   = self.hist_x,
            color  = (10, 186, 181),
            y_min  = -self.x_threshold,
            y_max  = +self.x_threshold,
            title  = "Position vs Reference (m)",
            data2  = self.hist_ref,
            color2 = (255, 0, 200),
            label  = "x",
            label2 = "xRef",
        )

        self._draw_plot(
            rect  = self.plot_vol,
            data  = self.hist_force,
            color = (5, 5, 5),
            y_min = -10.0,
            y_max = +10.0,
            title = "Applied Force (N)",
            label = "Force",
        )

        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()