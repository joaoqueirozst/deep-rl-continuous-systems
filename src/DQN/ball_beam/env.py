import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class BallBeamSystem(gym.Env):
    metadata = {"render_modes": ["human"]}

    DISCRETE_ACTIONS = np.linspace(-12.0, 12.0, 21) # Discrete actions

    def __init__(self, setpoint=0.0, randomParameters=False, randomSensor=False, randomActuator=False, render_mode=None):
        super().__init__()

        self.tau = 0.02
        self.tmax = 10.0

        self.randomParameters = randomParameters
        self.randomSensor = randomSensor
        self.randomActuator  = randomActuator
        self.render_mode = render_mode

        self._set_parameters()

        self.setpoint = setpoint
        self.limitRef()

        self.state = np.zeros(6, dtype=np.float32)
        self.last_action = 0.0
        self.nTimesteps = 0

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.DISCRETE_ACTIONS)) # CHANGE

        self.finish = False
        self.disturb = False

        # Pygame só inicializa se render_mode="human"
        if render_mode == "human":
            pygame.init()

            self.screen = pygame.display.set_mode((1600, 900))
            pygame.display.set_caption('Ball and Beam RL')
            self.font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()

            self.screenWidth = 800
            self.screenHeight = 400

            self.simul = pygame.Rect(50, 50, self.screenWidth, self.screenHeight)
            self.textbox = pygame.Rect(50, self.simul.bottom + 30, 800, 150)

            self.beam_pivot = (self.simul.centerx, self.simul.centery)

            self.px_per_m = 500
            self.beam_thickness = 4
            self.ball_radius = int(self.R * self.px_per_m)

            self.c_setpoint = (128, 128, 128)
            self.c_white = (255, 255, 255)
            self.c_black = (0, 0, 0)
            self.c_ref = (168, 139, 235)
            self.c_pos = (5, 5, 5)
            self.c_voltage = (150, 150, 150)

            self.plot_pos = pygame.Rect(900,  30, 650, 380)
            self.plot_vol = pygame.Rect(900, 460, 650, 380)

            self.maxHistory = 500
            self.hist_pos = []
            self.hist_ref = []
            self.hist_vol = []

    def _set_parameters(self):
        if not self.randomParameters:
            self.g      = 9.8
            self.mb     = 0.11
            self.R      = 0.04
            self.beam   = 1.0
            self.J      = 9.99e-6
            self.Jb     = 1.71e-3
            self.Rm     = 2.0
            self.Lm     = 0.5
            self.Km     = 0.0662
            self.atrito = 0.05
            self.Ke     = 0.05

        else:
            self.g      = 9.8     + 0.098   * np.random.randn()
            self.mb     = 0.11    + 0.0011  * np.random.randn()
            self.R      = 0.04    + 0.0004  * np.random.randn()
            self.beam   = 1.0     + 0.01    * np.random.randn()
            self.J      = 9.99e-6 + 9.99e-8 * np.random.randn()
            self.Jb     = 1.71e-3 + 1.71e-5 * np.random.randn()
            self.Rm     = 2.0     + 0.02    * np.random.randn()
            self.Lm     = 0.5     + 0.005   * np.random.randn()
            self.Km     = 0.0662  + 6.62e-4 * np.random.randn()
            self.atrito = 0.05    + 5e-4    * np.random.randn()
            self.Ke     = 0.05    + 5e-4    * np.random.randn()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._set_parameters()

        self.setpoint = np.random.uniform(-0.2, 0.2)
        self.nTimesteps = 0
        self.last_action = 0.0
        self.state = np.array(
            [0.05 * np.random.randn(), 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )

        obs = self.normalize_state(self.get_state())
        return obs, {}

    def get_state(self):
        obs = self.state.copy()
        if self.randomSensor:
            obs += 0.01 * np.random.randn(4).astype(np.float32)
        return np.append(obs, self.setpoint).astype(np.float32)

    def normalize_state(self, state):
        x, ball_vel, theta, theta_vel, i, erro_int, setpoint = state

        state_norm = np.array([
            x/(self.beam/2),
            ball_vel/5.0,
            theta/0.4,
            theta_vel/10.0,
            i/5.0,
            erro_int/0.2,
            setpoint/(self.beam/2)
        ], dtype=np.float32)

        return np.clip(state_norm, -5.0, 5.0)
    
    def limitRef(self):
        min = -(self.beam/2)
        max = +(self.beam/2)
        self.setpoint = np.clip(self.setpoint, min, max)

    def step(self, action):
        voltage_action = float(self.DISCRETE_ACTIONS[int(action)])
        voltage_action = np.clip(voltage_action, -12.0, 12.0)

        if self.randomActuator:
            voltage_action += 0.01 * np.random.randn()

        x, ball_vel, theta, theta_vel, i, erro_int = self.state

        erro = x - self.setpoint
        erro_int += self.tau * erro
        erro_int = np.clip(erro_int, -0.2, 0.2)

        # Elétrica do motor
        i_dot = (voltage_action - self.Rm * i - self.Ke * theta_vel) / self.Lm # atrito viscoso no eixo da viga
        i += self.tau * i_dot

        # Mecânica da viga
        theta_ace = (self.Km * i - self.atrito * theta_vel) / self.Jb
        theta_vel += self.tau * theta_ace
        theta += self.tau * theta_vel

        # Dinâmica da bola
        k = self.g / (1 + self.J / (self.mb * self.R ** 2))
        ball_acc = k * theta - self.R * theta_ace
        ball_vel += self.tau * ball_acc
        x += self.tau * ball_vel

        x = np.clip(x, -self.beam/2, self.beam/2)
        theta = np.clip(theta, -0.4, 0.4)

        if self.disturb:
            ball_vel += np.random.choice([-1.0, 1.0])*np.random.uniform(0.5, 1.5)
            self.disturb = False

        self.state = np.array([x, ball_vel, theta, theta_vel, i, erro_int], dtype=np.float32)
        self.last_action = voltage_action
        self.nTimesteps += self.tau

        erro_norm = erro/(self.beam/2)

        reward = (
            -erro_norm**2
            -0.5*ball_vel**2
            -0.05*theta**2
            -0.001*action**2
        )

        if abs(erro) < 0.05 and abs(ball_vel) < 0.05:
            reward += 2.0

        terminated = bool(abs(x) >= self.beam / 2)
        if terminated:
            reward -= 5.0

        truncated = bool(self.nTimesteps >= self.tmax)

        obs = self.normalize_state(self.get_state())

        if self.render_mode == "human":
            self.hist_pos.append(x)
            self.hist_ref.append(self.setpoint)
            self.hist_vol.append(voltage_action)
            if len(self.hist_pos) > self.maxHistory:
                self.hist_pos.pop(0)
                self.hist_ref.pop(0)
                self.hist_vol.pop(0)

        return obs, reward, terminated, truncated, {}

    def draw_plot(self, rect, data, color, y_min, y_max, title, data2=None, color2=None, label=None, label2=None):
        pygame.draw.rect(self.screen, self.c_white, rect)
        pygame.draw.rect(self.screen, self.c_black, rect, 2)

        surf = self.font.render(title, True, self.c_black)
        self.screen.blit(surf, (rect.x + 5, rect.y + 4))

        if y_min < 0 < y_max:
            y_zero = rect.bottom - int((0 - y_min) / (y_max - y_min) * rect.height)
            pygame.draw.line(self.screen, (200, 200, 200), (rect.x, y_zero), (rect.right, y_zero), 1)

        def to_px(val, idx, total):
            px_x = rect.x + int(idx / max(self.maxHistory - 1, 1) * rect.width)
            val_c = np.clip(val, y_min, y_max)
            px_y = rect.bottom - int((val_c - y_min) / (y_max - y_min) * rect.height)
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
                    if self.render_mode == "human":
                        self.hist_pos = []
                        self.hist_ref = []
                        self.hist_vol = []
                        self.reset()

                elif event.key == pygame.K_LEFT:
                    self.setpoint -= 0.02
                    self.limitRef()

                elif event.key == pygame.K_RIGHT:
                    self.setpoint += 0.02
                    self.limitRef()

                elif event.key == pygame.K_SPACE:
                    self.disturb = True

        self.screen.fill(self.c_white)
        pygame.draw.rect(self.screen, self.c_black, self.simul, width=2)

        x = self.state[0]
        theta = self.state[2]

        x_pivot, y_pivot = self.beam_pivot

        vet_tan = np.array([np.cos(theta), np.sin(theta)])
        normal_beam = np.array([np.sin(theta), -np.cos(theta)])

        tamanho_beam = (self.beam / 2) * self.px_per_m

        p1 = (x_pivot - tamanho_beam * vet_tan[0], y_pivot - tamanho_beam * vet_tan[1])
        p2 = (x_pivot + tamanho_beam * vet_tan[0], y_pivot + tamanho_beam * vet_tan[1])

        pygame.draw.line(self.screen, self.c_black, p1, p2, self.beam_thickness)

        ball_position = (np.array([x_pivot, y_pivot]) + vet_tan * (x * self.px_per_m) + normal_beam * self.ball_radius)

        x_ball, y_ball = ball_position.astype(int)

        pygame.draw.circle(self.screen, self.c_white, (x_ball, y_ball), self.ball_radius)
        pygame.draw.circle(self.screen, self.c_black, (x_ball, y_ball), self.ball_radius, 2)

        x_sp = x_pivot + self.setpoint * self.px_per_m * np.cos(theta)
        y_sp = y_pivot + self.setpoint * self.px_per_m * np.sin(theta)

        pygame.draw.circle(self.screen, self.c_setpoint, (int(x_sp), int(y_sp)), 6)

        info = [
            f'Setpoint: {self.setpoint:.3f}',
            f'Position: {x:.3f}',
            f'Angle: {theta:.3f}',
            f'Voltage: {self.last_action:.3f}'
        ]

        for idx, text in enumerate(info):
            surface = self.font.render(text, True, self.c_black)
            self.screen.blit(surface, (50, 500 + idx * 25))

        self.draw_plot(
            rect   = self.plot_pos,
            data   = self.hist_pos,
            color  = (self.c_pos),
            y_min  = -self.beam/2,
            y_max  = +self.beam/2,
            title  = "Ball position and Setpoint (m)",
            data2  = self.hist_ref,
            color2 = self.c_ref,
            label  = "Position",
            label2 = "Setpoint",
        )

        self.draw_plot(
            rect  = self.plot_vol,
            data  = self.hist_vol,
            color = self.c_voltage,
            y_min = -12.0,
            y_max = +12.0,
            title = "Voltage (V)",
            label = "Voltage",
        )

        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
