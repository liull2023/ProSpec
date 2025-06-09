import pygame
import numpy as np
from dm_control import suite

def main():
    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DMControl Cartpole Balance")
    clock = pygame.time.Clock()

    # 加载 Cartpole Balance 环境
    env = suite.load("finger", "spin")
    physics = env.physics

    # 打印一下 action_spec，确认 shape
    # print(env.action_spec())  

    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        # 读取键盘状态
        keys = pygame.key.get_pressed()
        # 左按键 -> -1，右按键 -> +1，否则 0
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            ctrl = -1.0
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            ctrl = +1.0
        else:
            ctrl = 0.0

        # ---- 这里是关键：直接生成一个 float32/float64 数组 ----
        # Cartpole 的 action_spec().shape == (1,)
        action = np.array([ctrl], dtype=np.float64)

        # 传给 env.step
        time_step = env.step(action)

        # 用 physics.render 拿到像素，注意 dm_control 输出 (H, W, 3)
        pixels = physics.render(height=height, width=width, camera_id=0)

        # pygame 的 surface 要 (W, H, 3)
        frame = pixels.transpose(1, 0, 2)
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
