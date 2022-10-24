# 作者：tomoya
# 创建：2022-09-30
# 更新：2022-09-30
# 用意：下坠的小鸟案例
import sys
import random
import time
import pygame
import os

if os.path.exists("flappy_bird_temp.ab"):
    os.remove("flappy_bird_temp.ab")
# FPS
FPS = 30
# 屏幕宽高
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
# 管道宽高
PIPE_WIDTH = 50
PIPE_HEIGHT = 300
# 管道之间空隙
PIPE_GAP_SIZE = 100
# 小鸟
BIRD_WIDTH = 20
BIRD_HEIGHT = 20
# 地面高度
FLOOR_HEIGHT = 80
# 游戏有效高度
BASE_HEIGHT = SCREEN_HEIGHT - FLOOR_HEIGHT


class Bird(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.rect = pygame.Rect(*position, BIRD_WIDTH, BIRD_HEIGHT)
        # 定义飞行变量
        self.is_flapped = False
        self.up_speed = 10
        self.down_speed = 0

        self.time_pass = FPS / 1000

    # 更新小鸟的位置
    def update(self):
        # 判断小鸟是上升还是下降
        if self.is_flapped:
            # 上升速度越来越小
            self.up_speed -= 60 * self.time_pass
            self.rect.top -= self.up_speed
            # 上升速度小于等于0, 改为下降状态
            if self.up_speed <= 0:
                self.down()
                self.up_speed = 10
                self.down_speed = 0
        else:
            # 下降速度越来越大
            self.down_speed += 30 * self.time_pass
            self.rect.bottom += self.down_speed

        # 判断小鸟是否撞到了边界死亡
        is_dead = False
        if self.rect.top <= 0:  # 上边界
            self.up_speed = 0
            self.rect.top = 0
            is_dead = True

        if self.rect.bottom >= BASE_HEIGHT:  # 下边界
            self.up_speed = 0
            self.down_speed = 0
            self.rect.bottom = BASE_HEIGHT
            is_dead = True

        return is_dead

    # 下落状态
    def down(self):
        self.is_flapped = False

    # 上升状态
    def up(self):
        if self.is_flapped:
            self.up_speed = max(12, self.up_speed + 1)
        else:
            self.is_flapped = True

    def draw(self, screen, color):
        pygame.draw.rect(screen, color, self.rect, 1)


class Pipe(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        left, top = position
        # 如果是下边的管道, 通过定义管道高度, 删除地面以下的管道
        pipe_height = PIPE_HEIGHT
        if top > 0:
            pipe_height = BASE_HEIGHT - top + 1
        self.rect = pygame.Rect(left, top, PIPE_WIDTH, pipe_height)
        # 用于计算分数
        self.used_for_score = False

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 1)

    @staticmethod
    def generate_pipe_position():
        # 生成上下两个管道的坐标
        random.seed(int(time.time()))
        top = int(BASE_HEIGHT * 0.2) + random.randrange(
            0, int(BASE_HEIGHT * 0.6 - PIPE_GAP_SIZE))
        return {
            'top': (SCREEN_WIDTH + 25, top - PIPE_HEIGHT),
            'bottom': (SCREEN_WIDTH + 25, top + PIPE_GAP_SIZE)
        }


# 初始化游戏
def init_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('you')
    return screen


# 初始化精灵
def init_sprite():
    # 小鸟类
    bird_position = [SCREEN_WIDTH * 0.2, (SCREEN_HEIGHT - BIRD_HEIGHT) / 3]
    bird = Bird(bird_position)
    # 管道类
    pipe_sprites = pygame.sprite.Group()
    for i in range(2):
        pipe_pos = Pipe.generate_pipe_position()
        # 添加上方的管道
        pipe_sprites.add(
            Pipe((SCREEN_WIDTH + i * SCREEN_WIDTH / 2,
                  pipe_pos.get('top')[-1])))
        # 添加下方的管道
        pipe_sprites.add(
            Pipe((SCREEN_WIDTH + i * SCREEN_WIDTH / 2,
                  pipe_pos.get('bottom')[-1])))
    return bird, pipe_sprites


# 精灵类碰撞检测和小鸟更新位置
def collision(bird, pipe_sprites):
    # 检测碰撞
    is_collision = False
    for pipe in pipe_sprites:
        if pygame.sprite.collide_rect(bird, pipe):
            is_collision = True

    # 更新小鸟
    is_dead = bird.update()
    if is_dead:
        is_collision = True

    return is_collision


# 移动pipe实现小鸟往前飞的效果
def move_pipe(bird, pipe_sprites, is_add_pipe, score):
    flag = False  # 下一次是否要增加新的pipe的标志位
    for pipe in pipe_sprites:
        pipe.rect.left -= 4
        # 小鸟飞过pipe 加分
        if pipe.rect.centerx < bird.rect.centerx and not pipe.used_for_score:
            pipe.used_for_score = True
            score += 0.5
        # 增加新的pipe
        if pipe.rect.left < 10 and pipe.rect.left > 0 and is_add_pipe:
            pipe_pos = Pipe.generate_pipe_position()
            pipe_sprites.add(Pipe(position=pipe_pos.get('top')))
            pipe_sprites.add(Pipe(position=pipe_pos.get('bottom')))
            is_add_pipe = False
        # 删除已不在屏幕的pipe, 更新标志位
        elif pipe.rect.right < 0:
            pipe_sprites.remove(pipe)
            flag = True
    if flag:
        is_add_pipe = True
    return is_add_pipe, score


# 画分数
def draw_score(screen, score):
    font_size = 32
    digits = len(str(int(score)))
    offset = (SCREEN_WIDTH - digits * font_size) / 2
    font = pygame.font.SysFont('Blod', font_size)
    screen.blit(font.render(str(int(score)), True, (255, 255, 255)),
                (offset, SCREEN_HEIGHT * 0.1))


# 画Game Over
def draw_game_over(screen, text):
    font_size = 24
    font = pygame.font.SysFont('arial', font_size)
    screen.blit(font.render(text, True, (255, 255, 255), (0, 0, 0)),
                (60, SCREEN_HEIGHT * 0.4))


# 按键
def press(is_game_running, bird, isRobot=False):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 点击关闭按钮退出
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if isRobot == False and event.key == pygame.K_SPACE or event.key == pygame.K_UP:  # 空格键或者up键小鸟上升
                if is_game_running:
                    bird.up()
            elif event.key == 13 and not is_game_running:  # 游戏结束时回车键继续
                return True


def up_or_down(bird, pipe_sprites):
    for index, pipe in enumerate(pipe_sprites):
        if index == 1:
            pipe1 = pipe
        elif index == 3:
            pipe2 = pipe
    if pipe1.rect[0] + 50 > bird.rect[0]:
        middle = pipe1.rect[1] - 50
    else:
        middle = pipe2.rect[1] - 50
    if bird.rect[1] + 10 > middle:
        bird.up()


def man():
    screen = init_game()  # 初始化游戏
    bird, pipe_sprites = init_sprite()  # 初始化精灵
    clock = pygame.time.Clock()
    is_add_pipe = True  # 是否需要增加管道
    is_game_running = True  # 是否在游戏中
    score = 0  # 初始分数
    while True:
        restart = press(is_game_running, bird)  # 按键
        if restart:
            return
        screen.fill((0, 0, 0))  # 填充背景
        is_collision = collision(bird, pipe_sprites)  # 碰撞检测
        if is_collision:
            is_game_running = False  # 如果碰撞 游戏结束
        if is_game_running:
            is_add_pipe, score = move_pipe(bird, pipe_sprites, is_add_pipe,
                                           score)  # 不碰撞 移动管道
        else:
            draw_game_over(screen, f'game over!! score:{int(score)}')  # 游戏结束

        bird.draw(screen, (0, 0, 255))  # 画鸟
        draw_score(screen, score)  # 画分数
        # 画地面
        pygame.draw.line(screen, (255, 255, 255), (0, BASE_HEIGHT),
                         (SCREEN_WIDTH, BASE_HEIGHT))
        # 画管道
        for pipe in pipe_sprites:
            pipe.draw(screen)

        # 更新画布
        pygame.display.update()
        clock.tick(FPS)
        if not is_game_running:
            time.sleep(2)
            open("flappy_bird_temp.ab", 'w').close()
            exit()


def robot():
    screen = init_game()  # 初始化游戏
    bird, pipe_sprites = init_sprite()  # 初始化精灵
    clock = pygame.time.Clock()
    is_add_pipe = True  # 是否需要增加管道
    is_game_running = True  # 是否在游戏中
    score = 0  # 初始分数
    while True:
        restart = press(is_game_running, bird, isRobot=True)  # 按键
        if restart:
            return
        screen.fill((0, 0, 0))  # 填充背景
        up_or_down(bird, pipe_sprites)
        is_collision = collision(bird, pipe_sprites)  # 碰撞检测
        if is_collision:
            is_game_running = False  # 如果碰撞 游戏结束
        if is_game_running:
            is_add_pipe, score = move_pipe(bird, pipe_sprites, is_add_pipe,
                                           score)  # 不碰撞 移动管道

        else:
            draw_game_over(screen, 'Robot Fail')  # 游戏结束
        bird.draw(screen, (255, 0, 0))  # 画鸟
        draw_score(screen, score)  # 画分数
        # 画地面
        pygame.draw.line(screen, (255, 255, 255), (0, BASE_HEIGHT),
                         (SCREEN_WIDTH, BASE_HEIGHT))
        # 画管道
        for pipe in pipe_sprites:
            pipe.draw(screen)

        # 更新画布
        pygame.display.update()
        clock.tick(FPS)
        if os.path.exists("flappy_bird_temp.ab"):
            exit()


if __name__ == "__main__":
    while True:
        robot()
