import pygame
import math
import random
import json
import os
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
GRAVITY = 800
JUMP_FORCE = -350
PLAYER_SPEED = 300

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
PINK = (255, 192, 203)
DARK_BLUE = (0, 50, 100)
LIGHT_BLUE = (173, 216, 230)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
GRAY = (128, 128, 128)

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    GAME_OVER = 3
    VICTORY = 4
    PAUSED = 5

class ParticleType(Enum):
    SPARKLE = 1
    EXPLOSION = 2
    TRAIL = 3
    COLLECT = 4

@dataclass
class Vector2:
    x: float
    y: float
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

class Particle:
    def __init__(self, x, y, vx, vy, color, lifetime, size, particle_type):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.initial_size = size
        self.particle_type = particle_type
        self.gravity_affected = particle_type == ParticleType.EXPLOSION
    
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        if self.gravity_affected:
            self.vy += GRAVITY * dt * 0.3
        
        self.lifetime -= dt
        life_ratio = self.lifetime / self.max_lifetime
        self.size = self.initial_size * life_ratio
        
        return self.lifetime > 0
    
    def draw(self, screen, camera_x, camera_y):
        if self.size <= 0:
            return
        
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y - camera_y)
        
        if -50 <= screen_x <= SCREEN_WIDTH + 50 and -50 <= screen_y <= SCREEN_HEIGHT + 50:
            if self.particle_type == ParticleType.SPARKLE:
                # Draw a star shape
                points = []
                for i in range(8):
                    angle = i * math.pi / 4
                    radius = self.size if i % 2 == 0 else self.size * 0.5
                    px = screen_x + radius * math.cos(angle)
                    py = screen_y + radius * math.sin(angle)
                    points.append((px, py))
                if len(points) >= 3:
                    pygame.draw.polygon(screen, self.color, points)
            else:
                pygame.draw.circle(screen, self.color, (screen_x, screen_y), max(1, int(self.size)))

class ParticleSystem:
    def __init__(self):
        self.particles = []
    
    def add_sparkle(self, x, y, color=GOLD):
        for _ in range(5):
            vx = random.uniform(-50, 50)
            vy = random.uniform(-100, -50)
            lifetime = random.uniform(0.5, 1.0)
            size = random.uniform(3, 6)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, size, ParticleType.SPARKLE))
    
    def add_explosion(self, x, y, count=15):
        colors = [RED, ORANGE, YELLOW, WHITE]
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(100, 200)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = random.choice(colors)
            lifetime = random.uniform(0.8, 1.5)
            size = random.uniform(2, 5)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, size, ParticleType.EXPLOSION))
    
    def add_collect_effect(self, x, y):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 50  # Upward bias
            color = random.choice([GOLD, YELLOW, WHITE])
            lifetime = random.uniform(1.0, 1.5)
            size = random.uniform(4, 8)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, size, ParticleType.COLLECT))
    
    def add_trail(self, x, y, color):
        vx = random.uniform(-20, 20)
        vy = random.uniform(-20, 20)
        lifetime = 0.3
        size = random.uniform(2, 4)
        self.particles.append(Particle(x, y, vx, vy, color, lifetime, size, ParticleType.TRAIL))
    
    def update(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)]
    
    def draw(self, screen, camera_x, camera_y):
        for particle in self.particles:
            particle.draw(screen, camera_x, camera_y)

class Platform:
    def __init__(self, x, y, width, height, platform_type="normal"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.platform_type = platform_type
        self.bounce_force = 0
        self.animation_offset = 0
        
        if platform_type == "bouncy":
            self.bounce_force = -500
            self.color = PINK
        elif platform_type == "ice":
            self.color = LIGHT_BLUE
        elif platform_type == "lava":
            self.color = RED
        else:
            self.color = (100, 100, 100)
    
    def update(self, dt):
        if self.platform_type == "lava":
            self.animation_offset += dt * 5
    
    def draw(self, screen, camera_x, camera_y):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y - camera_y)
        
        # Draw platform with gradient effect
        if self.platform_type == "lava":
            # Animated lava effect
            for i in range(self.height):
                intensity = int(200 + 55 * math.sin(self.animation_offset + i * 0.1))
                color = (intensity, intensity // 4, 0)
                pygame.draw.rect(screen, color, (screen_x, screen_y + i, self.width, 1))
        else:
            pygame.draw.rect(screen, self.color, (screen_x, screen_y, self.width, self.height))
        
        # Draw border
        pygame.draw.rect(screen, WHITE, (screen_x, screen_y, self.width, self.height), 2)
        
        # Special effects
        if self.platform_type == "bouncy":
            # Draw bounce indicators
            for i in range(0, self.width, 20):
                pygame.draw.circle(screen, WHITE, (screen_x + i + 10, screen_y + 5), 3)
        elif self.platform_type == "ice":
            # Draw ice crystals
            for i in range(0, self.width, 15):
                pygame.draw.polygon(screen, WHITE, [
                    (screen_x + i + 7, screen_y + 2),
                    (screen_x + i + 5, screen_y + 8),
                    (screen_x + i + 9, screen_y + 8)
                ])

class Crystal:
    def __init__(self, x, y, crystal_type="normal"):
        self.x = x
        self.y = y
        self.crystal_type = crystal_type
        self.collected = False
        self.animation_offset = random.uniform(0, 2 * math.pi)
        self.bob_height = 10
        self.size = 15
        self.value = 10
        
        if crystal_type == "special":
            self.color = PURPLE
            self.value = 50
            self.size = 20
        else:
            self.color = CYAN
    
    def update(self, dt):
        self.animation_offset += dt * 3
    
    def draw(self, screen, camera_x, camera_y):
        if self.collected:
            return
        
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y - camera_y + math.sin(self.animation_offset) * self.bob_height)
        
        # Draw crystal with glow effect
        # Outer glow
        for radius in range(self.size + 10, self.size, -2):
            alpha = 50 - (radius - self.size) * 3
            glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            glow_color = (*self.color, max(0, alpha))
            pygame.draw.circle(glow_surface, glow_color, (radius, radius), radius)
            screen.blit(glow_surface, (screen_x - radius, screen_y - radius))
        
        # Crystal shape (diamond)
        points = [
            (screen_x, screen_y - self.size),
            (screen_x + self.size, screen_y),
            (screen_x, screen_y + self.size),
            (screen_x - self.size, screen_y)
        ]
        pygame.draw.polygon(screen, self.color, points)
        pygame.draw.polygon(screen, WHITE, points, 2)
        
        # Inner shine
        shine_points = [
            (screen_x - 3, screen_y - 3),
            (screen_x + 3, screen_y - 8),
            (screen_x - 1, screen_y - 1)
        ]
        pygame.draw.polygon(screen, WHITE, shine_points)

class Enemy:
    def __init__(self, x, y, enemy_type="walker"):
        self.x = x
        self.y = y
        self.start_x = x
        self.enemy_type = enemy_type
        self.vx = 50 if enemy_type == "walker" else 0
        self.vy = 0
        self.width = 20
        self.height = 20
        self.patrol_distance = 100
        self.animation_offset = 0
        self.alive = True
        
        if enemy_type == "jumper":
            self.color = GREEN
            self.jump_timer = 0
            self.jump_interval = 2.0
        elif enemy_type == "flyer":
            self.color = PURPLE
            self.fly_height = 50
        else:
            self.color = RED
    
    def update(self, dt, platforms):
        if not self.alive:
            return
        
        self.animation_offset += dt * 5
        
        if self.enemy_type == "walker":
            # Simple patrol behavior
            if abs(self.x - self.start_x) > self.patrol_distance:
                self.vx = -self.vx
            
            self.x += self.vx * dt
            
            # Apply gravity
            self.vy += GRAVITY * dt
            self.y += self.vy * dt
            
            # Platform collision
            for platform in platforms:
                if (self.x + self.width > platform.x and self.x < platform.x + platform.width and
                    self.y + self.height > platform.y and self.y < platform.y + platform.height):
                    if self.vy > 0:  # Falling
                        self.y = platform.y - self.height
                        self.vy = 0
        
        elif self.enemy_type == "jumper":
            self.jump_timer += dt
            if self.jump_timer >= self.jump_interval:
                self.vy = -250
                self.jump_timer = 0
            
            # Apply gravity
            self.vy += GRAVITY * dt
            self.y += self.vy * dt
            
            # Platform collision
            for platform in platforms:
                if (self.x + self.width > platform.x and self.x < platform.x + platform.width and
                    self.y + self.height > platform.y and self.y < platform.y + platform.height):
                    if self.vy > 0:  # Falling
                        self.y = platform.y - self.height
                        self.vy = 0
        
        elif self.enemy_type == "flyer":
            # Sine wave flight pattern
            self.y = self.start_x + math.sin(self.animation_offset) * self.fly_height
            self.x += 30 * dt
    
    def draw(self, screen, camera_x, camera_y):
        if not self.alive:
            return
        
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y - camera_y)
        
        if self.enemy_type == "flyer":
            # Draw wings
            wing_offset = math.sin(self.animation_offset * 3) * 5
            pygame.draw.ellipse(screen, self.color, (screen_x - 5, screen_y + wing_offset, 10, 8))
            pygame.draw.ellipse(screen, self.color, (screen_x + 15, screen_y + wing_offset, 10, 8))
        
        # Draw body
        pygame.draw.rect(screen, self.color, (screen_x, screen_y, self.width, self.height))
        pygame.draw.rect(screen, WHITE, (screen_x, screen_y, self.width, self.height), 2)
        
        # Draw eyes
        eye_y = screen_y + 5
        pygame.draw.circle(screen, WHITE, (screen_x + 5, eye_y), 2)
        pygame.draw.circle(screen, WHITE, (screen_x + 15, eye_y), 2)
        pygame.draw.circle(screen, BLACK, (screen_x + 5, eye_y), 1)
        pygame.draw.circle(screen, BLACK, (screen_x + 15, eye_y), 1)

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.width = 20
        self.height = 30
        self.on_ground = False
        self.health = 100
        self.max_health = 100
        self.score = 0
        self.crystals_collected = 0
        self.animation_frame = 0
        self.facing_right = True
        self.invulnerable_time = 0
        self.dash_cooldown = 0
        self.dash_speed = 600
        self.dash_duration = 0.2
        self.dashing = False
        self.dash_timer = 0
        self.double_jump_available = True
        self.wall_slide = False
        self.coyote_time = 0.1
        self.coyote_timer = 0
    
    def update(self, dt, keys_pressed, platforms):
        # Handle invulnerability
        if self.invulnerable_time > 0:
            self.invulnerable_time -= dt
        
        # Handle dash cooldown
        if self.dash_cooldown > 0:
            self.dash_cooldown -= dt
        
        # Handle dashing
        if self.dashing:
            self.dash_timer -= dt
            if self.dash_timer <= 0:
                self.dashing = False
                self.vx *= 0.5  # Reduce speed after dash
        
        # Coyote time (brief ground time after leaving platform)
        if self.on_ground:
            self.coyote_timer = self.coyote_time
        else:
            self.coyote_timer -= dt
        
        # Handle input
        if not self.dashing:
            # Horizontal movement
            if keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
                self.vx = -PLAYER_SPEED
                self.facing_right = False
                self.animation_frame += dt * 10
            elif keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
                self.vx = PLAYER_SPEED
                self.facing_right = True
                self.animation_frame += dt * 10
            else:
                self.vx *= 0.85  # Friction
                if abs(self.vx) < 10:
                    self.vx = 0
        
        # Dash
        if (keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_x]) and self.dash_cooldown <= 0:
            if not self.dashing:  # Only start dash if not already dashing
                self.dashing = True
                self.dash_timer = self.dash_duration
                self.dash_cooldown = 1.0  # 1 second cooldown
                direction = 1 if self.facing_right else -1
                self.vx = self.dash_speed * direction
                self.vy = 0  # Stop vertical movement during dash
        
        # Jump
        if keys_pressed[pygame.K_SPACE] or keys_pressed[pygame.K_UP] or keys_pressed[pygame.K_w]:
            if self.coyote_timer > 0:  # Regular jump
                self.vy = JUMP_FORCE
                self.on_ground = False
                self.coyote_timer = 0
                self.double_jump_available = True
            elif self.double_jump_available and not self.on_ground:  # Double jump
                self.vy = JUMP_FORCE * 0.8
                self.double_jump_available = False
        
        # Apply gravity
        if not self.dashing:
            self.vy += GRAVITY * dt
        
        # Terminal velocity
        if self.vy > 600:
            self.vy = 600
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Platform collision
        self.on_ground = False
        for platform in platforms:
            if (self.x + self.width > platform.x and self.x < platform.x + platform.width and
                self.y + self.height > platform.y and self.y < platform.y + platform.height):
                
                # Top collision (landing on platform)
                if self.vy > 0 and self.y < platform.y:
                    self.y = platform.y - self.height
                    self.vy = 0
                    self.on_ground = True
                    self.double_jump_available = True
                    
                    # Special platform effects
                    if platform.platform_type == "bouncy":
                        self.vy = platform.bounce_force
                        self.on_ground = False
                    elif platform.platform_type == "lava":
                        self.take_damage(20)
                
                # Bottom collision (hitting head)
                elif self.vy < 0 and self.y > platform.y:
                    self.y = platform.y + platform.height
                    self.vy = 0
                
                # Side collisions
                elif self.vx > 0:  # Moving right
                    self.x = platform.x - self.width
                    if platform.platform_type == "ice":
                        self.vx *= -0.3  # Bounce off ice
                elif self.vx < 0:  # Moving left
                    self.x = platform.x + platform.width
                    if platform.platform_type == "ice":
                        self.vx *= -0.3  # Bounce off ice
        
        # Keep player in world bounds (basic implementation)
        if self.x < 0:
            self.x = 0
            self.vx = 0
        if self.y > 1000:  # Fell off the world
            self.take_damage(100)
    
    def take_damage(self, damage):
        if self.invulnerable_time <= 0:
            self.health -= damage
            self.invulnerable_time = 1.0
            return True
        return False
    
    def collect_crystal(self, crystal):
        self.score += crystal.value
        self.crystals_collected += 1
        crystal.collected = True
    
    def draw(self, screen, camera_x, camera_y, particle_system):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y - camera_y)
        
        # Add trail particles when dashing
        if self.dashing:
            particle_system.add_trail(self.x + self.width // 2, self.y + self.height // 2, CYAN)
        
        # Flash when invulnerable
        if self.invulnerable_time > 0 and int(self.invulnerable_time * 10) % 2 == 0:
            return
        
        # Player body
        color = CYAN if self.dashing else BLUE
        pygame.draw.rect(screen, color, (screen_x, screen_y, self.width, self.height))
        pygame.draw.rect(screen, WHITE, (screen_x, screen_y, self.width, self.height), 2)
        
        # Eyes
        eye_y = screen_y + 8
        if self.facing_right:
            pygame.draw.circle(screen, WHITE, (screen_x + 14, eye_y), 2)
            pygame.draw.circle(screen, BLACK, (screen_x + 15, eye_y), 1)
        else:
            pygame.draw.circle(screen, WHITE, (screen_x + 6, eye_y), 2)
            pygame.draw.circle(screen, BLACK, (screen_x + 5, eye_y), 1)
        
        # Movement animation (simple leg movement)
        if abs(self.vx) > 10 and self.on_ground:
            leg_offset = int(math.sin(self.animation_frame) * 3)
            pygame.draw.rect(screen, color, (screen_x + 2, screen_y + self.height, 4, 5 + leg_offset))
            pygame.draw.rect(screen, color, (screen_x + self.width - 6, screen_y + self.height, 4, 5 - leg_offset))

class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.target_x = 0
        self.target_y = 0
    
    def update(self, dt, player):
        # Smooth camera following
        self.target_x = player.x - SCREEN_WIDTH // 2
        self.target_y = player.y - SCREEN_HEIGHT // 2
        
        # Smooth interpolation
        self.x += (self.target_x - self.x) * dt * 5
        self.y += (self.target_y - self.y) * dt * 5
        
        # Keep camera within reasonable bounds
        self.y = max(self.y, -200)  # Don't go too high

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Crystal Runner - Advanced 2D Platformer")
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = GameState.MENU
        self.menu_selection = 0
        
        # Initialize game objects
        self.player = Player(100, 300)
        self.camera = Camera()
        self.particle_system = ParticleSystem()
        
        # Initialize fonts
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 48)
        self.font_huge = pygame.font.Font(None, 72)
        
        # Create level
        self.create_level()
        
        # Menu options
        self.menu_options = ["Start Game", "Instructions", "Quit"]
        
        # Background gradient
        self.create_background()
    
    def create_background(self):
        self.background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        for y in range(SCREEN_HEIGHT):
            ratio = y / SCREEN_HEIGHT
            r = int(20 + ratio * 30)
            g = int(30 + ratio * 70)
            b = int(60 + ratio * 140)
            color = (r, g, b)
            pygame.draw.line(self.background, color, (0, y), (SCREEN_WIDTH, y))
    
    def create_level(self):
        self.platforms = []
        self.crystals = []
        self.enemies = []
        
        # Create platforms
        platform_data = [
            # (x, y, width, height, type)
            (0, 400, 200, 20, "normal"),
            (300, 350, 150, 20, "normal"),
            (500, 300, 100, 20, "bouncy"),
            (700, 250, 150, 20, "normal"),
            (900, 200, 100, 20, "ice"),
            (1100, 150, 200, 20, "normal"),
            (1400, 300, 150, 20, "lava"),
            (1600, 250, 100, 20, "bouncy"),
            (1800, 200, 200, 20, "normal"),
            (2100, 150, 150, 20, "ice"),
            (2350, 100, 100, 20, "normal"),
            (2550, 50, 200, 20, "normal"),
            # Ground platforms
            (0, 500, 300, 100, "normal"),
            (400, 600, 400, 100, "normal"),
            (900, 550, 300, 100, "normal"),
            (1300, 650, 500, 100, "normal"),
            (1900, 600, 400, 100, "normal"),
            (2400, 500, 400, 100, "normal"),
        ]
        
        for x, y, w, h, ptype in platform_data:
            self.platforms.append(Platform(x, y, w, h, ptype))
        
        # Create crystals
        crystal_positions = [
            (350, 300), (550, 250), (750, 200), (950, 150),
            (1150, 100), (1450, 250), (1650, 200), (1850, 150),
            (2150, 100), (2400, 50), (2600, 0)
        ]
        
        for i, (x, y) in enumerate(crystal_positions):
            crystal_type = "special" if i % 4 == 0 else "normal"
            self.crystals.append(Crystal(x, y, crystal_type))
        
        # Create enemies
        enemy_data = [
            (450, 560, "walker"),
            (1000, 510, "jumper"),
            (1500, 610, "walker"),
            (2000, 560, "jumper"),
            (800, 150, "flyer"),
            (1700, 100, "flyer"),
        ]
        
        for x, y, etype in enemy_data:
            self.enemies.append(Enemy(x, y, etype))
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.state == GameState.MENU:
                    self.handle_menu_input(event.key)
                elif self.state == GameState.PLAYING:
                    if event.key == pygame.K_ESCAPE:
                        self.state = GameState.PAUSED
                elif self.state == GameState.GAME_OVER or self.state == GameState.VICTORY:
                    if event.key == pygame.K_SPACE:
                        self.restart_game()
                elif self.state == GameState.PAUSED:
                    if event.key == pygame.K_ESCAPE:
                        self.state = GameState.PLAYING
    
    def handle_menu_input(self, key):
        if key == pygame.K_UP:
            self.menu_selection = (self.menu_selection - 1) % len(self.menu_options)
        elif key == pygame.K_DOWN:
            self.menu_selection = (self.menu_selection + 1) % len(self.menu_options)
        elif key == pygame.K_RETURN or key == pygame.K_SPACE:
            if self.menu_selection == 0:  # Start Game
                self.state = GameState.PLAYING
            elif self.menu_selection == 1:  # Instructions
                pass  # Could add instructions screen
            elif self.menu_selection == 2:  # Quit
                self.running = False
    
    def update_game(self, dt):
        keys_pressed = pygame.key.get_pressed()
        
        # Update player
        self.player.update(dt, keys_pressed, self.platforms)
        
        # Update camera
        self.camera.update(dt, self.player)
        
        # Update platforms
        for platform in self.platforms:
            platform.update(dt)
        
        # Update crystals
        for crystal in self.crystals:
            crystal.update(dt)
        
        # Update enemies
        for enemy in self.enemies:
            enemy.update(dt, self.platforms)
        
        # Update particle system
        self.particle_system.update(dt)
        
        # Check collisions
        self.check_collisions()
        
        # Check win condition
        if all(crystal.collected for crystal in self.crystals):
            self.state = GameState.VICTORY
        
        # Check game over
        if self.player.health <= 0:
            self.state = GameState.GAME_OVER
    
    def check_collisions(self):
        # Player vs Crystals
        for crystal in self.crystals:
            if not crystal.collected:
                if (self.player.x < crystal.x + 15 and self.player.x + self.player.width > crystal.x - 15 and
                    self.player.y < crystal.y + 15 and self.player.y + self.player.height > crystal.y - 15):
                    self.player.collect_crystal(crystal)
                    self.particle_system.add_collect_effect(crystal.x, crystal.y)
        
        # Player vs Enemies
        for enemy in self.enemies:
            if enemy.alive:
                if (self.player.x < enemy.x + enemy.width and 
                    self.player.x + self.player.width > enemy.x and
                    self.player.y < enemy.y + enemy.height and 
                    self.player.y + self.player.height > enemy.y):
                    
                    if self.player.dashing:
                        # Player destroys enemy while dashing
                        enemy.alive = False
                        self.particle_system.add_explosion(enemy.x + enemy.width // 2, 
                                                         enemy.y + enemy.height // 2)
                        self.player.score += 25
                    else:
                        # Player takes damage
                        if self.player.take_damage(20):
                            self.particle_system.add_explosion(self.player.x + self.player.width // 2,
                                                             self.player.y + self.player.height // 2, 8)
    
    def restart_game(self):
        self.player = Player(100, 300)
        self.camera = Camera()
        self.particle_system = ParticleSystem()
        self.create_level()
        self.state = GameState.PLAYING
    
    def draw_menu(self):
        # Draw animated background
        self.screen.blit(self.background, (0, 0))
        
        # Add floating particles for menu effect
        if random.random() < 0.1:
            x = random.randint(0, SCREEN_WIDTH)
            y = SCREEN_HEIGHT
            self.particle_system.add_sparkle(x, y, random.choice([GOLD, CYAN, PURPLE]))
        
        self.particle_system.update(1/60)
        self.particle_system.draw(self.screen, 0, 0)
        
        # Draw title with glow effect
        title_text = "CRYSTAL RUNNER"
        for offset in range(5, 0, -1):
            title_surface = self.font_huge.render(title_text, True, (0, 100, 200, 100))
            title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2 + offset, 150 + offset))
            self.screen.blit(title_surface, title_rect)
        
        title_surface = self.font_huge.render(title_text, True, WHITE)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, 150))
        self.screen.blit(title_surface, title_rect)
        
        # Draw subtitle
        subtitle = "Advanced 2D Platformer Adventure"
        subtitle_surface = self.font_medium.render(subtitle, True, GOLD)
        subtitle_rect = subtitle_surface.get_rect(center=(SCREEN_WIDTH // 2, 200))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Draw menu options
        start_y = 300
        for i, option in enumerate(self.menu_options):
            color = YELLOW if i == self.menu_selection else WHITE
            
            # Add glow effect to selected option
            if i == self.menu_selection:
                for offset in range(3, 0, -1):
                    glow_surface = self.font_large.render(option, True, (255, 255, 0, 50))
                    glow_rect = glow_surface.get_rect(center=(SCREEN_WIDTH // 2 + offset, start_y + i * 60 + offset))
                    self.screen.blit(glow_surface, glow_rect)
            
            option_surface = self.font_large.render(option, True, color)
            option_rect = option_surface.get_rect(center=(SCREEN_WIDTH // 2, start_y + i * 60))
            self.screen.blit(option_surface, option_rect)
            
            # Selection indicator
            if i == self.menu_selection:
                pygame.draw.rect(self.screen, YELLOW, option_rect.inflate(20, 10), 3)
        
        # Draw controls hint
        controls_text = "Use ARROW KEYS/WASD to move, SPACE to jump, SHIFT/X to dash"
        controls_surface = self.font_small.render(controls_text, True, GRAY)
        controls_rect = controls_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        self.screen.blit(controls_surface, controls_rect)
    
    def draw_hud(self):
        # Health bar
        bar_width = 200
        bar_height = 20
        bar_x = 20
        bar_y = 20
        
        # Background
        pygame.draw.rect(self.screen, DARK_BLUE, (bar_x, bar_y, bar_width, bar_height))
        # Health fill
        health_ratio = max(0, self.player.health / self.player.max_health)
        health_width = int(bar_width * health_ratio)
        health_color = GREEN if health_ratio > 0.5 else (ORANGE if health_ratio > 0.25 else RED)
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, health_width, bar_height))
        # Border
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Health text
        health_text = f"Health: {int(self.player.health)}/{self.player.max_health}"
        health_surface = self.font_small.render(health_text, True, WHITE)
        self.screen.blit(health_surface, (bar_x + 5, bar_y + 2))
        
        # Score
        score_text = f"Score: {self.player.score}"
        score_surface = self.font_medium.render(score_text, True, WHITE)
        self.screen.blit(score_surface, (SCREEN_WIDTH - 200, 20))
        
        # Crystals collected
        crystals_text = f"Crystals: {self.player.crystals_collected}/{len(self.crystals)}"
        crystals_surface = self.font_medium.render(crystals_text, True, CYAN)
        self.screen.blit(crystals_surface, (SCREEN_WIDTH - 200, 50))
        
        # Dash cooldown indicator
        if self.player.dash_cooldown > 0:
            cooldown_text = f"Dash: {self.player.dash_cooldown:.1f}s"
            cooldown_surface = self.font_small.render(cooldown_text, True, YELLOW)
            self.screen.blit(cooldown_surface, (20, 50))
        else:
            ready_text = "Dash: READY"
            ready_surface = self.font_small.render(ready_text, True, GREEN)
            self.screen.blit(ready_surface, (20, 50))
        
        # Instructions
        if self.player.crystals_collected < 2:  # Show hints for new players
            hint_text = "Collect all crystals to win! Use SHIFT to dash through enemies!"
            hint_surface = self.font_small.render(hint_text, True, GOLD)
            hint_rect = hint_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
            self.screen.blit(hint_surface, hint_rect)
    
    def draw_game_over(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game Over title
        title = self.font_huge.render("GAME OVER", True, RED)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(title, title_rect)
        
        # Final score
        score_text = f"Final Score: {self.player.score}"
        score_surface = self.font_large.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 350))
        self.screen.blit(score_surface, score_rect)
        
        # Crystals collected
        crystals_text = f"Crystals Collected: {self.player.crystals_collected}/{len(self.crystals)}"
        crystals_surface = self.font_medium.render(crystals_text, True, CYAN)
        crystals_rect = crystals_surface.get_rect(center=(SCREEN_WIDTH // 2, 400))
        self.screen.blit(crystals_surface, crystals_rect)
        
        # Restart instruction
        restart_text = "Press SPACE to restart"
        restart_surface = self.font_medium.render(restart_text, True, YELLOW)
        restart_rect = restart_surface.get_rect(center=(SCREEN_WIDTH // 2, 500))
        self.screen.blit(restart_surface, restart_rect)
    
    def draw_victory(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 50))
        self.screen.blit(overlay, (0, 0))
        
        # Victory particles
        for _ in range(3):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT // 2)
            self.particle_system.add_sparkle(x, y, random.choice([GOLD, YELLOW, WHITE]))
        
        # Victory title with rainbow effect
        title_text = "VICTORY!"
        colors = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, PURPLE]
        for i, char in enumerate(title_text):
            color = colors[i % len(colors)]
            char_surface = self.font_huge.render(char, True, color)
            char_x = SCREEN_WIDTH // 2 - len(title_text) * 20 + i * 40
            self.screen.blit(char_surface, (char_x, 200))
        
        # Congratulations text
        congrats_text = "You collected all the crystals!"
        congrats_surface = self.font_large.render(congrats_text, True, WHITE)
        congrats_rect = congrats_surface.get_rect(center=(SCREEN_WIDTH // 2, 300))
        self.screen.blit(congrats_surface, congrats_rect)
        
        # Final score
        score_text = f"Final Score: {self.player.score}"
        score_surface = self.font_medium.render(score_text, True, GOLD)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 400))
        self.screen.blit(score_surface, score_rect)
        
        # Play again instruction
        again_text = "Press SPACE to play again"
        again_surface = self.font_medium.render(again_text, True, YELLOW)
        again_rect = again_surface.get_rect(center=(SCREEN_WIDTH // 2, 500))
        self.screen.blit(again_surface, again_rect)
    
    def draw_pause_screen(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Paused text
        paused_text = "PAUSED"
        paused_surface = self.font_huge.render(paused_text, True, WHITE)
        paused_rect = paused_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(paused_surface, paused_rect)
        
        # Instructions
        instruction_text = "Press ESC to resume"
        instruction_surface = self.font_medium.render(instruction_text, True, YELLOW)
        instruction_rect = instruction_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
        self.screen.blit(instruction_surface, instruction_rect)
    
    def draw_game(self):
        # Draw animated background
        self.screen.blit(self.background, (0, 0))
        
        # Add some atmospheric particles
        if random.random() < 0.02:
            x = random.randint(int(self.camera.x), int(self.camera.x + SCREEN_WIDTH))
            y = int(self.camera.y - 50)
            self.particle_system.add_sparkle(x, y, (100, 150, 255))
        
        # Draw platforms
        for platform in self.platforms:
            platform.draw(self.screen, self.camera.x, self.camera.y)
        
        # Draw crystals
        for crystal in self.crystals:
            crystal.draw(self.screen, self.camera.x, self.camera.y)
        
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen, self.camera.x, self.camera.y)
        
        # Draw particle system
        self.particle_system.draw(self.screen, self.camera.x, self.camera.y)
        
        # Draw player
        self.player.draw(self.screen, self.camera.x, self.camera.y, self.particle_system)
        
        # Draw HUD
        self.draw_hud()
    
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Convert to seconds
            
            self.handle_events()
            
            if self.state == GameState.PLAYING:
                self.update_game(dt)
                self.draw_game()
            elif self.state == GameState.MENU:
                self.draw_menu()
            elif self.state == GameState.GAME_OVER:
                self.draw_game()  # Show game state behind overlay
                self.draw_game_over()
            elif self.state == GameState.VICTORY:
                self.draw_game()  # Show game state behind overlay
                self.draw_victory()
            elif self.state == GameState.PAUSED:
                self.draw_game()  # Show game state behind overlay
                self.draw_pause_screen()
            
            pygame.display.flip()
        
        pygame.quit()

# Main execution
if __name__ == "__main__":
    game = Game()
    game.run()