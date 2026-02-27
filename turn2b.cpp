#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <iostream>
#include <list>

// Simulation parameters
constexpr int GRID_WIDTH = 400;
constexpr int GRID_HEIGHT = 300;
constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr float PIXEL_SCALE = 2.0f;  // Scale grid to window

// Physics parameters
constexpr float DT = 0.016f;  // Time step (approx 60fps)
constexpr float DX = 0.01f;   // Spatial step
constexpr float ALPHA = 0.0001f;  // Thermal diffusivity (adjusted for stability)
constexpr float AMBIENT_TEMP = 25.0f;
constexpr float MAX_TEMP = 800.0f;
constexpr float CURSOR_HEAT_RADIUS = 40.0f;
constexpr float CURSOR_HEAT_POWER = 800.0f;
constexpr float INITIAL_OBJECT_TEMP = 200.0f;
constexpr float PROXIMITY_HEAT_MAX_DIST = 150.0f;  // Max distance for proximity heating
constexpr float CYLINDER_COOLING_RATE = 0.005f; // Reduced cooling rate for cylinder

// Precomputed factors for heat equation
constexpr float FACTOR_AIR = (ALPHA * 0.5f) * DT / (DX * DX);
constexpr float FACTOR_METAL = (ALPHA * 2.0f) * DT / (DX * DX);

// Cylinder parameters
constexpr float CYLINDER_CENTER_X = GRID_WIDTH * 0.5f;
constexpr float CYLINDER_CENTER_Y = GRID_HEIGHT * 0.5f;
constexpr float CYLINDER_RADIUS_X = 60.0f;  // Horizontal radius
constexpr float CYLINDER_RADIUS_Y = 40.0f;  // Vertical radius

// Stability condition: alpha * dt / dx^2 < 0.25 for 2D
constexpr float STABILITY_FACTOR = ALPHA * DT / (DX * DX);

struct Particle {
    sf::Vector2f position;
    sf::Vector2f velocity;
    float life; // 0.0 to 1.0
    float initial_life;
    float size;
};

class HeatSimulation {
private:
    std::vector<float> temp_current;
    std::vector<float> temp_next;
    std::vector<float> thermal_conductivity;  // Varies by material
    std::vector<bool> is_cylinder;
    std::vector<int> cylinder_indices; // Cached indices of cylinder cells
    std::vector<sf::Uint8> pixels;
    sf::Texture texture;
    sf::Sprite sprite;
    
    // Random generator for flame animation
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    float flame_time = 0.0f;

    // Particle system
    std::list<Particle> particles;
    
public:
    HeatSimulation() 
        : temp_current(GRID_WIDTH * GRID_HEIGHT, AMBIENT_TEMP),
          temp_next(GRID_WIDTH * GRID_HEIGHT, AMBIENT_TEMP),
          thermal_conductivity(GRID_WIDTH * GRID_HEIGHT, ALPHA),
          is_cylinder(GRID_WIDTH * GRID_HEIGHT, false),
          pixels(GRID_WIDTH * GRID_HEIGHT * 4, 0),
          rng(std::random_device{}()),
          dist(0.0f, 1.0f) {
        
        // Validate stability condition at runtime
        validateStability();
        
        // Initialize texture and sprite
        texture.create(GRID_WIDTH, GRID_HEIGHT);
        sprite.setTexture(texture);
        sprite.setScale(PIXEL_SCALE, PIXEL_SCALE);
        
        // Initialize cylinder shape
        initializeCylinder();
        
        // Set initial temperatures
        initializeTemperatures();
    }
    
    void validateStability() {
        std::cout << "Stability Factor: " << STABILITY_FACTOR << " (must be < 0.25)" << std::endl;
        if (STABILITY_FACTOR >= 0.25f) {
            std::cerr << "WARNING: Stability condition violated! STABILITY_FACTOR = " 
                      << STABILITY_FACTOR << " >= 0.25" << std::endl;
            std::cerr << "Simulation may become unstable. Reduce DT or ALPHA, or increase DX." << std::endl;
        } else {
            std::cout << "Stability check passed." << std::endl;
        }
    }
    
    void initializeCylinder() {
        cylinder_indices.clear();
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                float dx = (x - CYLINDER_CENTER_X) / CYLINDER_RADIUS_X;
                float dy = (y - CYLINDER_CENTER_Y) / CYLINDER_RADIUS_Y;
                float dist_sq = dx * dx + dy * dy;
                
                int idx = y * GRID_WIDTH + x;
                if (dist_sq <= 1.0f) {
                    is_cylinder[idx] = true;
                    thermal_conductivity[idx] = ALPHA * 2.0f;  // Metal conducts better
                    temp_current[idx] = INITIAL_OBJECT_TEMP;
                    cylinder_indices.push_back(idx);
                } else {
                    is_cylinder[idx] = false;
                    thermal_conductivity[idx] = ALPHA * 0.5f;  // Air conducts poorly
                    temp_current[idx] = AMBIENT_TEMP;
                }
            }
        }
    }
    
    void initializeTemperatures() {
        // Add some random variation
        std::uniform_real_distribution<float> temp_var(-5.0f, 5.0f);
        for (size_t i = 0; i < temp_current.size(); ++i) {
            if (!is_cylinder[i]) {
                temp_current[i] = AMBIENT_TEMP + temp_var(rng);
            }
        }
    }
    
    inline int index(int x, int y) const {
        return y * GRID_WIDTH + x;
    }
    
    void applyProximityHeat(int mouse_x, int mouse_y) {
        // Convert mouse position to grid coordinates
        float grid_x = mouse_x / PIXEL_SCALE;
        float grid_y = mouse_y / PIXEL_SCALE;
        
        // Calculate distance from cursor to cylinder center
        float dx = grid_x - CYLINDER_CENTER_X;
        float dy = grid_y - CYLINDER_CENTER_Y;
        float dist_to_cylinder = std::sqrt(dx * dx + dy * dy);
        
        // Only apply proximity heat if cursor is within range
        if (dist_to_cylinder > PROXIMITY_HEAT_MAX_DIST) return;
        
        // Heat intensity scales inversely with distance (1/r relationship)
        // Closer = more intense heating
        float proximity_factor = 1.0f - (dist_to_cylinder / PROXIMITY_HEAT_MAX_DIST);
        proximity_factor = proximity_factor * proximity_factor;  // Square for smoother falloff
        
        float proximity_heat_power = CURSOR_HEAT_POWER * 0.5f;  // Less intense than direct heating
        
        // Apply heat to the cylinder based on proximity
        // Optimized: Only iterate over cached cylinder indices
        float heat_radius = CYLINDER_RADIUS_X * 2.0f;
        float heat_radius_sq = heat_radius * heat_radius;
        float heat_radius_factor = 1.0f / (2.0f * heat_radius_sq * 0.25f);

        for (int idx : cylinder_indices) {
            int x = idx % GRID_WIDTH;
            int y = idx / GRID_WIDTH;
            
            // Simple approach: heat based on distance from cursor to this cell
            float cell_dx = x - grid_x;
            float cell_dy = y - grid_y;
            float dist_sq = cell_dx * cell_dx + cell_dy * cell_dy;
            
            if (dist_sq > heat_radius_sq) continue;
            
            // Apply heat with Gaussian falloff from cursor position
            float heat_factor = std::exp(-dist_sq * heat_radius_factor);
            
            // Scale by proximity factor (closer cursor = more heat)
            float heat_addition = proximity_heat_power * proximity_factor * heat_factor * DT;
            
            temp_current[idx] += heat_addition;
            temp_current[idx] = std::min(temp_current[idx], MAX_TEMP);
        }
    }
    
    void solveHeatEquation() {
        // Explicit finite difference method
        // T_new = T_old + factor * (T_left + T_right + T_up + T_down - 4*T_old)
        // Using precomputed factors for air and metal
        
        for (int y = 1; y < GRID_HEIGHT - 1; ++y) {
            for (int x = 1; x < GRID_WIDTH - 1; ++x) {
                int idx = index(x, y);
                
                float T_center = temp_current[idx];
                float T_left = temp_current[index(x - 1, y)];
                float T_right = temp_current[index(x + 1, y)];
                float T_up = temp_current[index(x, y - 1)];
                float T_down = temp_current[index(x, y + 1)];
                
                // Select precomputed factor based on material
                float factor = is_cylinder[idx] ? FACTOR_METAL : FACTOR_AIR;
                
                // Laplacian operator
                float laplacian = T_left + T_right + T_up + T_down - 4.0f * T_center;
                
                temp_next[idx] = T_center + factor * laplacian;
                
                // Cooling to ambient
                if (!is_cylinder[idx]) {
                    // Air cooling (Newton's law)
                    float cooling_rate = 0.001f;
                    temp_next[idx] += cooling_rate * (AMBIENT_TEMP - temp_next[idx]) * DT;
                } else {
                    // Cylinder cooling (Radiation/Convection approximation)
                    // Metal radiates heat effectively
                    temp_next[idx] += CYLINDER_COOLING_RATE * (AMBIENT_TEMP - temp_next[idx]) * DT;
                }
            }
        }
        
        // Boundary conditions (Dirichlet: fixed ambient at edges)
        for (int x = 0; x < GRID_WIDTH; ++x) {
            temp_next[index(x, 0)] = AMBIENT_TEMP;
            temp_next[index(x, GRID_HEIGHT - 1)] = AMBIENT_TEMP;
        }
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            temp_next[index(0, y)] = AMBIENT_TEMP;
            temp_next[index(GRID_WIDTH - 1, y)] = AMBIENT_TEMP;
        }
        
        // Swap buffers
        std::swap(temp_current, temp_next);
    }
    
    sf::Color temperatureToColor(float temp) {
        // Map temperature to color: deep blue (cold) -> cyan -> green -> yellow -> red (hot)
        // temp range: AMBIENT_TEMP (25) to MAX_TEMP (800)
        
        float t = (temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP);
        t = std::clamp(t, 0.0f, 1.0f);
        
        sf::Uint8 r, g, b;
        
        if (t < 0.25f) {
            // Blue to Cyan
            float local_t = t / 0.25f;
            r = 0;
            g = static_cast<sf::Uint8>(local_t * 255);
            b = 255;
        } else if (t < 0.5f) {
            // Cyan to Green
            float local_t = (t - 0.25f) / 0.25f;
            r = 0;
            g = 255;
            b = static_cast<sf::Uint8>((1.0f - local_t) * 255);
        } else if (t < 0.75f) {
            // Green to Yellow
            float local_t = (t - 0.5f) / 0.25f;
            r = static_cast<sf::Uint8>(local_t * 255);
            g = 255;
            b = 0;
        } else {
            // Yellow to Red
            float local_t = (t - 0.75f) / 0.25f;
            r = 255;
            g = static_cast<sf::Uint8>((1.0f - local_t) * 255);
            b = 0;
        }
        
        return sf::Color(r, g, b);
    }
    
    void updatePixels() {
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                int idx = index(x, y);
                sf::Color color = temperatureToColor(temp_current[idx]);
                
                int pixel_idx = idx * 4;
                pixels[pixel_idx + 0] = color.r;
                pixels[pixel_idx + 1] = color.g;
                pixels[pixel_idx + 2] = color.b;
                pixels[pixel_idx + 3] = 255;  // Alpha
            }
        }
        
        texture.update(pixels.data());
    }
    
    void updateParticles(float dt, float mouse_x, float mouse_y) {
        // Add new particles
        int new_particles = 2; // Add a few particles per frame
        for (int i = 0; i < new_particles; ++i) {
            if (particles.size() < 100) {
                Particle p;
                p.position = sf::Vector2f(mouse_x, mouse_y);
                
                // Random velocity: slight horizontal, fast vertical
                std::uniform_real_distribution<float> vx_dist(-10.0f, 10.0f);
                std::uniform_real_distribution<float> vy_dist(-80.0f, -40.0f);
                std::uniform_real_distribution<float> life_dist(0.5f, 1.0f);
                std::uniform_real_distribution<float> size_dist(5.0f, 10.0f);
                
                p.velocity = sf::Vector2f(vx_dist(rng), vy_dist(rng));
                p.life = life_dist(rng);
                p.initial_life = p.life;
                p.size = size_dist(rng);
                
                particles.push_back(p);
            }
        }
        
        // Update existing particles
        auto it = particles.begin();
        while (it != particles.end()) {
            it->life -= dt;
            
            if (it->life <= 0.0f) {
                it = particles.erase(it);
            } else {
                it->position += it->velocity * dt;
                // Add some upward acceleration (buoyancy)
                it->velocity.y -= 20.0f * dt;
                // Shrink over time
                it->size = it->size * 0.99f;
                ++it;
            }
        }
    }
    
    void update(float dt, int& mouse_x, int& mouse_y) {
        flame_time += dt;
        
        // Collision detection: prevent cursor from entering cylinder
        float grid_x = mouse_x / PIXEL_SCALE;
        float grid_y = mouse_y / PIXEL_SCALE;
        float dx = grid_x - CYLINDER_CENTER_X;
        float dy = grid_y - CYLINDER_CENTER_Y;
        float dist_sq = dx * dx + dy * dy;
        float min_dist = CYLINDER_RADIUS_X + 2.0f; // Cylinder radius + buffer
        
        if (dist_sq < min_dist * min_dist) {
            float dist = std::sqrt(dist_sq);
            if (dist > 0.001f) {
                // Push cursor out
                float push_x = (dx / dist) * min_dist;
                float push_y = (dy / dist) * min_dist;
                
                mouse_x = static_cast<int>((CYLINDER_CENTER_X + push_x) * PIXEL_SCALE);
                mouse_y = static_cast<int>((CYLINDER_CENTER_Y + push_y) * PIXEL_SCALE);
                
                // Re-position mouse in window
                sf::Mouse::setPosition(sf::Vector2i(mouse_x, mouse_y));
            }
        }

        // Apply proximity-based heat from cursor (always active based on distance)
        applyProximityHeat(mouse_x, mouse_y);
        
        // Solve heat equation (multiple iterations for stability/speed)
        for (int i = 0; i < 2; ++i) {
            solveHeatEquation();
        }
        
        // Update particles
        updateParticles(dt, static_cast<float>(mouse_x), static_cast<float>(mouse_y));
        
        updatePixels();
    }
    
    void draw(sf::RenderWindow& window) {
        window.draw(sprite);
    }
    
    void drawCursor(sf::RenderWindow& window, int mouse_x, int mouse_y) {
        // Draw particles
        for (const auto& p : particles) {
            float life_ratio = p.life / p.initial_life;
            
            sf::CircleShape shape(p.size * life_ratio);
            shape.setOrigin(shape.getRadius(), shape.getRadius());
            shape.setPosition(p.position);
            
            // Color mapping based on life
            // White -> Yellow -> Orange -> Red -> Transparent
            sf::Color color;
            if (life_ratio > 0.8f) {
                // White to Yellow
                float t = (life_ratio - 0.8f) / 0.2f;
                color = sf::Color(255, 255, static_cast<sf::Uint8>(200 + 55 * t), static_cast<sf::Uint8>(255 * life_ratio));
            } else if (life_ratio > 0.5f) {
                // Yellow to Orange
                float t = (life_ratio - 0.5f) / 0.3f;
                color = sf::Color(255, static_cast<sf::Uint8>(140 + 115 * t), 0, static_cast<sf::Uint8>(255 * life_ratio));
            } else {
                // Orange to Red
                float t = life_ratio / 0.5f;
                color = sf::Color(255, static_cast<sf::Uint8>(60 * t), 0, static_cast<sf::Uint8>(255 * life_ratio));
            }
            
            shape.setFillColor(color);
            window.draw(shape);
        }
        
        // Draw core flame (small)
        sf::CircleShape core(5.0f);
        core.setOrigin(core.getRadius(), core.getRadius());
        core.setPosition(static_cast<float>(mouse_x), static_cast<float>(mouse_y));
        core.setFillColor(sf::Color(255, 255, 255, 200));
        window.draw(core);

        // Draw proximity indicator ring
        // Calculate distance to cylinder for visual feedback
        float grid_x = mouse_x / PIXEL_SCALE;
        float grid_y = mouse_y / PIXEL_SCALE;
        float dx = grid_x - CYLINDER_CENTER_X;
        float dy = grid_y - CYLINDER_CENTER_Y;
        float dist_to_cylinder = std::sqrt(dx * dx + dy * dy) * PIXEL_SCALE;
        
        if (dist_to_cylinder < PROXIMITY_HEAT_MAX_DIST) {
            float proximity = 1.0f - (dist_to_cylinder / PROXIMITY_HEAT_MAX_DIST);
            sf::CircleShape heat_ring(20.0f + 10.0f * proximity);
            heat_ring.setOrigin(heat_ring.getRadius(), heat_ring.getRadius());
            heat_ring.setPosition(static_cast<float>(mouse_x), static_cast<float>(mouse_y));
            heat_ring.setFillColor(sf::Color(0, 0, 0, 0));
            heat_ring.setOutlineColor(sf::Color(255, 100 + static_cast<sf::Uint8>(155 * proximity), 
                                                  50, static_cast<sf::Uint8>(100 * proximity)));
            heat_ring.setOutlineThickness(2.0f);
            window.draw(heat_ring);
        }
    }
    
    void drawCylinderOutline(sf::RenderWindow& window) {
        // Draw a simple 2D ellipse for the cylinder
        const int num_segments = 64;
        const float cx = CYLINDER_CENTER_X * PIXEL_SCALE;
        const float cy = CYLINDER_CENTER_Y * PIXEL_SCALE;
        const float rx = CYLINDER_RADIUS_X * PIXEL_SCALE;
        const float ry = CYLINDER_RADIUS_Y * PIXEL_SCALE;
        
        // Draw filled ellipse
        sf::ConvexShape ellipse;
        ellipse.setPointCount(num_segments);
        for (int i = 0; i < num_segments; ++i) {
            float angle = 2.0f * M_PI * i / num_segments;
            float x = cx + rx * std::cos(angle);
            float y = cy + ry * std::sin(angle);
            ellipse.setPoint(i, sf::Vector2f(x, y));
        }
        ellipse.setFillColor(sf::Color(100, 100, 100, 50));
        ellipse.setOutlineColor(sf::Color(150, 150, 150, 200));
        ellipse.setOutlineThickness(2.0f);
        window.draw(ellipse);
        
        // Draw center cross for visual reference
        sf::Vertex h_line[] = {
            sf::Vertex(sf::Vector2f(cx - rx * 0.3f, cy), sf::Color(150, 150, 150, 100)),
            sf::Vertex(sf::Vector2f(cx + rx * 0.3f, cy), sf::Color(150, 150, 150, 100))
        };
        sf::Vertex v_line[] = {
            sf::Vertex(sf::Vector2f(cx, cy - ry * 0.3f), sf::Color(150, 150, 150, 100)),
            sf::Vertex(sf::Vector2f(cx, cy + ry * 0.3f), sf::Color(150, 150, 150, 100))
        };
        window.draw(h_line, 2, sf::Lines);
        window.draw(v_line, 2, sf::Lines);
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Heat Equation Simulation");
    window.setFramerateLimit(60);
    
    // Hide default cursor
    window.setMouseCursorVisible(false);
    
    HeatSimulation simulation;
    
    sf::Clock clock;
    sf::Clock fps_clock;
    int frame_count = 0;
    float fps = 0.0f;
    
    // Font for FPS display
    sf::Font font;
    bool has_font = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    
    sf::Text fps_text;
    if (has_font) {
        fps_text.setFont(font);
        fps_text.setCharacterSize(14);
        fps_text.setFillColor(sf::Color::White);
        fps_text.setPosition(10, 10);
    }
    
    sf::Text info_text;
    if (has_font) {
        info_text.setFont(font);
        info_text.setCharacterSize(12);
        info_text.setFillColor(sf::Color(200, 200, 200));
        info_text.setPosition(10, 30);
        info_text.setString("Move cursor close to cylinder to heat it | R to reset | ESC to exit");
    }
    
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                }
                if (event.key.code == sf::Keyboard::R) {
                    // Reset simulation
                    simulation = HeatSimulation();
                }
            }
        }
        
        float dt = clock.restart().asSeconds();
        
        // Calculate FPS
        frame_count++;
        if (fps_clock.getElapsedTime().asSeconds() >= 1.0f) {
            fps = frame_count / fps_clock.restart().asSeconds();
            frame_count = 0;
            if (has_font) {
                fps_text.setString("FPS: " + std::to_string(static_cast<int>(fps)));
            }
        }
        
        // Get mouse position
        sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
        int mouse_x = mouse_pos.x;
        int mouse_y = mouse_pos.y;
        
        // Update simulation (no click-to-heat, only proximity)
        simulation.update(dt, mouse_x, mouse_y);
        
        // Render
        window.clear(sf::Color(20, 20, 40));
        
        simulation.draw(window);
        simulation.drawCylinderOutline(window);
        simulation.drawCursor(window, mouse_x, mouse_y);
        
        if (has_font) {
            window.draw(fps_text);
            window.draw(info_text);
        }
        
        window.display();
    }
    
    return 0;
}