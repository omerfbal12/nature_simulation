#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <iostream>
#include <list>
#include <string>
#include <sstream>

// Simulation parameters
constexpr int GRID_WIDTH = 400;
constexpr int GRID_HEIGHT = 300;
constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr float PIXEL_SCALE = 2.0f;  // Scale grid to window

// Physics parameters
constexpr float DT = 0.016f;  // Time step (approx 60fps)
constexpr float DX = 0.01f;   // Spatial step
constexpr float ALPHA = 0.0001f;  // Base thermal diffusivity
constexpr float AMBIENT_TEMP = 25.0f;
constexpr float MAX_TEMP = 800.0f;
constexpr float MIN_TEMP = -100.0f;  // Very cold temperature
constexpr float CURSOR_HEAT_RADIUS = 40.0f;
constexpr float CURSOR_HEAT_POWER = 800.0f;
constexpr float CURSOR_COLD_POWER = -200.0f;  // Cold power (negative)
constexpr float INITIAL_OBJECT_TEMP = 120.0f;
constexpr float PROXIMITY_HEAT_MAX_DIST = 150.0f;  // Max distance for proximity heating
constexpr float CYLINDER_COOLING_RATE = 0.005f; // Reduced cooling rate for cylinder

// Material types with different thermal properties
enum MaterialType {
    MATERIAL_METAL,     // High conductivity (copper, aluminum)
    MATERIAL_WOOD,      // Low conductivity
    MATERIAL_FOAM,      // Very low conductivity (insulator)
    MATERIAL_GLASS,     // Medium-low conductivity
    MATERIAL_STONE,     // Medium conductivity
    MATERIAL_COUNT
};

struct Material {
    std::string name;
    float thermal_diffusivity;  // Relative to base ALPHA
    float cooling_rate;         // How fast it cools to ambient
    float melting_point;        // Temperature where material liquefies
    sf::Color base_color;       // Base color for UI display
};

// Real-world inspired thermal properties (scaled for simulation)
// Thermal diffusivity values are relative to the base ALPHA
const Material MATERIALS[MATERIAL_COUNT] = {
    {"Metal",  3.0f,  0.008f, 600.0f, sf::Color(180, 180, 200)},  // High conductivity
    {"Wood",   0.3f,  0.002f, 320.0f, sf::Color(139, 90, 43)},    // Low conductivity
    {"Foam",   0.1f,  0.001f, 220.0f, sf::Color(255, 255, 240)},  // Very low (insulator)
    {"Glass",  0.5f,  0.003f, 450.0f, sf::Color(200, 230, 255)},  // Medium-low
    {"Stone",  1.2f,  0.005f, 700.0f, sf::Color(128, 128, 128)}   // Medium
};

// Cursor modes
enum CursorMode {
    CURSOR_HOT,   // Heat ball
    CURSOR_COLD,  // Ice ball
    CURSOR_COUNT
};

// Cylinder parameters
constexpr float CYLINDER_CENTER_X = GRID_WIDTH * 0.5f;
constexpr float CYLINDER_CENTER_Y = GRID_HEIGHT * 0.5f;
constexpr float CYLINDER_RADIUS_X = 60.0f;  // Horizontal radius
constexpr float CYLINDER_RADIUS_Y = 40.0f;  // Vertical radius

// Stability condition: alpha * dt / dx^2 < 0.25 for 2D
// Using maximum material diffusivity for safety
constexpr float MAX_MATERIAL_DIFFUSIVITY = 3.0f;  // Metal has highest
constexpr float STABILITY_FACTOR = ALPHA * MAX_MATERIAL_DIFFUSIVITY * DT / (DX * DX);

struct Particle {
    sf::Vector2f position;
    sf::Vector2f velocity;
    float life; // 0.0 to 1.0
    float initial_life;
    float size;
    bool is_ice;  // True for ice particles, false for fire particles
};

class HeatSimulation {
private:
    std::vector<float> temp_current;
    std::vector<float> temp_next;
    std::vector<float> thermal_conductivity;  // Varies by material
    std::vector<bool> is_cylinder;            // Material occupancy (solid or liquid)
    std::vector<bool> is_liquid;              // Liquid state for material cells
    std::vector<int> cylinder_indices; // Cached indices of cylinder cells
    std::vector<sf::Uint8> pixels;
    sf::Texture texture;
    sf::Sprite sprite;
    
    // Current material selection
    MaterialType current_material;
    
    // Cursor mode
    CursorMode cursor_mode;
    
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
          is_liquid(GRID_WIDTH * GRID_HEIGHT, false),
          pixels(GRID_WIDTH * GRID_HEIGHT * 4, 0),
          current_material(MATERIAL_METAL),
          cursor_mode(CURSOR_HOT),
          rng(std::random_device{}()),
          dist(0.0f, 1.0f) {
        
        // Validate stability condition at runtime
        validateStability();
        
        // Initialize texture and sprite
        texture.create(GRID_WIDTH, GRID_HEIGHT);
        sprite.setTexture(texture);
        sprite.setScale(PIXEL_SCALE, PIXEL_SCALE);
        
        // Initialize cylinder shape with default material
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
        float material_alpha = ALPHA * MATERIALS[current_material].thermal_diffusivity;
        
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                float dx = (x - CYLINDER_CENTER_X) / CYLINDER_RADIUS_X;
                float dy = (y - CYLINDER_CENTER_Y) / CYLINDER_RADIUS_Y;
                float dist_sq = dx * dx + dy * dy;
                
                int idx = y * GRID_WIDTH + x;
                if (dist_sq <= 1.0f) {
                    is_cylinder[idx] = true;
                    is_liquid[idx] = false;
                    thermal_conductivity[idx] = material_alpha;
                    temp_current[idx] = INITIAL_OBJECT_TEMP;
                    temp_next[idx] = INITIAL_OBJECT_TEMP;
                    cylinder_indices.push_back(idx);
                } else {
                    is_cylinder[idx] = false;
                    is_liquid[idx] = false;
                    thermal_conductivity[idx] = ALPHA * 0.5f;  // Air conducts poorly
                    temp_current[idx] = AMBIENT_TEMP;
                    temp_next[idx] = AMBIENT_TEMP;
                }
            }
        }
    }
    
    void initializeTemperatures() {
        // Keep air at ambient temperature without noise
        for (size_t i = 0; i < temp_current.size(); ++i) {
            if (!is_cylinder[i]) {
                temp_current[i] = AMBIENT_TEMP;
                temp_next[i] = AMBIENT_TEMP;
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
        
        // Determine power based on cursor mode
        float proximity_power = (cursor_mode == CURSOR_HOT) ? CURSOR_HEAT_POWER : CURSOR_COLD_POWER;
        
        // Apply heat/cooling to the material based on proximity
        // Iterate over current material cells (including displaced liquid)
        float heat_radius = CURSOR_HEAT_RADIUS;
        float heat_radius_sq = heat_radius * heat_radius;
        float heat_radius_factor = 1.0f / (2.0f * heat_radius_sq * 0.25f);

        for (int idx : cylinder_indices) {
            int x = idx % GRID_WIDTH;
            int y = idx / GRID_WIDTH;
            
            // Distance from cursor to this material cell
            float cell_dx = x - grid_x;
            float cell_dy = y - grid_y;
            float dist_sq = cell_dx * cell_dx + cell_dy * cell_dy;
            
            if (dist_sq > heat_radius_sq) continue;
            
            // Apply heat with Gaussian falloff from cursor position
            float heat_factor = std::exp(-dist_sq * heat_radius_factor);
            
            float heat_addition = proximity_power * heat_factor * DT;
            
            temp_current[idx] += heat_addition;
            // Clamp temperature based on mode
            if (cursor_mode == CURSOR_HOT) {
                temp_current[idx] = std::min(temp_current[idx], MAX_TEMP);
            } else {
                temp_current[idx] = std::max(temp_current[idx], MIN_TEMP);
            }
        }
    }
    
    void solveHeatEquation() {
        // Explicit finite difference method
        // T_new = T_old + factor * (T_left + T_right + T_up + T_down - 4*T_old)
        
        // Get material-specific cooling rate
        float material_cooling_rate = MATERIALS[current_material].cooling_rate;
        
        for (int y = 1; y < GRID_HEIGHT - 1; ++y) {
            for (int x = 1; x < GRID_WIDTH - 1; ++x) {
                int idx = index(x, y);
                
                float T_center = temp_current[idx];
                float T_left = temp_current[index(x - 1, y)];
                float T_right = temp_current[index(x + 1, y)];
                float T_up = temp_current[index(x, y - 1)];
                float T_down = temp_current[index(x, y + 1)];
                
                // Calculate factor dynamically based on local thermal conductivity
                float diffusion = thermal_conductivity[idx];
                if (is_liquid[idx]) {
                    diffusion *= 0.2f;  // Reduce diffusion to keep heat in flowing liquid
                }
                float factor = diffusion * DT / (DX * DX);
                
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
                    // Liquid cools more slowly to preserve heat while flowing
                    float cooling_rate = is_liquid[idx] ? (material_cooling_rate * 0.2f) : material_cooling_rate;
                    temp_next[idx] += cooling_rate * (AMBIENT_TEMP - temp_next[idx]) * DT;
                }
            }
        }
        
        // Boundary conditions (Dirichlet: fixed ambient at edges)
        for (int x = 0; x < GRID_WIDTH; ++x) {
            temp_next[index(x, 0)] = AMBIENT_TEMP;
            if (!is_cylinder[index(x, GRID_HEIGHT - 1)]) {
                temp_next[index(x, GRID_HEIGHT - 1)] = AMBIENT_TEMP;
            }
        }
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            temp_next[index(0, y)] = AMBIENT_TEMP;
            temp_next[index(GRID_WIDTH - 1, y)] = AMBIENT_TEMP;
        }
        
        // Swap buffers
        std::swap(temp_current, temp_next);
    }

    void moveMaterialCell(int from_idx, int to_idx) {
        is_cylinder[to_idx] = true;
        is_liquid[to_idx] = is_liquid[from_idx];
        thermal_conductivity[to_idx] = thermal_conductivity[from_idx];
        temp_current[to_idx] = temp_current[from_idx];
        temp_next[to_idx] = temp_current[from_idx];

        is_cylinder[from_idx] = false;
        is_liquid[from_idx] = false;
        thermal_conductivity[from_idx] = ALPHA * 0.5f;
        temp_current[from_idx] = AMBIENT_TEMP;
        temp_next[from_idx] = AMBIENT_TEMP;
    }

    void updatePhaseChangeAndFlow() {
        float melting_point = MATERIALS[current_material].melting_point;

        // Update phase state based on temperature
        for (size_t i = 0; i < is_cylinder.size(); ++i) {
            if (is_cylinder[i]) {
                is_liquid[i] = temp_current[i] >= melting_point;
            } else {
                is_liquid[i] = false;
            }
        }

        // Gravity-driven flow for liquid cells (bottom-up)
        std::uniform_int_distribution<int> side_choice(0, 1);
        for (int y = GRID_HEIGHT - 2; y >= 0; --y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                int idx = index(x, y);
                if (!is_cylinder[idx] || !is_liquid[idx]) {
                    continue;
                }

                int below = index(x, y + 1);
                if (!is_cylinder[below]) {
                    moveMaterialCell(idx, below);
                    continue;
                }

                bool moved = false;
                int first = side_choice(rng);
                int offsets[2] = { -1, 1 };
                for (int i = 0; i < 2; ++i) {
                    int offset = offsets[(first + i) % 2];
                    int nx = x + offset;
                    if (nx < 0 || nx >= GRID_WIDTH) {
                        continue;
                    }
                    int down_diag = index(nx, y + 1);
                    if (!is_cylinder[down_diag]) {
                        moveMaterialCell(idx, down_diag);
                        moved = true;
                        break;
                    }
                }

                if (!moved) {
                    // Stay in place if no available space
                    continue;
                }
            }
        }

        // Refresh material indices for proximity heating
        cylinder_indices.clear();
        for (size_t i = 0; i < is_cylinder.size(); ++i) {
            if (is_cylinder[i]) {
                cylinder_indices.push_back(static_cast<int>(i));
            }
        }
    }
    
    sf::Color temperatureToColor(float temp) {
        // Smooth temperature mapping:
        // Cold: deep blue -> cyan -> white
        // Hot: dark red -> orange -> yellow -> white (blackbody-like)
        // Avoids green background artifacts.
        
        sf::Uint8 r, g, b;
        
        if (temp <= AMBIENT_TEMP) {
            // Cold range: MIN_TEMP to AMBIENT_TEMP
            float t = (temp - MIN_TEMP) / (AMBIENT_TEMP - MIN_TEMP);
            t = std::clamp(t, 0.0f, 1.0f);
            
            // Gradient: deep blue (0,0,80) -> cyan (0,180,255) -> white (255,255,255)
            if (t < 0.5f) {
                float local_t = t / 0.5f;
                r = static_cast<sf::Uint8>(0 + local_t * 0);
                g = static_cast<sf::Uint8>(0 + local_t * 180);
                b = static_cast<sf::Uint8>(80 + local_t * 175);
            } else {
                float local_t = (t - 0.5f) / 0.5f;
                r = static_cast<sf::Uint8>(0 + local_t * 255);
                g = static_cast<sf::Uint8>(180 + local_t * 75);
                b = static_cast<sf::Uint8>(255);
            }
        } else {
            // Hot range: AMBIENT_TEMP to MAX_TEMP
            float t = (temp - AMBIENT_TEMP) / (MAX_TEMP - AMBIENT_TEMP);
            t = std::clamp(t, 0.0f, 1.0f);
            
            // Blackbody-like gradient: dark red -> red -> orange -> yellow -> white
            if (t < 0.25f) {
                float local_t = t / 0.25f;
                r = static_cast<sf::Uint8>(80 + local_t * 175);
                g = static_cast<sf::Uint8>(0 + local_t * 20);
                b = static_cast<sf::Uint8>(0);
            } else if (t < 0.5f) {
                float local_t = (t - 0.25f) / 0.25f;
                r = 255;
                g = static_cast<sf::Uint8>(20 + local_t * 120);
                b = static_cast<sf::Uint8>(0);
            } else if (t < 0.75f) {
                float local_t = (t - 0.5f) / 0.25f;
                r = 255;
                g = static_cast<sf::Uint8>(140 + local_t * 100);
                b = static_cast<sf::Uint8>(0 + local_t * 30);
            } else {
                float local_t = (t - 0.75f) / 0.25f;
                r = 255;
                g = static_cast<sf::Uint8>(240 + local_t * 15);
                b = static_cast<sf::Uint8>(30 + local_t * 225);
            }
        }
        
        return sf::Color(r, g, b);
    }
    
    void updatePixels() {
        const sf::Color air_color(10, 15, 35); // Static dark blue background
        
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                int idx = index(x, y);
                sf::Color color = air_color;
                
                if (is_cylinder[idx]) {
                    color = temperatureToColor(temp_current[idx]);
                }
                
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
        // Add new particles based on cursor mode
        int new_particles = 2; // Add a few particles per frame
        for (int i = 0; i < new_particles; ++i) {
            if (particles.size() < 100) {
                Particle p;
                p.position = sf::Vector2f(mouse_x, mouse_y);
                p.is_ice = (cursor_mode == CURSOR_COLD);
                
                if (p.is_ice) {
                    // Ice particles: fall down and spread
                    std::uniform_real_distribution<float> vx_dist(-15.0f, 15.0f);
                    std::uniform_real_distribution<float> vy_dist(20.0f, 60.0f);
                    std::uniform_real_distribution<float> life_dist(0.4f, 0.8f);
                    std::uniform_real_distribution<float> size_dist(3.0f, 8.0f);
                    
                    p.velocity = sf::Vector2f(vx_dist(rng), vy_dist(rng));
                    p.life = life_dist(rng);
                    p.initial_life = p.life;
                    p.size = size_dist(rng);
                } else {
                    // Fire particles: rise up
                    std::uniform_real_distribution<float> vx_dist(-10.0f, 10.0f);
                    std::uniform_real_distribution<float> vy_dist(-80.0f, -40.0f);
                    std::uniform_real_distribution<float> life_dist(0.5f, 1.0f);
                    std::uniform_real_distribution<float> size_dist(5.0f, 10.0f);
                    
                    p.velocity = sf::Vector2f(vx_dist(rng), vy_dist(rng));
                    p.life = life_dist(rng);
                    p.initial_life = p.life;
                    p.size = size_dist(rng);
                }
                
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
                
                if (it->is_ice) {
                    // Ice particles: gravity pulls down, slight deceleration
                    it->velocity.y += 40.0f * dt;
                    it->velocity.x *= 0.98f;
                } else {
                    // Fire particles: buoyancy pushes up
                    it->velocity.y -= 20.0f * dt;
                }
                
                // Shrink over time
                it->size = it->size * 0.99f;
                ++it;
            }
        }
    }
    
    void update(float dt, int& mouse_x, int& mouse_y) {
        flame_time += dt;
        
        // Collision detection removed to allow interaction with melted material
        
        // Apply proximity-based heat from cursor (always active based on distance)
        applyProximityHeat(mouse_x, mouse_y);
        
        // Solve heat equation (multiple iterations for stability/speed)
        for (int i = 0; i < 2; ++i) {
            solveHeatEquation();
        }
        
        // Update phase change and liquid flow
        updatePhaseChangeAndFlow();
        
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
            
            sf::Color color;
            
            if (p.is_ice) {
                // Ice particle colors: White -> Light Blue -> Blue -> Transparent
                if (life_ratio > 0.7f) {
                    // White to Light Blue
                    float t = (life_ratio - 0.7f) / 0.3f;
                    color = sf::Color(static_cast<sf::Uint8>(200 + 55 * t), 
                                      static_cast<sf::Uint8>(230 + 25 * t), 
                                      255, 
                                      static_cast<sf::Uint8>(255 * life_ratio));
                } else if (life_ratio > 0.4f) {
                    // Light Blue to Blue
                    float t = (life_ratio - 0.4f) / 0.3f;
                    color = sf::Color(static_cast<sf::Uint8>(100 + 100 * t), 
                                      static_cast<sf::Uint8>(180 + 50 * t), 
                                      255, 
                                      static_cast<sf::Uint8>(255 * life_ratio));
                } else {
                    // Blue to Dark Blue
                    float t = life_ratio / 0.4f;
                    color = sf::Color(static_cast<sf::Uint8>(50 + 50 * t), 
                                      static_cast<sf::Uint8>(100 + 80 * t), 
                                      static_cast<sf::Uint8>(200 + 55 * t), 
                                      static_cast<sf::Uint8>(255 * life_ratio));
                }
            } else {
                // Fire particle colors: White -> Yellow -> Orange -> Red -> Transparent
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
            }
            
            shape.setFillColor(color);
            window.draw(shape);
        }
        
        // Draw core (different for hot/cold)
        if (cursor_mode == CURSOR_HOT) {
            // Hot core - bright orange/yellow
            sf::CircleShape core(5.0f);
            core.setOrigin(core.getRadius(), core.getRadius());
            core.setPosition(static_cast<float>(mouse_x), static_cast<float>(mouse_y));
            core.setFillColor(sf::Color(255, 255, 255, 200));
            window.draw(core);
        } else {
            // Cold core - bright blue/white ice ball
            sf::CircleShape core(6.0f);
            core.setOrigin(core.getRadius(), core.getRadius());
            core.setPosition(static_cast<float>(mouse_x), static_cast<float>(mouse_y));
            core.setFillColor(sf::Color(200, 230, 255, 220));
            window.draw(core);
            
            // Inner glow
            sf::CircleShape inner_core(3.0f);
            inner_core.setOrigin(inner_core.getRadius(), inner_core.getRadius());
            inner_core.setPosition(static_cast<float>(mouse_x), static_cast<float>(mouse_y));
            inner_core.setFillColor(sf::Color(255, 255, 255, 255));
            window.draw(inner_core);
        }

        // Draw proximity indicator ring
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
            
            if (cursor_mode == CURSOR_HOT) {
                heat_ring.setOutlineColor(sf::Color(255, 100 + static_cast<sf::Uint8>(155 * proximity), 
                                                      50, static_cast<sf::Uint8>(100 * proximity)));
            } else {
                heat_ring.setOutlineColor(sf::Color(100, 150 + static_cast<sf::Uint8>(105 * proximity), 
                                                      255, static_cast<sf::Uint8>(100 * proximity)));
            }
            heat_ring.setOutlineThickness(2.0f);
            window.draw(heat_ring);
        }
    }
    
    void drawCylinderOutline(sf::RenderWindow& window) {
        // Draw a dynamic outline based on the current material cells
        sf::Color material_color = MATERIALS[current_material].base_color;
        sf::VertexArray outline(sf::Points);
        
        for (int y = 1; y < GRID_HEIGHT - 1; ++y) {
            for (int x = 1; x < GRID_WIDTH - 1; ++x) {
                int idx = index(x, y);
                if (!is_cylinder[idx]) {
                    continue;
                }
                
                // Edge detection: any neighbor is air
                bool edge = false;
                if (!is_cylinder[index(x - 1, y)] || !is_cylinder[index(x + 1, y)] ||
                    !is_cylinder[index(x, y - 1)] || !is_cylinder[index(x, y + 1)]) {
                    edge = true;
                }
                
                if (edge) {
                    sf::Vertex v;
                    v.position = sf::Vector2f(x * PIXEL_SCALE, y * PIXEL_SCALE);
                    v.color = sf::Color(material_color.r, material_color.g, material_color.b, 180);
                    outline.append(v);
                }
            }
        }
        
        window.draw(outline);
    }
    
    // Public methods for controlling simulation
    void setMaterial(MaterialType material) {
        current_material = material;
        // Reinitialize cylinder with new material properties and reset state
        std::fill(is_cylinder.begin(), is_cylinder.end(), false);
        std::fill(is_liquid.begin(), is_liquid.end(), false);
        initializeCylinder();
        initializeTemperatures();
    }
    
    void cycleMaterial() {
        int next = (static_cast<int>(current_material) + 1) % MATERIAL_COUNT;
        setMaterial(static_cast<MaterialType>(next));
    }
    
    void setCursorMode(CursorMode mode) {
        cursor_mode = mode;
    }
    
    void toggleCursorMode() {
        cursor_mode = (cursor_mode == CURSOR_HOT) ? CURSOR_COLD : CURSOR_HOT;
    }
    
    MaterialType getMaterial() const { return current_material; }
    CursorMode getCursorMode() const { return cursor_mode; }
    
    std::string getMaterialName() const {
        return MATERIALS[current_material].name;
    }
    
    std::string getCursorModeName() const {
        return (cursor_mode == CURSOR_HOT) ? "Heat Ball" : "Ice Ball";
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Heat Equation Simulation - Interactive Materials");
    window.setFramerateLimit(60);
    
    // Hide default cursor
    window.setMouseCursorVisible(false);
    
    HeatSimulation simulation;
    
    sf::Clock clock;
    sf::Clock fps_clock;
    int frame_count = 0;
    float fps = 0.0f;
    
    // Font for text display (macOS paths)
    sf::Font font;
    bool has_font = font.loadFromFile("/System/Library/Fonts/SFNS.ttf");
    if (!has_font) {
        has_font = font.loadFromFile("/System/Library/Fonts/Helvetica.ttc");
    }
    if (!has_font) {
        has_font = font.loadFromFile("/Library/Fonts/Arial.ttf");
    }
    
    // HUD background rectangle
    sf::RectangleShape hud_bg(sf::Vector2f(160, 130));
    hud_bg.setPosition(WINDOW_WIDTH - 170, 10);
    hud_bg.setFillColor(sf::Color(20, 20, 35, 200));
    hud_bg.setOutlineColor(sf::Color(60, 60, 80, 200));
    hud_bg.setOutlineThickness(1.0f);
    
    sf::Text hud_title;
    if (has_font) {
        hud_title.setFont(font);
        hud_title.setCharacterSize(13);
        hud_title.setFillColor(sf::Color(180, 180, 200));
        hud_title.setPosition(WINDOW_WIDTH - 160, 16);
        hud_title.setString("CONTROLS");
    }
    
    sf::Text hud_keys;
    if (has_font) {
        hud_keys.setFont(font);
        hud_keys.setCharacterSize(11);
        hud_keys.setFillColor(sf::Color(160, 160, 180));
        hud_keys.setPosition(WINDOW_WIDTH - 160, 36);
        hud_keys.setString("[M] Material\n[Space] Heat/Ice\n[ESC] Exit");
    }
    
    sf::Text hud_material;
    if (has_font) {
        hud_material.setFont(font);
        hud_material.setCharacterSize(12);
        hud_material.setFillColor(sf::Color(255, 220, 100));
        hud_material.setPosition(WINDOW_WIDTH - 160, 90);
    }
    
    sf::Text hud_cursor;
    if (has_font) {
        hud_cursor.setFont(font);
        hud_cursor.setCharacterSize(12);
        hud_cursor.setPosition(WINDOW_WIDTH - 160, 110);
    }
    
    sf::Text fps_text;
    if (has_font) {
        fps_text.setFont(font);
        fps_text.setCharacterSize(11);
        fps_text.setFillColor(sf::Color(120, 120, 140));
        fps_text.setPosition(WINDOW_WIDTH - 60, WINDOW_HEIGHT - 25);
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
                if (event.key.code == sf::Keyboard::M) {
                    // Cycle through materials
                    simulation.cycleMaterial();
                }
                if (event.key.code == sf::Keyboard::Space) {
                    // Toggle between heat and ice cursor
                    simulation.toggleCursorMode();
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
        
        // Update simulation
        simulation.update(dt, mouse_x, mouse_y);
        
        // Update HUD text displays
        if (has_font) {
            hud_material.setString("Material: " + simulation.getMaterialName());
            
            if (simulation.getCursorMode() == CURSOR_HOT) {
                hud_cursor.setFillColor(sf::Color(255, 150, 50));
                hud_cursor.setString("Mode: Heat Ball");
            } else {
                hud_cursor.setFillColor(sf::Color(100, 200, 255));
                hud_cursor.setString("Mode: Ice Ball");
            }
        }
        
        // Render
        window.clear(sf::Color(20, 20, 40));
        
        simulation.draw(window);
        simulation.drawCylinderOutline(window);
        simulation.drawCursor(window, mouse_x, mouse_y);
        
        // Draw HUD
        if (has_font) {
            window.draw(hud_bg);
            window.draw(hud_title);
            window.draw(hud_keys);
            window.draw(hud_material);
            window.draw(hud_cursor);
            window.draw(fps_text);
        }
        
        window.display();
    }
    
    return 0;
}