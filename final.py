"""
Deterministic Ecosystem Simulator

A complex ecosystem simulation with plants, herbivores, and carnivores.
Features:
- Deterministic RNG with seed control
- Fixed timestep simulation
- Energy accounting and metabolism
- State machine driven behaviors (chase, eat, flee, mate, wander)
- Seasons and environmental events (winter, drought)
- Herding behavior for herbivores
- Pack hunting for carnivores
- Complex entity interactions
"""

import math
import random
import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from collections import defaultdict
import json


# =============================================================================
# DETERMINISTIC RNG SYSTEM
# =============================================================================

class SeededRNG:
    """Deterministic random number generator with single seed control."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.call_count = 0
    
    def reset(self, seed: Optional[int] = None):
        """Reset RNG to initial state or new seed."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.call_count = 0
    
    def random(self) -> float:
        """Get random float [0, 1)."""
        self.call_count += 1
        return self.rng.random()
    
    def uniform(self, a: float, b: float) -> float:
        """Get random float in range [a, b)."""
        self.call_count += 1
        return self.rng.uniform(a, b)
    
    def randint(self, a: int, b: int) -> int:
        """Get random integer in range [a, b]."""
        self.call_count += 1
        return self.rng.randint(a, b)
    
    def choice(self, seq: List[Any]) -> Any:
        """Get random element from sequence."""
        self.call_count += 1
        return self.rng.choice(seq)
    
    def shuffle(self, seq: List[Any]):
        """Shuffle sequence in place."""
        self.call_count += 1
        self.rng.shuffle(seq)
    
    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Get Gaussian distributed random number."""
        self.call_count += 1
        return self.rng.gauss(mu, sigma)


# Global RNG instance - all randomness flows through here
_global_rng = SeededRNG(42)


def set_seed(seed: int):
    """Set global simulation seed."""
    _global_rng.reset(seed)


def rng() -> SeededRNG:
    """Get global RNG instance."""
    return _global_rng


# =============================================================================
# FIXED TIMESTEP SYSTEM
# =============================================================================

class FixedTimestep:
    """Fixed timestep accumulator for deterministic simulation."""
    
    def __init__(self, dt: float = 1.0 / 60.0):
        self.dt = dt  # Fixed timestep in seconds
        self.accumulator = 0.0
        self.total_time = 0.0
        self.tick_count = 0
    
    def step(self, delta_time: float) -> int:
        """
        Accumulate time and return number of fixed steps to process.
        This ensures deterministic simulation regardless of frame rate.
        """
        self.accumulator += delta_time
        steps = 0
        
        while self.accumulator >= self.dt:
            self.accumulator -= self.dt
            self.total_time += self.dt
            steps += 1
        
        self.tick_count += steps
        return steps
    
    def get_alpha(self) -> float:
        """Get interpolation factor for rendering (if needed)."""
        return self.accumulator / self.dt


# =============================================================================
# VECTOR MATH
# =============================================================================

@dataclass
class Vec2:
    """2D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x / scalar, self.y / scalar)
    
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y
    
    def normalized(self) -> 'Vec2':
        l = self.length()
        if l < 0.0001:
            return Vec2(0, 0)
        return self / l
    
    def distance_to(self, other: 'Vec2') -> float:
        return (self - other).length()
    
    def distance_sq_to(self, other: 'Vec2') -> float:
        return (self - other).length_sq()
    
    def clamp_magnitude(self, max_length: float) -> 'Vec2':
        l = self.length()
        if l > max_length:
            return self.normalized() * max_length
        return Vec2(self.x, self.y)
    
    def limit_within(self, bounds: 'Bounds') -> 'Vec2':
        """Clamp position within bounds."""
        return Vec2(
            max(bounds.min_x, min(bounds.max_x, self.x)),
            max(bounds.min_y, min(bounds.max_y, self.y))
        )


@dataclass
class Bounds:
    """World boundaries."""
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 100.0
    max_y: float = 100.0
    
    def contains(self, pos: Vec2) -> bool:
        return (self.min_x <= pos.x <= self.max_x and 
                self.min_y <= pos.y <= self.max_y)
    
    def random_position(self) -> Vec2:
        return Vec2(
            rng().uniform(self.min_x, self.max_x),
            rng().uniform(self.min_y, self.max_y)
        )
    
    def wrap_position(self, pos: Vec2) -> Vec2:
        """Wrap position around world edges (toroidal)."""
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        
        x = pos.x
        y = pos.y
        
        while x < self.min_x:
            x += width
        while x > self.max_x:
            x -= width
        while y < self.min_y:
            y += height
        while y > self.max_y:
            y -= height
            
        return Vec2(x, y)


# =============================================================================
# STATE MACHINE
# =============================================================================

class State(Enum):
    """Entity behavior states."""
    IDLE = auto()
    WANDER = auto()
    CHASE = auto()      # Pursue prey/mate
    FLEE = auto()       # Run from predator
    EAT = auto()        # Consuming food
    MATE = auto()       # Reproducing
    HUNT = auto()       # Pack hunting coordination
    HERD = auto()       # Grouping with same species
    REST = auto()       # Recovering energy
    DEAD = auto()


class StateMachine:
    """Finite state machine for entity behavior."""
    
    def __init__(self, initial_state: State = State.IDLE):
        self.current_state = initial_state
        self.previous_state = initial_state
        self.state_timer = 0.0
        self.state_data: Dict[str, Any] = {}
    
    def change_state(self, new_state: State, **data):
        """Change to new state with optional data."""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_timer = 0.0
        self.state_data = data
    
    def update(self, dt: float):
        """Update state timer."""
        self.state_timer += dt
    
    def is_in_state(self, state: State, min_time: float = 0.0) -> bool:
        """Check if in state for at least min_time."""
        return self.current_state == state and self.state_timer >= min_time


# =============================================================================
# SEASONS AND ENVIRONMENT
# =============================================================================

class Season(Enum):
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()
    WINTER = auto()


class WeatherEvent(Enum):
    NONE = auto()
    DROUGHT = auto()      # Reduced plant growth
    FLOOD = auto()        # Increased plant growth but movement penalty
    COLD_SNAP = auto()    # Extra harsh winter conditions
    PLAGUE = auto()       # Disease spread


@dataclass
class Environment:
    """Environmental conditions affecting the ecosystem."""
    season: Season = Season.SPRING
    season_progress: float = 0.0  # 0.0 to 1.0 within season
    day_of_year: float = 0.0      # 0 to 365
    
    # Season durations (in simulation days)
    season_length: float = 30.0
    year_length: float = 120.0
    
    # Current weather event
    active_event: WeatherEvent = WeatherEvent.NONE
    event_timer: float = 0.0
    event_duration: float = 0.0
    
    # Growth multipliers
    plant_growth_rate: float = 1.0
    plant_energy_value: float = 1.0
    metabolism_multiplier: float = 1.0
    reproduction_cost_multiplier: float = 1.0
    
    def update(self, dt: float):
        """Update environmental conditions."""
        self.day_of_year += dt
        
        # Update season
        season_index = int((self.day_of_year % self.year_length) / self.season_length)
        season_index = season_index % 4
        
        seasons = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
        new_season = seasons[season_index]
        
        if new_season != self.season:
            self._on_season_change(new_season)
        
        self.season = new_season
        self.season_progress = (self.day_of_year % self.season_length) / self.season_length
        
        # Update weather events
        self._update_weather_events(dt)
        self._apply_season_effects()
    
    def _on_season_change(self, new_season: Season):
        """Handle season transition."""
        # Chance to trigger weather event on season change
        if rng().random() < 0.15 and self.active_event == WeatherEvent.NONE:
            self._trigger_random_event()
    
    def _trigger_random_event(self):
        """Trigger a random weather event."""
        events = [WeatherEvent.DROUGHT, WeatherEvent.COLD_SNAP, WeatherEvent.FLOOD]
        weights = [0.4, 0.4, 0.2]
        
        r = rng().random()
        cumulative = 0.0
        for event, weight in zip(events, weights):
            cumulative += weight
            if r < cumulative:
                self.active_event = event
                self.event_duration = rng().uniform(10, 25)
                self.event_timer = 0.0
                break
    
    def _update_weather_events(self, dt: float):
        """Update active weather events."""
        if self.active_event != WeatherEvent.NONE:
            self.event_timer += dt
            if self.event_timer >= self.event_duration:
                self.active_event = WeatherEvent.NONE
                self.event_timer = 0.0
    
    def _apply_season_effects(self):
        """Apply seasonal modifiers."""
        # Reset base values to avoid compounding
        self.plant_growth_rate = 1.0
        self.plant_energy_value = 1.0
        self.metabolism_multiplier = 1.0

        base_growth = {
            Season.SPRING: 1.5,
            Season.SUMMER: 1.2,
            Season.AUTUMN: 0.8,
            Season.WINTER: 0.2
        }
        
        base_metabolism = {
            Season.SPRING: 1.0,
            Season.SUMMER: 1.0,
            Season.AUTUMN: 0.9,
            Season.WINTER: 1.4
        }
        
        self.plant_growth_rate = base_growth[self.season]
        self.metabolism_multiplier = base_metabolism[self.season]
        
        # Apply weather event effects
        if self.active_event == WeatherEvent.DROUGHT:
            self.plant_growth_rate *= 0.3
            self.plant_energy_value *= 0.7
        elif self.active_event == WeatherEvent.FLOOD:
            self.plant_growth_rate *= 2.0
            self.plant_energy_value *= 0.8
        elif self.active_event == WeatherEvent.COLD_SNAP:
            self.metabolism_multiplier *= 1.5
            self.plant_growth_rate *= 0.1
    
    def get_season_name(self) -> str:
        return self.season.name
    
    def get_event_name(self) -> str:
        return self.active_event.name if self.active_event != WeatherEvent.NONE else "None"


# =============================================================================
# SPATIAL INDEXING
# =============================================================================

class SpatialGrid:
    """Efficient spatial queries using a grid."""
    
    def __init__(self, bounds: Bounds, cell_size: float = 10.0):
        self.bounds = bounds
        self.cell_size = cell_size
        self.cells: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
        self.entity_cell: Dict[int, Tuple[int, int]] = {}
        
        self.num_x = int((bounds.max_x - bounds.min_x) / cell_size) + 1
        self.num_y = int((bounds.max_y - bounds.min_y) / cell_size) + 1
    
    def _get_cell_coords(self, pos: Vec2) -> Tuple[int, int]:
        """Get grid cell coordinates for position."""
        cx = int((pos.x - self.bounds.min_x) / self.cell_size)
        cy = int((pos.y - self.bounds.min_y) / self.cell_size)
        return (max(0, min(cx, self.num_x - 1)), max(0, min(cy, self.num_y - 1)))
    
    def insert(self, entity):
        """Insert entity into grid."""
        cell = self._get_cell_coords(entity.position)
        self.cells[cell].append(entity)
        self.entity_cell[id(entity)] = cell
    
    def remove(self, entity):
        """Remove entity from grid."""
        entity_id = id(entity)
        if entity_id in self.entity_cell:
            cell = self.entity_cell[entity_id]
            if entity in self.cells[cell]:
                self.cells[cell].remove(entity)
            del self.entity_cell[entity_id]
    
    def update(self, entity):
        """Update entity position in grid."""
        self.remove(entity)
        self.insert(entity)
    
    def query_radius(self, pos: Vec2, radius: float, filter_type: Optional[type] = None) -> List[Any]:
        """Query entities within radius."""
        radius_sq = radius * radius
        results = []
        
        # Get cells in range
        min_cell = self._get_cell_coords(Vec2(pos.x - radius, pos.y - radius))
        max_cell = self._get_cell_coords(Vec2(pos.x + radius, pos.y + radius))
        
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cy in range(min_cell[1], max_cell[1] + 1):
                for entity in self.cells[(cx, cy)]:
                    if entity.position.distance_sq_to(pos) <= radius_sq:
                        if filter_type is None or isinstance(entity, filter_type):
                            results.append(entity)
        
        return results
    
    def clear(self):
        """Clear all entities from grid."""
        self.cells.clear()
        self.entity_cell.clear()


# =============================================================================
# BASE ENTITY CLASS
# =============================================================================

class Entity:
    """Base class for all living entities in the ecosystem."""
    
    _id_counter = 0
    
    def __init__(self, position: Vec2, energy: float = 100.0):
        Entity._id_counter += 1
        self.id = Entity._id_counter
        self.position = position
        self.velocity = Vec2(0, 0)
        self.energy = energy
        self.max_energy = energy * 1.5
        self.age = 0.0
        self.alive = True
        
        self.state_machine = StateMachine(State.IDLE)
        
        # Energy accounting
        self.base_metabolism = 0.1
        self.movement_cost_factor = 0.05
        self.base_reproduction_threshold = energy * 0.8
        self.reproduction_threshold = self.base_reproduction_threshold
        self.reproduction_cost = energy * 0.4
        
        # Aging
        self.max_age = 200.0
        self.aging_start = 120.0
        self.aging_base_chance = 0.005  # Daily probability
        self.aging_daily_increase = 0.01
        self.aging_max_chance = 0.95
        
        # Sensory parameters
        self.perception_radius = 15.0
        
        # Statistics
        self.energy_consumed = 0.0
        self.distance_traveled = 0.0
        self.children_spawned = 0
    
    def get_metabolism_cost(self, environment: Environment) -> float:
        """Calculate base metabolism cost."""
        return self.base_metabolism * environment.metabolism_multiplier
    
    def get_movement_cost(self, speed: float) -> float:
        """Calculate movement energy cost."""
        return speed * self.movement_cost_factor
    
    def consume_energy(self, amount: float):
        """Consume energy, die if depleted."""
        self.energy -= amount
        if self.energy <= 0:
            self.energy = 0
            self.alive = False
            self.state_machine.change_state(State.DEAD)
    
    def gain_energy(self, amount: float):
        """Gain energy, capped at max."""
        self.energy = min(self.max_energy, self.energy + amount)
        self.energy_consumed += amount
    
    def can_reproduce(self) -> bool:
        """Check if entity can reproduce."""
        return self.energy >= self.reproduction_threshold and self.alive
    
    def update(self, dt: float, environment: Environment, world: 'World'):
        """Update entity state."""
        self.age += dt
        self.state_machine.update(dt)
        
        # Base metabolism
        self.consume_energy(self.get_metabolism_cost(environment) * dt)
        
        # Aging-based mortality
        self._update_aging(world, dt)
    
    def _update_aging(self, world: 'World', dt: float):
        """Apply age-based mortality with increasing probability."""
        if not self.alive:
            return
        
        # Hard cap on lifespan
        if self.age >= self.max_age:
            self.on_death()
            world.log_event(f"{self.__class__.__name__} {self.id} died of old age")
            return
        
        # Probabilistic aging after aging_start
        if self.age >= self.aging_start:
            days_over = self.age - self.aging_start
            daily_chance = min(
                self.aging_max_chance,
                self.aging_base_chance + self.aging_daily_increase * days_over
            )
            scaled_chance = min(1.0, daily_chance * dt)
            if rng().random() < scaled_chance:
                self.on_death()
                world.log_event(f"{self.__class__.__name__} {self.id} died of old age")

    def on_death(self):
        """Called when entity dies."""
        self.alive = False
        self.state_machine.change_state(State.DEAD)


# =============================================================================
# PLANT CLASS
# =============================================================================

class Plant(Entity):
    """Plant - producer in the ecosystem."""
    
    def __init__(self, position: Vec2, energy: float = 50.0):
        super().__init__(position, energy)
        self.state_machine.change_state(State.IDLE)
        
        self.base_metabolism = 0.02
        self.growth_rate = 2.0
        self.max_size = rng().uniform(0.8, 1.2)
        self.size = 0.1
        self.spread_radius = 8.0
        self.base_reproduction_threshold = 80.0
        self.reproduction_threshold = self.base_reproduction_threshold
        
        # Aging
        self.max_age = 1200.0
        self.aging_start = 900.0
        self.aging_base_chance = 0.002
        self.aging_daily_increase = 0.01
        self.aging_max_chance = 0.85
        
        # Visual/energy value
        self.energy_value = energy * self.size
    
    def update(self, dt: float, environment: Environment, world: 'World'):
        """Plant growth and reproduction."""
        super().update(dt, environment, world)
        
        if not self.alive:
            return
        
        # Growth
        if self.size < self.max_size:
            growth = self.growth_rate * environment.plant_growth_rate * dt
            self.size = min(self.max_size, self.size + growth * 0.1)
            self.energy_value = 50.0 * self.size * environment.plant_energy_value
            self.gain_energy(growth)
        
        # Reproduction via spores/seeds
        if self.energy >= self.reproduction_threshold:
            self._try_spread(world, environment)
    
    def _try_spread(self, world: 'World', environment: Environment):
        """Try to spread to nearby location."""
        # Check local plant density
        nearby = world.spatial_grid.query_radius(self.position, self.spread_radius, Plant)
        if len(nearby) >= 5:  # Too crowded
            return
        
        # Create offspring at random nearby position
        angle = rng().uniform(0, 2 * math.pi)
        distance = rng().uniform(2, self.spread_radius)
        offset = Vec2(math.cos(angle) * distance, math.sin(angle) * distance)
        new_pos = (self.position + offset).limit_within(world.bounds)
        
        # Cost to reproduce
        self.consume_energy(self.reproduction_cost)
        
        # Create new plant
        child = Plant(new_pos, energy=30.0)
        world.add_entity(child)
        self.children_spawned += 1
    
    def get_energy_value(self) -> float:
        """Get nutritional energy value."""
        return self.energy_value


# =============================================================================
# HERBIVORE CLASS
# =============================================================================

class Herbivore(Entity):
    """Herbivore - eats plants, exhibits herding behavior."""
    
    def __init__(self, position: Vec2, energy: float = 150.0):
        super().__init__(position, energy)
        self.state_machine.change_state(State.WANDER)
        
        # Movement
        self.max_speed = 3.0
        self.max_force = 2.0
        self.base_metabolism = 0.15
        
        # Behavior parameters
        self.perception_radius = 20.0
        self.eating_range = 2.0
        self.herding_radius = 15.0
        self.flee_radius = 18.0
        
        # State thresholds
        self.hunger_threshold = 60.0  # Start looking for food
        self.fear_threshold = 15.0    # Start fleeing
        
        # Aging
        self.max_age = 900.0
        self.aging_start = 700.0
        self.aging_base_chance = 0.003
        self.aging_daily_increase = 0.015
        self.aging_max_chance = 0.9
        
        # Herding behavior weights
        self.separation_weight = 1.5
        self.alignment_weight = 0.8
        self.cohesion_weight = 1.0
        self.flee_weight = 3.0
        
        # Memory
        self.target_plant: Optional[Plant] = None
        self.target_mate: Optional['Herbivore'] = None
        self.last_safe_position: Optional[Vec2] = None
    
    def update(self, dt: float, environment: Environment, world: 'World'):
        """Update herbivore behavior."""
        super().update(dt, environment, world)
        
        if not self.alive:
            return
        
        # Sense environment
        nearby_plants = world.spatial_grid.query_radius(
            self.position, self.perception_radius, Plant
        )
        nearby_carnivores = world.spatial_grid.query_radius(
            self.position, self.flee_radius, Carnivore
        )
        nearby_herbivores = world.spatial_grid.query_radius(
            self.position, self.herding_radius, Herbivore
        )
        
        # Filter out self and dead entities
        nearby_herbivores = [h for h in nearby_herbivores if h.id != self.id and h.alive]
        nearby_carnivores = [c for c in nearby_carnivores if c.alive]
        nearby_plants = [p for p in nearby_plants if p.alive]
        
        # State machine logic
        self._update_state_machine(nearby_plants, nearby_carnivores, nearby_herbivores, dt)
        
        # Execute current state
        self._execute_state(nearby_plants, nearby_carnivores, nearby_herbivores, dt, world)
        
        # Update position
        self._update_movement(dt, world)
    
    def _update_state_machine(self, plants: List[Plant], carnivores: List['Carnivore'], 
                              herd: List['Herbivore'], dt: float):
        """Determine appropriate state based on conditions."""
        current = self.state_machine.current_state
        
        # Priority 1: Flee from predators
        if carnivores:
            closest = min(carnivores, key=lambda c: c.position.distance_to(self.position))
            if closest.position.distance_to(self.position) < self.flee_radius:
                if current != State.FLEE:
                    self.state_machine.change_state(State.FLEE, predator=closest)
                return
        
        # Priority 2: Mate if possible and energy is high
        if self.can_reproduce() and current not in [State.EAT, State.FLEE]:
            potential_mates = [h for h in herd if h.can_reproduce() and h.id != self.id]
            if potential_mates:
                closest = min(potential_mates, key=lambda h: h.position.distance_to(self.position))
                if closest.position.distance_to(self.position) < self.perception_radius:
                    if current != State.MATE:
                        self.state_machine.change_state(State.MATE, mate=closest)
                    return
        
        # Priority 3: Eat if hungry
        if self.energy < self.hunger_threshold and plants:
            if current != State.EAT:
                closest = min(plants, key=lambda p: p.position.distance_to(self.position))
                self.state_machine.change_state(State.EAT, target=closest)
            return
        
        # Priority 4: Herd (group with others for safety)
        if len(herd) > 0 and current == State.WANDER:
            if rng().random() < 0.3:  # Chance to start herding
                self.state_machine.change_state(State.HERD)
                return
        
        # Default: Wander
        if current in [State.IDLE, State.HERD] and self.state_machine.state_timer > 5.0:
            self.state_machine.change_state(State.WANDER)
    
    def _execute_state(self, plants: List[Plant], carnivores: List['Carnivore'],
                       herd: List['Herbivore'], dt: float, world: 'World'):
        """Execute behavior for current state."""
        state = self.state_machine.current_state
        
        if state == State.FLEE:
            self._do_flee(dt, carnivores)
        elif state == State.EAT:
            self._do_eat(dt, plants, world)
        elif state == State.MATE:
            self._do_mate(dt, herd, world)
        elif state == State.HERD:
            self._do_herd(dt, herd)
        elif state == State.WANDER:
            self._do_wander(dt)
    
    def _do_flee(self, dt: float, carnivores: List['Carnivore']):
        """Flee from nearest predator."""
        if not carnivores:
            self.state_machine.change_state(State.WANDER)
            return
        
        predator = min(carnivores, key=lambda c: c.position.distance_to(self.position))
        flee_direction = (self.position - predator.position).normalized()
        
        # Boost speed when fleeing
        speed = self.max_speed * 1.5
        self.velocity = flee_direction * speed
        
        # High energy cost for fleeing
        self.consume_energy(self.get_movement_cost(speed) * dt * 2)
    
    def _do_eat(self, dt: float, plants: List[Plant], world: 'World'):
        """Move to and eat plants."""
        target = self.state_machine.state_data.get('target')
        
        if target is None or not target.alive:
            # Find new target
            if plants:
                target = min(plants, key=lambda p: p.position.distance_to(self.position))
                self.state_machine.state_data['target'] = target
            else:
                self.state_machine.change_state(State.WANDER)
                return
        
        distance = self.position.distance_to(target.position)
        
        if distance <= self.eating_range:
            # Eat the plant
            energy_gained = target.get_energy_value()
            self.gain_energy(energy_gained)
            target.on_death()
            world.remove_entity(target)
            self.state_machine.change_state(State.WANDER)
        else:
            # Move toward plant
            direction = (target.position - self.position).normalized()
            self.velocity = direction * self.max_speed * 0.8
            self.consume_energy(self.get_movement_cost(self.max_speed) * dt)
    
    def _do_mate(self, dt: float, herd: List['Herbivore'], world: 'World'):
        """Find and mate with partner."""
        mate = self.state_machine.state_data.get('mate')
        
        if mate is None or not mate.alive or not mate.can_reproduce():
            self.state_machine.change_state(State.WANDER)
            return
        
        distance = self.position.distance_to(mate.position)
        
        if distance < 2.0:
            # Reproduce
            self._reproduce(mate, world)
            self.state_machine.change_state(State.WANDER)
        else:
            # Move toward mate
            direction = (mate.position - self.position).normalized()
            self.velocity = direction * self.max_speed * 0.6
            self.consume_energy(self.get_movement_cost(self.max_speed * 0.6) * dt)
    
    def _do_herd(self, dt: float, herd: List['Herbivore']):
        """Group with other herbivores for safety."""
        if len(herd) < 2:
            self.state_machine.change_state(State.WANDER)
            return
        
        # Calculate herding forces
        separation = Vec2(0, 0)
        alignment = Vec2(0, 0)
        cohesion = Vec2(0, 0)
        
        for other in herd:
            if other.id == self.id:
                continue
            
            diff = self.position - other.position
            dist_sq = diff.length_sq()
            
            # Separation - avoid crowding
            if dist_sq < 25:  # 5 units squared
                separation = separation + diff.normalized() / math.sqrt(dist_sq + 0.1)
            
            # Alignment - match velocity
            alignment = alignment + other.velocity
            
            # Cohesion - move toward center
            cohesion = cohesion + other.position
        
        num_others = len(herd)
        if num_others > 0:
            alignment = (alignment / num_others).normalized() * self.max_speed - self.velocity
            cohesion = ((cohesion / num_others) - self.position).normalized() * self.max_speed
        
        # Combine forces
        acceleration = (separation * self.separation_weight + 
                       alignment * self.alignment_weight + 
                       cohesion * self.cohesion_weight)
        
        acceleration = acceleration.clamp_magnitude(self.max_force)
        self.velocity = (self.velocity + acceleration * dt).clamp_magnitude(self.max_speed * 0.7)
        
        self.consume_energy(self.get_movement_cost(self.velocity.length()) * dt)
    
    def _do_wander(self, dt: float):
        """Random wandering behavior."""
        # Random walk with momentum
        if rng().random() < 0.05:  # Change direction occasionally
            angle = rng().uniform(0, 2 * math.pi)
            desired = Vec2(math.cos(angle), math.sin(angle)) * self.max_speed * 0.5
            steer = (desired - self.velocity).clamp_magnitude(self.max_force * 0.5)
            self.velocity = self.velocity + steer * dt
        
        self.velocity = self.velocity.clamp_magnitude(self.max_speed * 0.5)
        self.consume_energy(self.get_movement_cost(self.velocity.length()) * dt)
    
    def _update_movement(self, dt: float, world: 'World'):
        """Apply velocity and handle boundaries."""
        movement = self.velocity * dt
        self.position = self.position + movement
        self.distance_traveled += movement.length()
        
        # Wrap around world boundaries
        self.position = world.bounds.wrap_position(self.position)
    
    def _reproduce(self, mate: 'Herbivore', world: 'World'):
        """Create offspring with partner."""
        # Both parents pay cost
        self.consume_energy(self.reproduction_cost)
        mate.consume_energy(mate.reproduction_cost)
        
        # Create child at midpoint
        child_pos = (self.position + mate.position) / 2
        child = Herbivore(child_pos, energy=self.reproduction_cost)
        
        # Inherit some traits with slight mutation
        child.max_speed = (self.max_speed + mate.max_speed) / 2 * rng().uniform(0.95, 1.05)
        child.perception_radius = (self.perception_radius + mate.perception_radius) / 2 * rng().uniform(0.95, 1.05)
        
        world.add_entity(child)
        self.children_spawned += 1
        mate.children_spawned += 1


# =============================================================================
# CARNIVORE CLASS
# =============================================================================

class Carnivore(Entity):
    """Carnivore - hunts herbivores, exhibits pack hunting behavior."""
    
    def __init__(self, position: Vec2, energy: float = 200.0):
        super().__init__(position, energy)
        self.state_machine.change_state(State.WANDER)
        
        # Movement
        self.max_speed = 5.5
        self.max_force = 3.0
        self.base_metabolism = 1.0
        
        # Behavior parameters
        self.perception_radius = 25.0
        self.attack_range = 3.0
        self.pack_radius = 20.0
        
        # State thresholds
        self.hunger_threshold = 180.0
        
        # Aging
        self.max_age = 820.0
        self.aging_start = 620.0
        self.aging_base_chance = 0.003
        self.aging_daily_increase = 0.015
        self.aging_max_chance = 0.9
        
        # Pack hunting
        self.pack_members: List['Carnivore'] = []
        self.pack_leader: Optional['Carnivore'] = None
        self.is_leader = False
        
        # Combat
        self.attack_damage = 100.0
        self.hunt_bonus = 1.0  # Increases with pack size
        
        # Memory
        self.target_prey: Optional[Herbivore] = None
        self.last_known_prey_pos: Optional[Vec2] = None
    
    def update(self, dt: float, environment: Environment, world: 'World'):
        """Update carnivore behavior."""
        super().update(dt, environment, world)
        
        if not self.alive:
            return
        
        # Sense environment
        nearby_herbivores = world.spatial_grid.query_radius(
            self.position, self.perception_radius, Herbivore
        )
        nearby_carnivores = world.spatial_grid.query_radius(
            self.position, self.pack_radius, Carnivore
        )
        
        # Filter
        nearby_herbivores = [h for h in nearby_herbivores if h.alive]
        nearby_carnivores = [c for c in nearby_carnivores if c.id != self.id and c.alive]
        
        # Update pack
        self._update_pack(nearby_carnivores)
        
        # State machine
        self._update_state_machine(nearby_herbivores, nearby_carnivores, dt)
        
        # Execute state
        self._execute_state(nearby_herbivores, nearby_carnivores, dt, world)
        
        # Update position
        self._update_movement(dt, world)
    
    def _update_pack(self, nearby_carnivores: List['Carnivore']):
        """Update pack membership and leadership."""
        self.pack_members = [c for c in nearby_carnivores if c.position.distance_to(self.position) < self.pack_radius]
        
        # Determine leader (highest energy in pack)
        if self.pack_members:
            all_pack = self.pack_members + [self]
            leader = max(all_pack, key=lambda c: c.energy)
            self.is_leader = (leader.id == self.id)
            self.pack_leader = leader if not self.is_leader else None
            
            # Calculate hunt bonus (pack synergy)
            self.hunt_bonus = 1.0 + (len(all_pack) - 1) * 0.2
        else:
            self.is_leader = True
            self.pack_leader = None
            self.hunt_bonus = 1.0
    
    def _update_state_machine(self, herbivores: List[Herbivore], pack: List['Carnivore'], dt: float):
        """Determine appropriate state."""
        current = self.state_machine.current_state
        
        # Priority 1: Hunt if hungry and prey available
        if self.energy < self.hunger_threshold and herbivores:
            if current not in [State.CHASE, State.HUNT, State.EAT]:
                # Choose between solo chase and pack hunt
                if len(self.pack_members) >= 1 and self.is_leader:
                    self.state_machine.change_state(State.HUNT, pack=self.pack_members)
                    # FIX: Set nearby members to hunt state as well
                    for member in self.pack_members:
                        if member.state_machine.current_state not in [State.CHASE, State.HUNT, State.EAT]:
                            if rng().random() < 0.8:  # 80% chance to join hunt
                                member.state_machine.change_state(State.HUNT, leader=self)
                else:
                    target = min(herbivores, key=lambda h: h.position.distance_to(self.position))
                    self.state_machine.change_state(State.CHASE, target=target)
            return
        
        # Priority 2: Mate if energy high
        if self.can_reproduce() and current not in [State.CHASE, State.HUNT, State.EAT]:
            potential_mates = [c for c in pack if c.can_reproduce()]
            if potential_mates:
                closest = min(potential_mates, key=lambda c: c.position.distance_to(self.position))
                if closest.position.distance_to(self.position) < self.perception_radius:
                    if current != State.MATE:
                        self.state_machine.change_state(State.MATE, mate=closest)
                    return
        
        # Default: Wander
        if current in [State.IDLE, State.HUNT] and self.state_machine.state_timer > 3.0:
            self.state_machine.change_state(State.WANDER)
    
    def _execute_state(self, herbivores: List[Herbivore], pack: List['Carnivore'], 
                       dt: float, world: 'World'):
        """Execute current state behavior."""
        state = self.state_machine.current_state
        
        if state == State.CHASE:
            self._do_chase(dt, herbivores, world)
        elif state == State.HUNT:
            self._do_pack_hunt(dt, herbivores, world)
        elif state == State.EAT:
            self._do_eat(dt, world)
        elif state == State.MATE:
            self._do_mate(dt, pack, world)
        elif state == State.WANDER:
            self._do_wander(dt)
    
    def _do_chase(self, dt: float, herbivores: List[Herbivore], world: 'World'):
        """Solo chase behavior."""
        target = self.state_machine.state_data.get('target')
        
        if target is None or not target.alive:
            if herbivores:
                target = min(herbivores, key=lambda h: h.position.distance_to(self.position))
                self.state_machine.state_data['target'] = target
            else:
                self.state_machine.change_state(State.WANDER)
                return
        
        distance = self.position.distance_to(target.position)
        
        if distance <= self.attack_range:
            # Attack and eat
            self._attack(target, world)
            if target.alive:
                self.state_machine.change_state(State.CHASE, target=target)
            else:
                self.state_machine.change_state(State.EAT, food=target)
        else:
            # Pursue at high speed
            direction = (target.position - self.position).normalized()
            self.velocity = direction * self.max_speed * 1.2
            self.consume_energy(self.get_movement_cost(self.max_speed * 1.2) * dt * 1.5)
    
    def _do_pack_hunt(self, dt: float, herbivores: List[Herbivore], world: 'World'):
        """Coordinated pack hunting."""
        if not self.is_leader:
            # Follow leader
            if self.pack_leader:
                direction = (self.pack_leader.position - self.position).normalized()
                self.velocity = direction * self.max_speed * 0.9
                self.consume_energy(self.get_movement_cost(self.max_speed * 0.9) * dt)
            return
        
        # Leader coordinates the hunt
        pack = self.state_machine.state_data.get('pack', [])
        
        if not herbivores:
            self.state_machine.change_state(State.WANDER)
            return
        
        # Target selection: prioritize isolated herbivores or those fleeing
        valid_targets = [h for h in herbivores if h.alive]
        if not valid_targets:
            self.state_machine.change_state(State.WANDER)
            return
        
        # Pick target
        target = min(valid_targets, key=lambda h: h.position.distance_to(self.position))
        
        # Coordinate pack to surround prey
        target_pos = target.position
        
        # Leader moves directly toward target
        direction = (target_pos - self.position).normalized()
        self.velocity = direction * self.max_speed
        
        # Check if caught
        if self.position.distance_to(target_pos) <= self.attack_range:
            self._attack(target, world)
            if target.alive:
                self.state_machine.change_state(State.HUNT, pack=self.pack_members)
            else:
                self.state_machine.change_state(State.EAT, food=target)
            return
        
        self.consume_energy(self.get_movement_cost(self.max_speed) * dt * 1.3)
    
    def _do_eat(self, dt: float, world: 'World'):
        """Consume prey."""
        food = self.state_machine.state_data.get('food')
        
        if food and food.alive:
            # Still eating
            return
        
        if food is None:
            self.state_machine.change_state(State.WANDER)
            return
        
        # Done eating or food gone
        self.state_machine.change_state(State.WANDER)
    
    def _do_mate(self, dt: float, pack: List['Carnivore'], world: 'World'):
        """Find and mate with partner."""
        mate = self.state_machine.state_data.get('mate')
        
        if mate is None or not mate.alive or not mate.can_reproduce():
            self.state_machine.change_state(State.WANDER)
            return
        
        distance = self.position.distance_to(mate.position)
        
        if distance < 3.0:
            self._reproduce(mate, world)
            self.state_machine.change_state(State.WANDER)
        else:
            direction = (mate.position - self.position).normalized()
            self.velocity = direction * self.max_speed * 0.5
            self.consume_energy(self.get_movement_cost(self.max_speed * 0.5) * dt)
    
    def _do_wander(self, dt: float):
        """Patrol behavior."""
        if rng().random() < 0.03:
            angle = rng().uniform(0, 2 * math.pi)
            desired = Vec2(math.cos(angle), math.sin(angle)) * self.max_speed * 0.6
            steer = (desired - self.velocity).clamp_magnitude(self.max_force * 0.3)
            self.velocity = self.velocity + steer * dt
        
        self.velocity = self.velocity.clamp_magnitude(self.max_speed * 0.6)
        self.consume_energy(self.get_movement_cost(self.velocity.length()) * dt)
    
    def _attack(self, prey: Herbivore, world: 'World'):
        """Attack and kill prey."""
        damage = self.attack_damage * self.hunt_bonus
        
        # Calculate how much energy we can take
        # We take either the damage dealt or what's left of prey
        energy_available = min(prey.energy, damage)
        energy_to_gain = min(energy_available * 0.8, self.max_energy - self.energy)
        
        # Prey tries to flee but takes damage
        prey.consume_energy(damage)
        
        # Predator gains energy even if prey survives (no energy leak)
        self.gain_energy(energy_to_gain)
        
        if not prey.alive:
            # Successful kill - log it
            prey.on_death()
            world.remove_entity(prey)
            world.log_event(f"Carnivore {self.id} killed Herbivore {prey.id}")
    
    def _update_movement(self, dt: float, world: 'World'):
        """Apply velocity and handle boundaries."""
        movement = self.velocity * dt
        self.position = self.position + movement
        self.distance_traveled += movement.length()
        
        self.position = world.bounds.wrap_position(self.position)
    
    def _reproduce(self, mate: 'Carnivore', world: 'World'):
        """Create offspring."""
        self.consume_energy(self.reproduction_cost)
        mate.consume_energy(mate.reproduction_cost)
        
        child_pos = (self.position + mate.position) / 2
        child = Carnivore(child_pos, energy=self.reproduction_cost)
        
        # Inherit traits
        child.max_speed = (self.max_speed + mate.max_speed) / 2 * rng().uniform(0.95, 1.05)
        child.attack_damage = (self.attack_damage + mate.attack_damage) / 2 * rng().uniform(0.95, 1.05)
        
        world.add_entity(child)
        self.children_spawned += 1
        mate.children_spawned += 1


# =============================================================================
# WORLD SIMULATION
# =============================================================================

class World:
    """Main simulation world."""
    
    def __init__(self, bounds: Bounds, seed: int = 42):
        set_seed(seed)
        
        self.bounds = bounds
        self.environment = Environment()
        self.spatial_grid = SpatialGrid(bounds, cell_size=15.0)
        
        self.entities: List[Entity] = []
        self.plants: List[Plant] = []
        self.herbivores: List[Herbivore] = []
        self.carnivores: List[Carnivore] = []
        
        self.time = 0.0
        self.day = 0
        
        # Statistics
        self.stats_history: List[Dict] = []
        self.stat_interval = 1.0
        self.stat_timer = 0.0
        self.daily_events: List[str] = []
    
    def log_event(self, message: str):
        """Log a simulation event."""
        self.daily_events.append(message)
    
    def add_entity(self, entity: Entity):
        """Add entity to world."""
        self.entities.append(entity)
        self.spatial_grid.insert(entity)
        
        if isinstance(entity, Plant):
            self.plants.append(entity)
        elif isinstance(entity, Herbivore):
            self.herbivores.append(entity)
        elif isinstance(entity, Carnivore):
            self.carnivores.append(entity)
    
    def remove_entity(self, entity: Entity):
        """Remove entity from world."""
        self.spatial_grid.remove(entity)
        
        if entity in self.entities:
            self.entities.remove(entity)
        if isinstance(entity, Plant) and entity in self.plants:
            self.plants.remove(entity)
        if isinstance(entity, Herbivore) and entity in self.herbivores:
            self.herbivores.remove(entity)
        if isinstance(entity, Carnivore) and entity in self.carnivores:
            self.carnivores.remove(entity)
    
    def initialize(self, num_plants: int = 100, num_herbivores: int = 30, num_carnivores: int = 8):
        """Initialize world with starting population."""
        # Create plants
        for _ in range(num_plants):
            pos = self.bounds.random_position()
            plant = Plant(pos, energy=rng().uniform(30, 60))
            self.add_entity(plant)
        
        # Create herbivores
        for _ in range(num_herbivores):
            pos = self.bounds.random_position()
            herbivore = Herbivore(pos, energy=rng().uniform(100, 150))
            self.add_entity(herbivore)
        
        # Create carnivores
        for _ in range(num_carnivores):
            pos = self.bounds.random_position()
            carnivore = Carnivore(pos, energy=rng().uniform(150, 200))
            self.add_entity(carnivore)
    
    def update(self, dt: float):
        """Update simulation by time step."""
        old_day = self.day
        self.time += dt
        self.day = int(self.time)
        
        # New day processing
        if self.day > old_day:
            self.daily_events = []
            if self.environment.active_event != WeatherEvent.NONE:
                self.log_event(f"Environmental Event: {self.environment.active_event.name}")

        # Update environment
        self.environment.update(dt)
        
        # Update dynamic reproduction thresholds
        if self.day > old_day:
            self._update_dynamic_reproduction()
        
        # Update spatial grid
        self.spatial_grid.clear()
        for entity in self.entities:
            if entity.alive:
                self.spatial_grid.insert(entity)
        
        # Update all entities
        for entity in self.entities[:]:
            if entity.alive:
                entity.update(dt, self.environment, self)
            else:
                self.remove_entity(entity)
        
        # Collect statistics
        self.stat_timer += dt
        if self.stat_timer >= self.stat_interval:
            self._collect_stats()
            self.stat_timer = 0.0
    
    def _update_dynamic_reproduction(self):
        """Adjust reproduction thresholds based on population pressure."""
        plant_count = len(self.plants)
        herbivore_count = len(self.herbivores)
        carnivore_count = len(self.carnivores)
        
        # Plants: boost reproduction if population is low
        plant_threshold_multiplier = 1.0
        if plant_count < 100:
            plant_threshold_multiplier = 0.6
        elif plant_count < 200:
            plant_threshold_multiplier = 0.8
        
        # Herbivores: reduce reproduction if crowded
        herbivore_threshold_multiplier = 1.0
        if herbivore_count > 60:
            herbivore_threshold_multiplier = 1.4
        elif herbivore_count > 40:
            herbivore_threshold_multiplier = 1.2
        
        # Carnivores: reduce reproduction if crowded
        carnivore_threshold_multiplier = 1.0
        if carnivore_count > 20:
            carnivore_threshold_multiplier = 1.5
        elif carnivore_count > 12:
            carnivore_threshold_multiplier = 1.25
        
        for plant in self.plants:
            plant.reproduction_threshold = plant.base_reproduction_threshold * plant_threshold_multiplier
        for herbivore in self.herbivores:
            herbivore.reproduction_threshold = herbivore.base_reproduction_threshold * herbivore_threshold_multiplier
        for carnivore in self.carnivores:
            carnivore.reproduction_threshold = carnivore.base_reproduction_threshold * carnivore_threshold_multiplier
        
        # Log key changes
        if plant_threshold_multiplier < 1.0:
            self.log_event("Low plant population: boosted plant reproduction")
        if herbivore_threshold_multiplier > 1.0:
            self.log_event("High herbivore population: reduced herbivore reproduction")
        if carnivore_threshold_multiplier > 1.0:
            self.log_event("High carnivore population: reduced carnivore reproduction")

    def _collect_stats(self):
        """Collect simulation statistics."""
        stats = {
            'time': self.time,
            'day': self.day,
            'season': self.environment.get_season_name(),
            'event': self.environment.get_event_name(),
            'plants': len(self.plants),
            'herbivores': len(self.herbivores),
            'carnivores': len(self.carnivores),
            'total_entities': len(self.entities),
            'avg_plant_energy': sum(p.energy for p in self.plants) / max(1, len(self.plants)),
            'avg_herbivore_energy': sum(h.energy for h in self.herbivores) / max(1, len(self.herbivores)),
            'avg_carnivore_energy': sum(c.energy for c in self.carnivores) / max(1, len(self.carnivores)),
        }
        self.stats_history.append(stats)
    
    def get_population_stats(self) -> Dict:
        """Get current population statistics."""
        return {
            'plants': len(self.plants),
            'herbivores': len(self.herbivores),
            'carnivores': len(self.carnivores),
            'total': len(self.entities)
        }


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

class Simulation:
    """Main simulation controller."""
    
    def __init__(self, seed: int = 42, world_size: float = 200.0):
        self.seed = seed
        self.world = World(
            Bounds(0, 0, world_size, world_size),
            seed=seed
        )
        self.timestep = FixedTimestep(dt=1.0)  # Daily updates (weekly reporting)
        
        self.running = False
        self.simulation_speed = 1.0
    
    def setup(self, plants: int = 150, herbivores: int = 40, carnivores: int = 10):
        """Setup initial simulation state."""
        print(f"Initializing ecosystem simulation with seed {self.seed}")
        print(f"World size: {self.world.bounds.max_x:.0f}x{self.world.bounds.max_y:.0f}")
        print(f"Initial population: {plants} plants, {herbivores} herbivores, {carnivores} carnivores")
        
        self.world.initialize(plants, herbivores, carnivores)
    
    def run(self, duration: float = 1095.0, print_interval: float = 7.0):
        """Run simulation for specified duration (in sim days)."""
        print(f"\n{'='*60}")
        print(f"Starting simulation for {duration:.0f} days...")
        print(f"{'='*60}\n")
        
        self.running = True
        last_print = 0.0
        
        while self.running and self.world.time < duration:
            # Calculate how many steps to process
            steps = self.timestep.step(self.simulation_speed)
            
            for _ in range(steps):
                self.world.update(self.timestep.dt)
            
            # Print status
            if self.world.time - last_print >= print_interval:
                self._print_status()
                last_print = self.world.time
            
            # Check for extinction
            if (len(self.world.herbivores) == 0 and len(self.world.carnivores) == 0):
                print("\n!!! EXTINCTION EVENT - All animals have died !!!")
                break
        
        self.running = False
        print(f"\n{'='*60}")
        print("Simulation complete!")
        print(f"{'='*60}")
        self._print_final_stats()
    
    def _print_status(self):
        """Print current simulation status."""
        env = self.world.environment
        stats = self.world.get_population_stats()
        
        print(f"Day {self.world.day:4.0f} | {env.get_season_name():8} | "
              f"Plants: {stats['plants']:4} | Herbivores: {stats['herbivores']:3} | "
              f"Carnivores: {stats['carnivores']:2} | Event: {env.get_event_name()}")
        
        # Print daily events
        if self.world.daily_events:
            # Print up to 5 random events to avoid cluttering
            events_to_show = self.world.daily_events
            if len(events_to_show) > 5:
                # We can't use random.sample here because it might break determinism 
                # if we don't use the seeded rng, but this is just for display.
                # However, to be safe and consistent, let's just show the first 5.
                events_to_show = events_to_show[:5]
                print(f"    ... and {len(self.world.daily_events) - 5} more events")
            
            for event in events_to_show:
                print(f"    > {event}")
    
    def _print_final_stats(self):
        """Print final simulation statistics."""
        stats = self.world.get_population_stats()
        
        print(f"\nFinal Population:")
        print(f"  Plants:     {stats['plants']}")
        print(f"  Herbivores: {stats['herbivores']}")
        print(f"  Carnivores: {stats['carnivores']}")
        print(f"  Total:      {stats['total']}")
        
        # Calculate some interesting stats
        all_herbivores = [e for e in self.world.entities if isinstance(e, Herbivore)]
        all_carnivores = [e for e in self.world.entities if isinstance(e, Carnivore)]
        
        if all_herbivores:
            avg_children = sum(h.children_spawned for h in all_herbivores) / len(all_herbivores)
            print(f"\nHerbivore Stats:")
            print(f"  Avg offspring: {avg_children:.2f}")
        
        if all_carnivores:
            avg_children = sum(c.children_spawned for c in all_carnivores) / len(all_carnivores)
            print(f"\nCarnivore Stats:")
            print(f"  Avg offspring: {avg_children:.2f}")
    
    def get_history(self) -> List[Dict]:
        """Get simulation history."""
        return self.world.stats_history


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the ecosystem simulation."""
    # Create simulation with fixed seed for reproducibility
    sim = Simulation(seed=12345, world_size=200.0)
    
    # Setup initial conditions
    sim.setup(
        plants=200,
        herbivores=50,
        carnivores=12
    )
    
    # Run for 3 years with weekly updates
    sim.run(duration=1095.0, print_interval=7.0)
    
    return sim


if __name__ == "__main__":
    sim = main()
