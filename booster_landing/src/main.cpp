#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct State
{
    float pos_x;
    float pos_y;
    float vel_x;
    float vel_y;
    float angle; // In radians
    float ang_vel;
    float fuel;       // Remaining fuel percentage
    float wind_speed; // Current wind speed
};

class BoosterLandingEnv
{
public:
    // Constants for simulation
    static constexpr float DEFAULT_TIME_SCALE = 1.0f;
    static constexpr int WINDOW_WIDTH = 1920;
    static constexpr int WINDOW_HEIGHT = 1080;
    static constexpr float INITIAL_HEIGHT = 8000.0f;        // 8km starting height
    static constexpr float INITIAL_LATERAL_RANGE = 2000.0f; // ±2km lateral range
    static constexpr float GRAVITY = 9.81f;
    static constexpr float AIR_DENSITY = 1.225f; // kg/m³ at sea level
    static constexpr float DRAG_COEFFICIENT = 0.2f;
    static constexpr float ROCKET_WIDTH = 3.7f;          // meters
    static constexpr float ROCKET_HEIGHT = 70.0f;        // meters
    static constexpr float SCALE_FACTOR = 0.1f;          // pixels per meter
    static constexpr float LANDING_PAD_WIDTH = 50.0f;    // meters
    static constexpr float FUEL_CONSUMPTION_RATE = 0.1f; // % per second at max thrust

    BoosterLandingEnv(bool render_mode = false, float time_scale = DEFAULT_TIME_SCALE)
        : render_mode(render_mode),
          dt(0.02f),
          gravity(GRAVITY),
          dry_mass(25000.0f),  // Dry mass in kg
          fuel_mass(15000.0f), // Initial fuel mass in kg
          moment_of_inertia(2500000.0f),
          max_thrust(845000.0f), // Merlin 1D engine thrust in N
          max_torque(5000000.0f),
          wind_base_dist(-2.0f, 2.0f),   // Base wind in m/s
          wind_gust_dist(0.0f, 5.0f),    // Wind gusts in m/s
          altitude_wind_factor(0.0001f), // Wind increases with altitude
          initial_position_dist(-INITIAL_LATERAL_RANGE, INITIAL_LATERAL_RANGE),
          initial_velocity_dist(-50.0f, 50.0f),
          random_engine(std::random_device{}()),
          rocket_sprite(rocket_texture),
          time_scale(time_scale)
    {
        if (render_mode)
        {
            window = new sf::RenderWindow(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}),
                                          "Falcon 9 Landing Simulation",
                                          sf::Style::Default);

            // Load rocket texture
            if (!rocket_texture.loadFromFile("resources/rocket.png"))
            {
                std::cerr << "Error loading rocket texture" << std::endl;
                // Fallback to a rectangle if texture loading fails
                rocket_shape.setSize(sf::Vector2f(ROCKET_WIDTH * SCALE_FACTOR,
                                                  ROCKET_HEIGHT * SCALE_FACTOR));
                rocket_shape.setOrigin({ROCKET_WIDTH * SCALE_FACTOR / 2.0f,
                                        ROCKET_HEIGHT * SCALE_FACTOR / 2.0f});
                rocket_shape.setFillColor(sf::Color::White);
                use_sprite = false;
            }
            else
            {
                rocket_sprite.setTexture(rocket_texture);
                sf::Vector2u textureSize = rocket_texture.getSize();
                rocket_sprite.setOrigin({textureSize.x / 2.0f, textureSize.y / 2.0f});
                float sprite_scale = (ROCKET_HEIGHT * SCALE_FACTOR) / textureSize.y;
                rocket_sprite.setScale({sprite_scale, sprite_scale});
                use_sprite = true;
            }
        }
        reset();
    }

    ~BoosterLandingEnv()
    {
        if (window)
        {
            delete window;
            window = nullptr;
        }
    }

    State current_state()
    {
        return {
            position.x,
            position.y,
            velocity.x,
            velocity.y,
            angle,
            angular_velocity,
            fuel_percentage,
            current_wind_speed};
    }

    void reset()
    {
        // Random initial conditions
        position.x = initial_position_dist(random_engine);
        position.y = INITIAL_HEIGHT;
        velocity.x = initial_velocity_dist(random_engine);
        velocity.y = 0.0f;
        angle = 0.0f;
        angular_velocity = 0.0f;
        fuel_percentage = 100.0f;
        current_wind_speed = wind_base_dist(random_engine);
        base_wind = current_wind_speed;
    }

    std::tuple<State, float, bool> step(float throttle, float torque_control)
    {
        float scaled_dt = dt * time_scale;
        // Clamp inputs
        throttle = std::clamp(throttle, 0.0f, 1.0f);
        torque_control = std::clamp(torque_control, -1.0f, 1.0f);

        if (fuel_percentage <= 0)
        {
            throttle = 0;
        }

        // Update wind conditions
        updateWind();

        // Calculate current mass
        float current_mass = dry_mass + (fuel_mass * fuel_percentage / 100.0f);

        // Calculate thrust and fuel consumption
        float thrust = throttle * max_thrust;
        fuel_percentage -= throttle * FUEL_CONSUMPTION_RATE * scaled_dt;
        fuel_percentage = std::max(0.0f, fuel_percentage);

        // Calculate forces
        // Thrust forces
        float thrust_x = thrust * std::sin(angle);
        float thrust_y = -thrust * std::cos(angle);

        // Air resistance
        float air_density_at_altitude = AIR_DENSITY * std::exp(-position.y / 7400.0f); // Atmospheric scale height
        float velocity_magnitude = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
        float drag = 0.5f * air_density_at_altitude * velocity_magnitude * velocity_magnitude *
                     DRAG_COEFFICIENT * ROCKET_WIDTH * ROCKET_HEIGHT;

        float drag_x = velocity.x != 0 ? -drag * velocity.x / velocity_magnitude : 0;
        float drag_y = velocity.y != 0 ? -drag * velocity.y / velocity_magnitude : 0;

        // Wind force
        float wind_force = 0.5f * air_density_at_altitude *
                           current_wind_speed * std::abs(current_wind_speed) *
                           DRAG_COEFFICIENT * ROCKET_HEIGHT;

        // Total forces
        float force_x = thrust_x + drag_x + wind_force;
        float force_y = thrust_y + drag_y + (current_mass * gravity);

        // Update linear dynamics (RK4 integration could be used for better accuracy)
        float acc_x = force_x / current_mass;
        float acc_y = force_y / current_mass;

        velocity.x += acc_x * scaled_dt;
        velocity.y += acc_y * scaled_dt;
        position.x += velocity.x * scaled_dt;
        position.y += velocity.y * scaled_dt;

        // Update rotational dynamics
        float applied_torque = torque_control * max_torque;
        float angular_acc = applied_torque / moment_of_inertia;
        angular_velocity += angular_acc * scaled_dt;
        angle += angular_velocity * scaled_dt;

        // Ground collision check
        bool done = false;
        const float ground_y = WINDOW_HEIGHT - 50.0f;
        const float landing_pad_x = WINDOW_WIDTH / 2.0f;
        const float landing_pad_min = landing_pad_x - (LANDING_PAD_WIDTH * SCALE_FACTOR / 2.0f);
        const float landing_pad_max = landing_pad_x + (LANDING_PAD_WIDTH * SCALE_FACTOR / 2.0f);

        if (position.y >= ground_y)
        {
            done = true;
            position.y = ground_y;
            velocity = {0, 0};
            angular_velocity = 0;
        }

        // Calculate reward
        float distance_to_pad = std::abs(position.x - landing_pad_x);
        float velocity_penalty = (velocity.x * velocity.x + velocity.y * velocity.y) / 100.0f;
        float angle_penalty = std::abs(angle) * 10.0f;
        float base_reward = -(distance_to_pad * 0.01f + velocity_penalty + angle_penalty);

        if (done)
        {
            bool good_landing = position.x >= landing_pad_min &&
                                position.x <= landing_pad_max &&
                                std::abs(velocity.x) < 2.0f &&
                                std::abs(velocity.y) < 2.0f &&
                                std::abs(angle) < 0.1f;

            if (good_landing)
            {
                base_reward += 5000.0f * fuel_percentage / 100.0f; // Bonus for efficient landing
            }
            else
            {
                base_reward -= 5000.0f;
            }
        }

        return {current_state(), base_reward, done};
    }

    void updateWind()
    {
        // Update base wind slowly
        base_wind += wind_base_dist(random_engine) * dt;
        base_wind = std::clamp(base_wind, -20.0f, 20.0f);

        // Add gusts and altitude effects
        float gust = wind_gust_dist(random_engine) * dt;
        float altitude_effect = position.y * altitude_wind_factor;
        current_wind_speed = base_wind + gust + altitude_effect;
    }

    void setTimeScale(float new_scale)
    {
        time_scale = new_scale;
    }

    float getTimeScale() const
    {
        return time_scale;
    }

    void render()
    {
        if (!render_mode || !window)
            return;

        // Handle window events
        while (const std::optional event = window->pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window->close();
        }

        window->clear(sf::Color(20, 20, 40)); // Dark blue background

        // Draw sky gradient
        sf::RectangleShape sky(sf::Vector2f(WINDOW_WIDTH, WINDOW_HEIGHT));
        sky.setFillColor(sf::Color(20, 20, 40));
        window->draw(sky);

        // Draw ground
        sf::RectangleShape ground(sf::Vector2f(WINDOW_WIDTH, 100));
        ground.setFillColor(sf::Color(70, 70, 70));
        ground.setPosition({0, WINDOW_HEIGHT - 50});
        window->draw(ground);

        // Draw landing pad
        sf::RectangleShape landing_pad(sf::Vector2f(LANDING_PAD_WIDTH * SCALE_FACTOR, 10));
        landing_pad.setFillColor(sf::Color::Red);
        landing_pad.setPosition({WINDOW_WIDTH / 2.0f - (LANDING_PAD_WIDTH * SCALE_FACTOR / 2.0f),
                                 WINDOW_HEIGHT - 55});
        window->draw(landing_pad);

        // Draw wind indicator
        drawWindIndicator();

        // Draw rocket
        if (use_sprite)
        {
            rocket_sprite.setPosition({position.x, position.y});
            rocket_sprite.setRotation(sf::degrees(angle * 180 / M_PI));
            window->draw(rocket_sprite);
        }
        else
        {
            rocket_shape.setPosition({position.x, position.y});
            rocket_shape.setRotation(sf::degrees(angle * 180 / M_PI));
            window->draw(rocket_shape);
        }

        // Draw engine flame when throttle is active
        drawEngineFlame();

        // Draw telemetry
        drawTelemetry();

        window->display();
    }

    bool isWindowOpen() const
    {
        return (render_mode && window) ? window->isOpen() : true;
    }

private:
    void drawWindIndicator()
    {
        // Draw wind speed indicator
        sf::RectangleShape wind_arrow(sf::Vector2f(50 * std::abs(current_wind_speed) / 20.0f, 10));
        wind_arrow.setPosition({50, 50});
        wind_arrow.setFillColor(sf::Color(200, 200, 255, 128));

        if (current_wind_speed < 0)
        {
            wind_arrow.setRotation(sf::degrees(180));
        }

        window->draw(wind_arrow);

        // Draw wind speed text
        if (!telemetry_font.openFromFile("resources/fonts/arial.ttf"))
        {
            static bool reported = false;
            if (!reported)
            {
                std::cerr << "Error loading font for telemetry" << std::endl;
                reported = true;
            }
            return;
        }

        sf::Text wind_text(telemetry_font);
        wind_text.setFont(telemetry_font);
        wind_text.setCharacterSize(14);
        wind_text.setFillColor(sf::Color::White);
        wind_text.setPosition({120, 45});
        wind_text.setString("Wind: " + std::to_string(int(current_wind_speed)) + " m/s");
        window->draw(wind_text);
    }

    void drawEngineFlame()
    {
        if (fuel_percentage <= 0)
            return;

        sf::ConvexShape flame;
        flame.setPointCount(3);

        float flame_width = ROCKET_WIDTH * SCALE_FACTOR * 0.8f;
        float flame_length = ROCKET_HEIGHT * SCALE_FACTOR * 0.5f * throttle;

        flame.setPoint(0, sf::Vector2f(-flame_width / 2, 0));
        flame.setPoint(1, sf::Vector2f(flame_width / 2, 0));
        flame.setPoint(2, sf::Vector2f(0, flame_length));

        flame.setFillColor(sf::Color(255, 100, 0, 200));

        // Position the flame at the bottom of the rocket
        sf::Transform transform;
        transform.translate({position.x, position.y});
        transform.rotate(sf::degrees(angle * 180 / M_PI), sf::Vector2f(0, 0));
        transform.translate({0, ROCKET_HEIGHT * SCALE_FACTOR / 2});

        window->draw(flame, transform);
    }

    void drawTelemetry()
    {
        if (!telemetry_font.openFromFile("resources/fonts/arial.ttf"))
            return;

        sf::Text telemetry(telemetry_font);
        telemetry.setFont(telemetry_font);
        telemetry.setCharacterSize(14);
        telemetry.setFillColor(sf::Color::White);
        telemetry.setPosition({10, 10});

        std::string telemetry_text =
            "Time Scale: " + std::to_string(time_scale) + "x\n" +
            "Altitude: " + std::to_string(int(INITIAL_HEIGHT - position.y)) + " m\n" +
            "Lateral: " + std::to_string(int(position.x - WINDOW_WIDTH / 2)) + " m\n" +
            "Velocity: " + std::to_string(int(std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y))) + " m/s\n" +
            "Fuel: " + std::to_string(int(fuel_percentage)) + "%\n" +
            "Angle: " + std::to_string(int(angle * 180 / M_PI)) + " deg";

        telemetry.setString(telemetry_text);
        window->draw(telemetry);
    }

private:
    bool render_mode;
    bool use_sprite;
    sf::RenderWindow *window;
    sf::Texture rocket_texture;
    sf::Sprite rocket_sprite;
    sf::RectangleShape rocket_shape;
    sf::Font telemetry_font;

    sf::Vector2f position;
    sf::Vector2f velocity;
    float angle;
    float angular_velocity;
    float fuel_percentage;
    float current_wind_speed;
    float base_wind;
    float throttle;

    const float gravity;
    const float dry_mass;
    const float fuel_mass;
    const float moment_of_inertia;
    const float max_thrust;
    const float max_torque;
    const float dt;

    std::mt19937 random_engine;
    std::uniform_real_distribution<float> wind_base_dist;
    std::uniform_real_distribution<float> wind_gust_dist;
    std::uniform_real_distribution<float> initial_position_dist;
    std::uniform_real_distribution<float> initial_velocity_dist;
    const float altitude_wind_factor;

    float time_scale;
};

int main()
{
    // Create the environment with rendering enabled
    BoosterLandingEnv env(true, 1.0f);

    // Control variables
    float throttle = 0.0f;
    float torque_control = 0.0f;
    bool auto_hover = false;
    float target_altitude = 0.0f;

    // Performance metrics
    int total_landings = 0;
    int successful_landings = 0;

    // Main simulation loop
    while (env.isWindowOpen())
    {

        // Add time scale control
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num1))
        {
            env.setTimeScale(1.0f); // Realtime
        }
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num2))
        {
            env.setTimeScale(2.0f); // 2x speed
        }
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num3))
        {
            env.setTimeScale(5.0f); // 5x speed
        }
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num0))
        {
            env.setTimeScale(0.5f); // Half speed
        }

        // Modify the sleep time based on time scale
        float sleep_time = std::max(1.0f, 10.0f / env.getTimeScale());
        sf::sleep(sf::milliseconds(static_cast<int>(sleep_time)));

        // Manual controls
        if (!auto_hover)
        {
            // Throttle control
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up))
            {
                throttle = std::min(throttle + 0.02f, 1.0f);
            }
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down))
            {
                throttle = std::max(throttle - 0.02f, 0.0f);
            }

            // Rotation control
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left))
            {
                torque_control = -1.0f;
            }
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right))
            {
                torque_control = 1.0f;
            }
            else
            {
                torque_control = 0.0f;
            }
        }

        // Toggle auto-hover mode
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::H))
        {
            auto_hover = !auto_hover;
            target_altitude = env.current_state().pos_y;
            std::cout << "Auto-hover " << (auto_hover ? "enabled" : "disabled") << std::endl;
        }

        // Auto-hover control
        if (auto_hover)
        {
            State current = env.current_state();

            // PID control constants
            const float Kp_alt = 0.001f;
            const float Kd_alt = 0.002f;
            const float Kp_angle = 2.0f;
            const float Kd_angle = 1.0f;

            // Altitude control
            float altitude_error = target_altitude - current.pos_y;
            float vertical_speed_error = -current.vel_y;
            throttle = 0.5f + Kp_alt * altitude_error + Kd_alt * vertical_speed_error;
            throttle = std::clamp(throttle, 0.0f, 1.0f);

            // Attitude control
            float angle_error = -current.angle; // Target angle is 0 (vertical)
            float angular_velocity_error = -current.ang_vel;
            torque_control = Kp_angle * angle_error + Kd_angle * angular_velocity_error;
            torque_control = std::clamp(torque_control, -1.0f, 1.0f);
        }

        // Step the simulation
        auto [state, reward, done] = env.step(throttle, torque_control);

        // Display telemetry
        std::cout << "\rAlt: " << std::fixed << std::setprecision(1)
                  << (BoosterLandingEnv::INITIAL_HEIGHT - state.pos_y)
                  << "m | Vel: " << std::sqrt(state.vel_x * state.vel_x + state.vel_y * state.vel_y)
                  << "m/s | Fuel: " << state.fuel
                  << "% | Wind: " << state.wind_speed << "m/s"
                  << std::flush;

        // Check landing condition
        if (done)
        {
            total_landings++;
            bool success = (std::abs(state.vel_x) < 2.0f &&
                            std::abs(state.vel_y) < 2.0f &&
                            std::abs(state.angle) < 0.1f);

            if (success)
            {
                successful_landings++;
                std::cout << "\nSuccessful landing! Efficiency score: "
                          << (state.fuel * reward / 5000.0f) << std::endl;
            }
            else
            {
                std::cout << "\nCrash! Final velocity: "
                          << std::sqrt(state.vel_x * state.vel_x + state.vel_y * state.vel_y)
                          << "m/s" << std::endl;
            }

            std::cout << "Success rate: "
                      << (float)successful_landings / total_landings * 100
                      << "% (" << successful_landings << "/" << total_landings << ")\n";

            // Reset for next attempt
            env.reset();
            throttle = 0.0f;
            torque_control = 0.0f;
            auto_hover = false;
        }

        // Render the scene
        env.render();

        // Small delay to prevent excessive CPU usage
        sf::sleep(sf::milliseconds(10));
    }

    return 0;
}
