#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <string>
#include <chrono>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// BoosterLandingEnv simuliert eine vereinfachte 2D-Boosterlandung.
class BoosterLandingEnv
{
public:
    BoosterLandingEnv()
    {
        dt = 0.1;         // Zeitschritt in Sekunden
        mass = 1000.0;    // Masse (kg)
        inertia = 5000.0; // Trägheitsmoment
        g = 9.81;         // Erdbeschleunigung (m/s^2)

        maxThrust = 15000.0; // Maximale Schubkraft (Newton)
        maxTorque = 5000.0;  // Maximales Drehmoment

        // Startzustand:
        // state = {x, y, vx, vy, theta, omega}
        state = {0.0, 1000.0, // x, y (Start in großer Höhe)
                 0.0, -50.0,  // vx, vy (leichte Abwärtsbewegung)
                 0.1, 0.0};   // theta (leichte Neigung), omega
    }

    // Setzt den Zustand zurück und gibt den Startzustand zurück.
    std::vector<double> reset()
    {
        state = {0.0, 1000.0, 0.0, -50.0, 0.1, 0.0};
        return state;
    }

    // Führt einen Simulationsschritt durch.
    // thrustCommand: Wert zwischen 0.0 und 1.0 (Prozent des max. Schubs)
    // torqueCommand: Wert zwischen -1.0 und 1.0 (Prozent des max. Drehmoments)
    // Rückgabe: tuple (neuer Zustand, Reward, done)
    std::tuple<std::vector<double>, double, bool>
    step(double thrustCommand, double torqueCommand)
    {
        // Eingaben begrenzen:
        thrustCommand = std::clamp(thrustCommand, 0.0, 1.0);
        torqueCommand = std::clamp(torqueCommand, -1.0, 1.0);

        double thrust = thrustCommand * maxThrust;
        double torque = torqueCommand * maxTorque;

        // Zustand auspacken:
        double x = state[0];
        double y = state[1];
        double vx = state[2];
        double vy = state[3];
        double theta = state[4]; // Boosterwinkel (0 = vertikal)
        double omega = state[5]; // Winkelgeschwindigkeit

        // Berechne Beschleunigungen:
        double ax = (thrust * sin(theta)) / mass;
        double ay = (thrust * cos(theta)) / mass - g;
        double angularAcc = torque / inertia;

        // Euler-Integration:
        vx += ax * dt;
        vy += ay * dt;
        x += vx * dt;
        y += vy * dt;
        omega += angularAcc * dt;
        theta += omega * dt;

        state = {x, y, vx, vy, theta, omega};

        // Pro Zeitschritt kleiner Strafwert:
        double reward = -1.0;
        bool done = false;

        // Prüfe, ob der Booster den Boden berührt (y <= 0)
        if (y <= 0.0)
        {
            done = true;
            y = 0.0;
            state[1] = 0.0;
            // Kriterien für eine sichere Landung:
            // - |vy| < 5 m/s, |vx| < 5 m/s, |theta| < 0.1 Radiant
            bool safeLanding =
                (std::abs(vy) < 5.0) && (std::abs(vx) < 5.0) &&
                (std::abs(theta) < 0.1);
            reward = safeLanding ? 100.0 : -100.0;
        }

        return std::make_tuple(state, reward, done);
    }

    // Getter für den Zeitschritt dt
    double getTimeStep() const { return dt; }

    // Gibt den aktuellen Zustand zurück
    const std::vector<double> &getState() const { return state; }

private:
    std::vector<double> state; // [x, y, vx, vy, theta, omega]
    double dt;
    double mass;
    double inertia;
    double g;
    double maxThrust;
    double maxTorque;
};

//
// Schnelle, headless Simulation (ohne Rendering)
// Läuft so schnell wie möglich, beispielsweise zum Training
//
void runFastSimulation()
{
    BoosterLandingEnv env;
    std::vector<double> state = env.reset();
    bool done = false;
    int stepCount = 0;

    // Beispiel-Policy: Konstanter Schub, kein Drehmoment
    double thrustCommand = 1.0;
    double torqueCommand = 0.0;

    auto startTime = std::chrono::high_resolution_clock::now();

    while (!done && stepCount < 10000)
    {
        auto result = env.step(thrustCommand, torqueCommand);
        state = std::get<0>(result);
        done = std::get<2>(result);
        stepCount++;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    std::cout << "Fast Simulation abgeschlossen in " << stepCount
              << " Schritten (" << elapsed.count() << " Sekunden).\n";

    double finalReward = std::get<1>(env.step(thrustCommand, torqueCommand));
    if (finalReward > 0)
        std::cout << "Sichere Landung!\n";
    else
        std::cout << "Harte Landung oder Absturz!\n";
}

//
// Zeigt ein Raketen-Sprite, das in Echtzeit landet.
//
void runRealTimeSimulation()
{
    BoosterLandingEnv env;
    env.reset();

    const int windowWidth = 800;
    const int windowHeight = 600;
    sf::RenderWindow window(sf::VideoMode({800, 600}),
                            "Booster Landung Simulation");
    window.setFramerateLimit(120);

    sf::Texture rocketTexture;
    if (!rocketTexture.loadFromFile("resources/rocket.png"))
    {
        std::cerr << "Fehler: resources/rocket.png konnte nicht geladen werden!\n";
        return;
    }
    sf::Sprite rocketSprite(rocketTexture);

    rocketSprite.setScale({0.2, 0.2});
    // Setze den Ursprung auf die Mitte, sodass Drehungen um die Mitte erfolgen
    sf::FloatRect bounds = rocketSprite.getLocalBounds();
    rocketSprite.setOrigin({bounds.size.x / 2, bounds.size.y / 2});

    double simulationDT = env.getTimeStep();
    double accumulator = 0.0;
    sf::Clock clock;

    // Beispielhafte Steuerung: Konstanter Schub, kein Drehmoment.
    double thrustCommand = 1.0;
    double torqueCommand = 0.0;
    bool done = false;

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
            else if (const auto *keyPressed = event->getIf<sf::Event::KeyPressed>())
            {
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape)
                    window.close();
            }
        }
        // Berechne verstrichene Zeit und simuliere in festen Schritten:
        double frameTime = clock.restart().asSeconds();
        accumulator += frameTime;
        while (accumulator >= simulationDT && !done)
        {
            std::tuple<std::vector<double>, double, bool> result = env.step(thrustCommand, torqueCommand);
            accumulator -= simulationDT;
            done = std::get<2>(result);
        }

        // Hole den aktuellen Zustand zur grafischen Darstellung:
        const std::vector<double> &state = env.getState();
        double x = state[0];
        double y = state[1];
        double theta = state[4];

        // Übersetze Simulationskoordinaten in Fensterkoordinaten:
        // x=0 in der Mitte, y=0 entspricht dem unteren Fensterrand.
        double scale = 0.5; // Skalierungsfaktor (anpassen, falls gewünscht)
        float spriteX = windowWidth / 2 + static_cast<float>(x * scale);
        float spriteY = windowHeight - static_cast<float>(y * scale);
        rocketSprite.setPosition({spriteX, spriteY});

        // Konvertiere den Winkel von Radiant in Grad (SFML erwartet Grad)
        float spriteRotation = static_cast<float>(theta * (180.0 / M_PI));
        rocketSprite.setRotation(sf::degrees(spriteRotation));

        window.clear(sf::Color::Black);
        window.draw(rocketSprite);
        window.display();

        // Wenn der Booster den Boden berührt hat:
        if (done)
        {
            std::cout << "Simulation beendet: Booster hat den Boden erreicht.\n";
            // Kurze Pause, damit das Endergebnis sichtbar bleibt:
            std::this_thread::sleep_for(std::chrono::seconds(3));
            window.close();
        }
    }
}

int main(int argc, char *argv[])
{

    runRealTimeSimulation();

    // runFastSimulation();
    return 0;
}
