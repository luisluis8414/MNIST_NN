#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>

int main()
{
    sf::RenderWindow window(sf::VideoMode::getFullscreenModes()[0],
                            "SFML Rocket");
    window.setFramerateLimit(60);

    sf::Texture rocketTexture;
    if (!rocketTexture.loadFromFile("resources/rocket.png"))
    {
        std::cerr << "Failed to load rocket texture!" << std::endl;
        return 1;
    }

    sf::Sprite rocketSprite(rocketTexture);
    rocketSprite.setTexture(rocketTexture);
    rocketSprite.setPosition(
        {window.getSize().x / 2.0f - rocketTexture.getSize().x / 2.0f,
         window.getSize().y / 2.0f - rocketTexture.getSize().y / 2.0f});
    rocketSprite.setScale({0.5f, 0.5f});
    rocketSprite.setOrigin({rocketTexture.getSize().x / 2.0f,
                            rocketTexture.getSize().y / 2.0f});

    float maxSpeed = 200.0f;
    float acceleration = 100.0f;
    float deceleration = 50.0f;
    float gravity = 50.0f;
    float rotationSpeed = 200.0f;
    sf::Vector2f velocity(0.0f, 0.0f);
    float rotation = 0.0f;

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        float deltaTime = 1.0f / 60.0f;
        sf::Vector2f currentAcceleration(0.0f, 0.0f);
        float currentRotation = 0.0f;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up))
        {
            float angleRad = sf::degrees(rotation).asRadians();
            currentAcceleration.x += acceleration * std::sin(angleRad);
            currentAcceleration.y -= acceleration * std::cos(angleRad);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left))
        {
            currentRotation -= rotationSpeed;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right))
        {
            currentRotation += rotationSpeed;
        }

        // Apply gravity
        currentAcceleration.y += gravity;

        // Apply acceleration
        velocity += currentAcceleration * deltaTime;

        // Deceleration (only for X)
        if (currentAcceleration.x == 0.0f)
        {
            if (velocity.x > 0.0f)
            {
                velocity.x -= deceleration * deltaTime;
                if (velocity.x < 0.0f)
                    velocity.x = 0.0f;
            }
            else if (velocity.x < 0.0f)
            {
                velocity.x += deceleration * deltaTime;
                if (velocity.x > 0.0f)
                    velocity.x = 0.0f;
            }
        }

        // Limit speed
        if (velocity.x > maxSpeed)
        {
            velocity.x = maxSpeed;
        }
        else if (velocity.x < -maxSpeed)
        {
            velocity.x = -maxSpeed;
        }

        if (velocity.y > maxSpeed)
        {
            velocity.y = maxSpeed;
        }
        else if (velocity.y < -maxSpeed)
        {
            velocity.y = -maxSpeed;
        }

        rocketSprite.move(velocity * deltaTime);

        rotation += currentRotation * deltaTime * 0.01f;
        rocketSprite.setRotation(sf::degrees(rotation));

        window.clear();

        window.draw(rocketSprite);

        window.display();
    }

    return 0;
}
