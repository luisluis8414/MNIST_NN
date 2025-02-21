#include <iostream>
#include <SFML/Graphics.hpp>

class Rocket
{
public:
    Rocket(const std::string &texturePath, float startX, float startY) : sprite_(texture_)
    {
        if (!texture_.loadFromFile(texturePath))
        {
            std::cerr << "Failed to load rocket texture!" << std::endl;
            // Handle error appropriately, maybe throw an exception
        }

        sprite_.setTexture(texture_);
        sprite_.setScale({0.5f, 0.5f});
        sprite_.setOrigin({texture_.getSize().x / 2.0f,
                           texture_.getSize().y / 2.0f});
        sprite_.setPosition({startX, startY});

        maxSpeed_ = 200.0f;
        acceleration_ = 100.0f;
        deceleration_ = 50.0f;
        rotationSpeed_ = 200.0f;
        velocity_ = sf::Vector2f(0.0f, 0.0f);
        rotation_ = 0.0f;
    }

    void update(float deltaTime)
    {
        // Apply acceleration
        velocity_ += currentAcceleration_ * deltaTime;

        // Deceleration (only for X)
        if (currentAcceleration_.x == 0.0f)
        {
            if (velocity_.x > 0.0f)
            {
                velocity_.x -= deceleration_ * deltaTime;
                if (velocity_.x < 0.0f)
                    velocity_.x = 0.0f;
            }
            else if (velocity_.x < 0.0f)
            {
                velocity_.x += deceleration_ * deltaTime;
                if (velocity_.x > 0.0f)
                    velocity_.x = 0.0f;
            }
        }

        // Limit speed
        if (velocity_.x > maxSpeed_)
        {
            velocity_.x = maxSpeed_;
        }
        else if (velocity_.x < -maxSpeed_)
        {
            velocity_.x = -maxSpeed_;
        }

        if (velocity_.y > maxSpeed_)
        {
            velocity_.y = maxSpeed_;
        }
        else if (velocity_.y < -maxSpeed_)
        {
            velocity_.y = -maxSpeed_;
        }

        // Apply velocity
        sprite_.move(velocity_ * deltaTime);

        // Apply rotation
        rotation_ += currentRotation_ * deltaTime;
        sprite_.setRotation(sf::degrees(rotation_));

        // Reset current values
        currentAcceleration_ = sf::Vector2f(0.0f, 0.0f);
        currentRotation_ = 0.0f;
    }

    void draw(sf::RenderWindow &window) { window.draw(sprite_); }

    void thrust(float acceleration)
    {
        float angleRad = sf::degrees(rotation_).asRadians();
        currentAcceleration_.x += acceleration * std::sin(angleRad);
        currentAcceleration_.y -= acceleration * std::cos(angleRad);
    }

    void rotate(float rotationSpeed) { currentRotation_ += rotationSpeed; }

    sf::Sprite &getSprite() { return sprite_; }

private:
    sf::Texture texture_;
    sf::Sprite sprite_;
    float maxSpeed_;
    float acceleration_;
    float deceleration_;
    float rotationSpeed_;
    sf::Vector2f velocity_;
    float rotation_;
    sf::Vector2f currentAcceleration_;
    float currentRotation_;
};
