#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>

struct IMUData {
    double yaw;  // radians
    double vx;   // velocity_north (m/s)
    double vy;   // velocity_east (m/s)
    double omega = 0.0; // yaw rate (rad/s)
};

class INSDataHandler {
public:
    bool loadFromCSV(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        std::string line;
        std::getline(file, line);  // Skip header

        std::vector<std::pair<uint64_t, IMUData>> raw;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;

            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            if (tokens.size() < 15) continue;

            uint64_t timestamp = std::stoull(tokens[0]);         // nanoseconds
            double vel_north = std::stod(tokens[9]);             // m/s
            double vel_east  = std::stod(tokens[10]);            // m/s
            double yaw       = std::stod(tokens[14]);            // radians

            raw.emplace_back(timestamp, IMUData{yaw, vel_north, vel_east});
        }

        // Compute yaw rate (omega)
        for (size_t i = 1; i < raw.size(); ++i) {
            uint64_t t1 = raw[i - 1].first;
            uint64_t t2 = raw[i].first;
            double dt = (t2 - t1) * 1e-9;  // seconds

            if (dt < 1e-4) continue;

            double yaw1 = raw[i - 1].second.yaw;
            double yaw2 = raw[i].second.yaw;

            // Handle wrap-around
            if (std::fabs(yaw2 - yaw1) > M_PI) {
                if (yaw2 > yaw1) yaw1 += 2.0 * M_PI;
                else             yaw2 += 2.0 * M_PI;
            }

            double omega = (yaw2 - yaw1) / dt;
            raw[i].second.omega = omega;
        }

        // Set first omega to second's to avoid discontinuity
        if (raw.size() >= 2) raw[0].second.omega = raw[1].second.omega;

        // Transfer to map
        for (const auto& [t, d] : raw) {
            ins_data[t] = d;
        }

        return true;
    }

    IMUData getInterpolated(uint64_t timestamp) const {
        auto it_high = ins_data.lower_bound(timestamp);

        if (it_high == ins_data.end()) return std::prev(ins_data.end())->second;
        if (it_high == ins_data.begin()) return it_high->second;

        auto it_low = std::prev(it_high);

        uint64_t t1 = it_low->first;
        uint64_t t2 = it_high->first;
        const auto& d1 = it_low->second;
        const auto& d2 = it_high->second;

        double alpha = static_cast<double>(timestamp - t1) / static_cast<double>(t2 - t1);

        auto lerp = [&](double a, double b) { return a + alpha * (b - a); };

        IMUData result;
        result.vx = lerp(d1.vx, d2.vx);
        result.vy = lerp(d1.vy, d2.vy);
        result.omega = lerp(d1.omega, d2.omega);

        double yaw1 = d1.yaw;
        double yaw2 = d2.yaw;

        // Handle yaw wrap
        if (std::fabs(yaw2 - yaw1) > M_PI) {
            if (yaw2 > yaw1) yaw1 += 2.0 * M_PI;
            else             yaw2 += 2.0 * M_PI;
        }

        result.yaw = lerp(yaw1, yaw2);

        // Wrap back to [-π, π]
        if (result.yaw > M_PI) result.yaw -= 2.0 * M_PI;
        if (result.yaw < -M_PI) result.yaw += 2.0 * M_PI;

        return result;
    }

    double getYaw(uint64_t timestamp) const {
        return getInterpolated(timestamp).yaw;
    }

private:
    std::map<uint64_t, IMUData> ins_data;
};
