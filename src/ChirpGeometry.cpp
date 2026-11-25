/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <algorithm>
#include <cmath>
#include "ChirpGeometry.h"

// sanity limit, as the size of the dictionary grows exponentially with the number of layers
static const int MAX_LAYERS = 64;

//////////////////////////////////////////////////////////////////////////////

ChirpGeometry::ChirpGeometry()
        : enabled(false), layers(0), layer_step(INFINITY), log_scale_step(1.0), chirp_max(0.0), beta_max(0.0), frequency_max(0.5),
          envelope_half_width(1.0), scale_min(1.0), scale_max(1.0) {}

ChirpGeometry::ChirpGeometry(double chirp_energy_error, double log_scale_step, double chirp_max, double frequency_max,
                             double envelope_half_width, double scale_min, double scale_max, double beta_max)
        : enabled(true), layers(0), layer_step(INFINITY), log_scale_step(log_scale_step), chirp_max(chirp_max), beta_max(beta_max), frequency_max(frequency_max),
          envelope_half_width(envelope_half_width), scale_min(scale_min), scale_max(scale_max) {
    if (chirp_energy_error > 0 && chirp_energy_error < 1) {
        // squared product between corresponding atoms of neighbouring layers is 1/cosh(Δd/2) = (1−ε²)²
        layer_step = 2 * std::acosh(std::pow(1 - chirp_energy_error, -2));
        while (layers < MAX_LAYERS && layer_scale_start(layers + 1) <= scale_max) {
            ++layers;
        }
    }
}

int ChirpGeometry::layer_count() const {
    return layers;
}

double ChirpGeometry::layer_distance(int layer) const {
    return layer ? layer * layer_step : 0.0;
}

double ChirpGeometry::layer_beta(int layer) const {
    return std::sinh(layer_distance(layer));
}

double ChirpGeometry::layer_log_scale_step(int layer) const {
    return log_scale_step / std::cosh(layer_distance(layer));
}

double ChirpGeometry::layer_scale_start(int layer) const {
    if (!layer) {
        return scale_min;
    }
    const double beta = layer_beta(layer);
    if (beta > beta_max) {
        return INFINITY;
    }
    // there must be at least some frequencies for which this β is allowed: see beta_allowed()
    return std::max({
        scale_min,
        std::sqrt(beta / chirp_max),
        beta * envelope_half_width / (0.5 * frequency_max)
    });
}

double ChirpGeometry::beta_allowed(double scale, double frequency_margin) const {
    if (!enabled) {
        return 0.0;
    }
    return std::min({
            beta_max,
            chirp_max * scale * scale,
            std::max(0.0, frequency_margin) * scale / envelope_half_width
    });
}

double ChirpGeometry::distance_allowed(double log_scale_foot, double frequency_margin) const {
    // chirplet at distance D from the geodesic of non-chirped atoms, with foot point at scale exp(λ),
    // has β = sinh(D) and scale exp(λ)·√cosh(D), so each of the three limits can be solved for D
    double result = std::asinh(beta_max);

    // sinh(D) ≤ chirp_max · exp(2λ) · cosh(D)
    const double tanh_max = chirp_max * std::exp(2 * log_scale_foot);
    if (tanh_max < 1) {
        result = std::min(result, std::atanh(tanh_max));
    }

    // sinh(D) ≤ q·√cosh(D), where q = frequency_margin · exp(λ) / envelope_half_width
    const double q = std::max(0.0, frequency_margin) * std::exp(log_scale_foot) / envelope_half_width;
    const double q2 = q * q;
    result = std::min(result, std::acosh(0.5 * (q2 + std::sqrt(q2 * q2 + 4))));

    return result;
}

double ChirpGeometry::booster_multiplier(int layer, double scale, double frequency, double frequency_step) const {
    if (!enabled) {
        return 1.0;
    }
    const double half_step = 0.5 * layer_log_scale_step(layer);
    const double D = layer_distance(layer);
    const double frequency_margin = std::min(frequency, frequency_max - frequency) + 0.5 * frequency_step;

    // Position of the atom in coordinates (λ, D), where exp(λ) is the scale of the nearest non-chirped atom ("foot point").
    // This atom is responsible for all allowed chirplets within half of step in λ and half of the distance between layers.
    const double log_scale_foot = std::log(scale) - 0.5 * std::log(std::cosh(D));
    const double D_lower = layer ? D - 0.5 * layer_step : 0.0;

    // The last atom of each layer has to cover also chirplets with larger λ:
    // for scale_max, foot points of chirplets with smaller D have larger λ than those of this layer.
    double log_scale_foot_upper = log_scale_foot + half_step;
    if (scale >= scale_max * (1 - 1.0e-9)) {
        log_scale_foot_upper = std::max(log_scale_foot_upper, std::log(scale_max) - 0.5 * std::log(std::cosh(D_lower)));
    }

    // the largest D of all allowed chirplets this atom may be responsible for (it is an increasing function of λ)
    const double D_top = distance_allowed(log_scale_foot_upper, frequency_margin);
    if (D_top < D_lower) {
        // none of the chirplets in the neighbourhood of this atom are allowed
        return 0.0;
    }
    const double D_upper = (layer < layers) ? std::min(D_top, D + 0.5 * layer_step) : D_top;

    // The first atom of each layer has to cover also chirplets with smaller λ, as the layer starts only at the scale
    // for which its own β is allowed. A lower bound for λ of allowed chirplets with D between D_lower and D_upper is used.
    double log_scale_foot_lower = log_scale_foot - half_step;
    if (scale <= layer_scale_start(layer) * (1 + 1.0e-9)) {
        double log_scale_foot_min = std::log(scale_min) - 0.5 * std::log(std::cosh(D_upper));
        if (D_lower > 0) {
            // sinh(D) ≤ chirp_max · exp(2λ) · cosh(D)
            log_scale_foot_min = std::max(log_scale_foot_min, 0.5 * std::log(std::tanh(D_lower) / chirp_max));
            // sinh(D) ≤ frequency_margin · exp(λ) · √cosh(D) / envelope_half_width
            log_scale_foot_min = std::max(
                    log_scale_foot_min,
                    std::log(std::sinh(D_lower) * envelope_half_width / frequency_margin) - 0.5 * std::log(std::cosh(D_lower))
            );
        }
        log_scale_foot_lower = std::min(log_scale_foot_lower, log_scale_foot_min);
    }

    // as cosh(distance) is a convex function of the chirplet's D, it is enough to check both ends of the interval
    const double difference = std::max(log_scale_foot_upper - log_scale_foot, log_scale_foot - log_scale_foot_lower);
    const double cosh_D = std::cosh(D), sinh_D = std::sinh(D);
    const double cosh_difference = std::cosh(2 * difference);
    double cosh_distance = 1.0;
    for (double D_chirplet : {D_lower, D_upper}) {
        cosh_distance = std::max(cosh_distance, cosh_D * std::cosh(D_chirplet) * cosh_difference - sinh_D * std::sinh(D_chirplet));
    }

    // the case of non-chirped atoms half-way (in log-scale) between two atoms of the dictionary
    // is already accounted for in the "optimality factor" of the envelope function
    const double cosh_distance_reference = std::cosh(log_scale_step);

    // squared product is equal to 1/cosh(distance/2)
    return std::sqrt((1 + cosh_distance) / (1 + cosh_distance_reference));
}
