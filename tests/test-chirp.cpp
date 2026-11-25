/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cstdio>
#include <memory>
#include "BlockAtom.h"
#include "BlockAtomObjective.h"
#include "Configuration.h"
#include "GaussianFamily.h"
#include "PinnedArray.h"
#include "Testing.h"

static const index_t N = 4096;

/**
 * Generate a single-channel signal consisting of a single chirped Gabor atom (Gaussian chirplet).
 * @return energy of the generated signal
 */
double generate_chirplet(PinnedArray2D<double> &data, const std::shared_ptr<Family> &family, const BlockAtomParams &params, double phase,
                         double amplitude) {
    const double value_max = family->value(0.0);
    double energy = 0.0;
    for (index_t i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) - params.position;
        const double arg = t / params.scale;
        const double envelope = (arg >= family->min_arg() && arg <= family->max_arg()) ? family->value(arg) / value_max : 0.0;
        data[0][i] = amplitude * envelope * std::cos(2 * M_PI * (params.frequency + 0.5 * params.chirp * t) * t + phase);
        energy += data[0][i] * data[0][i];
    }
    return energy;
}

/**
 * Objective function evaluated at exact parameters must explain the entire signal,
 * and subtracting such an atom must leave no residual.
 */
void test_exact_parameters(double frequency, double position, double scale, double beta, double phase, double amplitude) {
    auto family = std::make_shared<GaussianFamily>();
    const BlockAtomParams params(frequency, position, scale, beta / (scale * scale));
    PinnedArray2D<double> data(1, N);
    const double energy = generate_chirplet(data, family, params, phase, amplitude);

    auto converter = std::make_shared<BlockAtomParamsConverter>();
    BlockAtomObjective objective(family, data, extractorSingleChannel, converter);
    ExtraData extra_data;
    double norm;
    const double actual_energy = objective.calculate_energy(converter->arrayFromParams(params), &norm, &extra_data);
    ASSERT_APPROX(energy, actual_energy, 1.0e-8 * energy);
    ASSERT_SAME_PHASE(phase, extra_data.phase, 1.0e-8);

    Array1D<ExtraData> extra(1);
    extra[0] = extra_data;
    extra[0].amplitude *= norm * family->value(0.0);
    ASSERT_APPROX(amplitude, extra[0].amplitude, 1.0e-8);
    BlockExtendedAtom atom(data, actual_energy, family, params.frequency, params.position, params.scale, extra, params.chirp);
    atom.subtract_from_signal();
    for (index_t i = 0; i < N; ++i) {
        ASSERT_NEAR_ZERO(data[0][i]);
    }
}

/**
 * Fast (Gribonval) estimate computed from any Gabor atom sharing position and frequency with the chirplet
 * should recover both chirp rate and scale.
 */
void test_estimate(double frequency, double position, double scale, double beta, double gabor_scale) {
    auto family = std::make_shared<GaussianFamily>(3.0);
    const BlockAtomParams params(frequency, position, scale, beta / (scale * scale));
    PinnedArray2D<double> data(1, N);
    generate_chirplet(data, family, params, 0.3, 1.0);

    auto converter = std::make_shared<BlockAtomParamsConverter>();
    BlockAtomObjective objective(family, data, extractorSingleChannel, converter);
    auto estimate = objective.estimate_chirp(BlockAtomParams(frequency, position, gabor_scale));
    ASSERT(estimate);
    ASSERT_APPROX(params.chirp, estimate->chirp, 0.01 * std::abs(params.chirp) + 1.0e-9);
    ASSERT_APPROX(params.scale, estimate->scale, 0.01 * params.scale);
}

/**
 * Local optimization started from the best-matching Gabor atom (which is shorter than the chirplet,
 * and slightly misplaced as it comes from a discrete dictionary) should find the actual chirplet.
 */
void test_optimization(double frequency, double position, double scale, double beta) {
    auto family = std::make_shared<GaussianFamily>();
    const BlockAtomParams params(frequency, position, scale, beta / (scale * scale));
    PinnedArray2D<double> data(1, N);
    const double energy = generate_chirplet(data, family, params, 1.1, 2.0);

    // the Gabor atom best matching the chirplet with given β has its scale shortened by a factor of (1+β²)^¼
    const double gabor_scale = scale / std::pow(1 + beta * beta, 0.25);
    auto converter = std::make_shared<BlockAtomParamsConverter>(0.25 / gabor_scale, 0.25 * gabor_scale, 0.3, 0.5);
    BlockAtom block_atom(data, NAN, NAN, family, frequency + 0.1 / gabor_scale, position - 0.1 * gabor_scale, gabor_scale * 1.1,
                         extractorSingleChannel, converter);

    Configuration::chirp_enabled = false;
    auto gabor = std::dynamic_pointer_cast<BlockExtendedAtom>(block_atom.extend(true));
    ASSERT(gabor);
    ASSERT_EQUALS(0.0, gabor->params.chirp);
    // closed-form result for the fraction of chirplet's energy explained by the best Gabor atom
    const double expected_fraction = std::sqrt(2 / (1 + std::sqrt(1 + beta * beta)));
    ASSERT_APPROX(expected_fraction, gabor->energy / energy, 0.02);

    Configuration::chirp_enabled = true;
    auto chirplet = std::dynamic_pointer_cast<BlockExtendedAtom>(block_atom.extend(true));
    Configuration::chirp_enabled = false;
    ASSERT(chirplet);
    ASSERT_APPROX(1.0, chirplet->energy / energy, 1.0e-4);
    ASSERT_APPROX(params.chirp, chirplet->params.chirp, 0.01 * std::abs(params.chirp));
    ASSERT_APPROX(params.scale, chirplet->params.scale, 0.01 * params.scale);
    ASSERT_APPROX(params.frequency, chirplet->params.frequency, 0.01 / params.scale);
    ASSERT_APPROX(params.position, chirplet->params.position, 0.01 * params.scale);
}

/**
 * Instantaneous frequency of the fitted atoms must not leave the allowed frequency band.
 */
void test_bounds() {
    const double half_width = 1.5, frequency_max = 0.5, chirp_max = 1.0e-5;
    BlockAtomParamsConverterBounded converter(0.001, 1.0, 0.1, frequency_max, 10.0, 1000.0, 1.0, chirp_max, half_width);
    for (double frequency : {0.0, 0.01, 0.25, 0.49, 0.5}) {
        for (double scale : {10.0, 100.0, 1000.0}) {
            for (double chirp : {-1.0, -1.0e-6, 0.0, 1.0e-6, 1.0}) {
                const BlockAtomParams requested(frequency, 0.0, scale, chirp);
                auto [fixed, penalty] = converter.paramsFromArray(converter.arrayFromParams(requested));
                const double deviation = std::abs(fixed.chirp) * half_width * scale;
                ASSERT(std::abs(fixed.chirp) <= chirp_max * (1 + 1.0e-12));
                ASSERT(fixed.frequency - deviation >= -1.0e-12);
                ASSERT(fixed.frequency + deviation <= frequency_max + 1.0e-12);
                ASSERT(penalty >= 0);
                if (fixed.chirp == requested.chirp) {
                    ASSERT_EQUALS(0.0, penalty);
                }
            }
        }
    }
}

int main() {
    for (double beta : {0.0, 1.0, -3.0, 10.0}) {
        test_exact_parameters(0.2, 2000.5, 150.0, beta, 0.7, 1.5);
    }
    for (double beta : {0.5, -1.0, 3.0, -10.0}) {
        test_estimate(0.2, 2000, 150.0, beta, 150.0);
        test_estimate(0.2, 2000, 150.0, beta, 150.0 / std::pow(1 + beta * beta, 0.25));
    }
    for (double beta : {0.3, 1.0, -1.0, 3.0, -3.0, 10.0, -10.0}) {
        test_optimization(0.2, 2000, 150.0, beta);
    }
    test_bounds();
    puts("OK");
}
