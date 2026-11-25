/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cstdio>
#include <list>
#include <memory>
#include <random>
#include "BlockAtom.h"
#include "BlockDictionary.h"
#include "BlockDictionaryStructure.h"
#include "ChirpGeometry.h"
#include "Configuration.h"
#include "GaussianFamily.h"
#include "PinnedArray.h"
#include "SpectrogramCalculatorFFTW.h"
#include "Testing.h"

static const index_t N = 1024;
static const double SCALE_MIN = 8.0;
static const double SCALE_MAX = 128.0;
static const double BETA_MAX = 6.0;

void test_geometry() {
    ChirpGeometry disabled;
    ASSERT(!disabled.is_enabled());
    ASSERT_EQUALS(0, disabled.layer_count());
    ASSERT_EQUALS(1.0, disabled.booster_multiplier(0, 10.0, 0.25, 0.01));

    // without any layers of chirped atoms, non-chirped atoms are responsible for all chirplets
    ChirpGeometry sparse(1.0, 0.4, INFINITY, 0.5, 1.5, SCALE_MIN, SCALE_MAX, BETA_MAX);
    ASSERT(sparse.is_enabled());
    ASSERT_EQUALS(0, sparse.layer_count());
    // for chirplets with |β| ≤ 6, the best non-chirped atom has at least √(2/(1+√(1+β²))) = 0.5309 of the energy;
    // the multiplier is slightly larger than the inverse of that (1.8836), as the step in scale matters more for chirplets
    ASSERT(sparse.booster_multiplier(0, 50.0, 0.25, 0.0) >= 1.8836);
    ASSERT(sparse.booster_multiplier(0, 50.0, 0.25, 0.0) <= 1.95);
    // close to zero and maximum frequency, no chirp is possible at all
    ASSERT_NEAR_ZERO(sparse.booster_multiplier(0, 50.0, 0.0, 0.0) - 1.0);
    ASSERT_NEAR_ZERO(sparse.booster_multiplier(0, 50.0, 0.5, 0.0) - 1.0);

    ChirpGeometry dense(0.05, 0.4, INFINITY, 0.5, 1.5, SCALE_MIN, SCALE_MAX, BETA_MAX);
    ASSERT(dense.layer_count() >= 2);
    for (int layer = 1; layer <= dense.layer_count(); ++layer) {
        ASSERT(dense.layer_beta(layer) <= BETA_MAX);
        ASSERT(dense.layer_beta(layer) > dense.layer_beta(layer - 1));
        ASSERT(dense.layer_scale_start(layer) >= dense.layer_scale_start(layer - 1));
        ASSERT(dense.layer_scale_start(layer) <= SCALE_MAX);
        ASSERT(dense.layer_log_scale_step(layer) < dense.layer_log_scale_step(layer - 1));
    }
    ASSERT(dense.layer_beta(dense.layer_count() + 1) > BETA_MAX);
    // with layers of chirped atoms, each atom is responsible only for its nearest neighbourhood
    ASSERT(dense.booster_multiplier(0, 50.0, 0.25, 0.0) < 1.1);
    ASSERT(dense.booster_multiplier(1, 50.0, 0.25, 0.0) < 1.1);
}

/**
 * For random chirplets with allowed parameters, check that
 * 1) there is at least one atom in the dictionary for which the upper bound for energy exceeds the energy of the chirplet,
 * 2) local optimization started from such candidates recovers the chirplet.
 * These two properties are what makes the global optimization mode find the best chirplet in each iteration.
 */
void test_dictionary(double chirp_energy_error, int trials, unsigned seed) {
    auto family = std::make_shared<GaussianFamily>();
    BlockDictionaryStructure structure(family, 0.05, SCALE_MIN, SCALE_MAX, 0.5, true, INFINITY, chirp_energy_error, BETA_MAX);

    PinnedArray2D<double> data(1, N);
    SpectrogramCalculatorFFTW calculator(1, structure.get_transform_sizes());
    BlockDictionary dictionary(structure, data, extractorSingleChannel, calculator, true);

    const double half_width = family->max_arg();
    const double value_max = family->value(0.0);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (int trial = 0; trial < trials; ++trial) {
        const double scale = 10.0 * std::pow(10.0, uniform(generator));
        const double frequency = 0.03 + 0.44 * uniform(generator);
        const double position = 400.0 + 200.0 * uniform(generator);
        const double phase = 2 * M_PI * uniform(generator);
        const double beta_allowed = std::min(BETA_MAX, std::min(frequency, 0.5 - frequency) * scale / half_width);
        const double beta = beta_allowed * (2 * uniform(generator) - 1);
        const double chirp = beta / (scale * scale);

        double energy = 0.0;
        for (index_t i = 0; i < N; ++i) {
            const double t = static_cast<double>(i) - position;
            const double arg = t / scale;
            const double envelope = (std::abs(arg) <= half_width) ? family->value(arg) / value_max : 0.0;
            data[0][i] = envelope * std::cos(2 * M_PI * (frequency + 0.5 * chirp * t) * t + phase);
            energy += data[0][i] * data[0][i];
        }

        std::list<SpectrogramRequest> requests;
        dictionary.fetch_requests({0, N}, requests);
        for (const auto &request : requests) {
            calculator.compute(request);
            request.interface->notify();
        }

        BasicAtomPointer best_match = dictionary.get_best_match();
        ASSERT(best_match);
        ASSERT(best_match->energy <= energy * (1 + 1.0e-9));

        auto candidates = dictionary.get_candidate_matches(energy);
        if (candidates.empty()) {
            errx(EXIT_FAILURE, "no candidates for chirplet with scale=%lf frequency=%lf beta=%lf (best match has %.1lf%% of energy)",
                 scale, frequency, beta, 100 * best_match->energy / energy);
        }

        double best_energy = 0.0;
        for (const auto &candidate : candidates) {
            best_energy = std::max(best_energy, candidate->extend(true)->energy);
        }
        if (best_energy < energy * (1 - 1.0e-3)) {
            errx(EXIT_FAILURE, "optimization reached only %.2lf%% of energy for chirplet with scale=%lf frequency=%lf beta=%lf",
                 100 * best_energy / energy, scale, frequency, beta);
        }
    }
}

int main() {
    test_geometry();

    Configuration::chirp_enabled = true;
    test_dictionary(0.05, 40, 1);
    test_dictionary(0.3, 40, 2);
    test_dictionary(1.0, 40, 3);
    Configuration::chirp_enabled = false;
    puts("OK");
}
