/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <array>
#include <cmath>
#include <memory>
#include <vector>
#include "nelder_mead.h"
#include "BlockAtom.h"
#include "BlockAtomObjective.h"
#include "Configuration.h"
#include "Corrector.h"
#include "Logger.h"

//////////////////////////////////////////////////////////////////////////////

std::atomic<size_t> BlockAtom::failed_optimization_count;
std::atomic<size_t> BlockAtom::total_optimization_count;

BlockAtom::BlockAtom(Array2D<double> data, double energy, double energy_upper_bound, std::shared_ptr<Family> family_,
                     double frequency, double position, double scale,
                     Extractor extractor, std::shared_ptr<BlockAtomParamsConverter> converter_, double chirp)
        : BasicAtom(std::move(data), energy), energy_upper_bound(energy_upper_bound),
          extractor(extractor), family(std::move(family_)), converter(std::move(converter_)), params(frequency, position, scale, chirp) {}

void BlockAtom::connect_cache(std::shared_ptr<BlockAtomCache> cache, size_t key) {
    this->cache_slot = BlockAtomCacheSlot(std::move(cache), key);
}

ExtendedAtomPointer BlockAtom::extend(bool allow_optimization) {
    BlockAtomObjective objective(family, data, extractor, converter);

    BlockAtomParamsArray array = converter->arrayFromParams(params);
    if (allow_optimization) {
        if (cache_slot) {
            auto result = cache_slot->get();
            if (result) {
                return result;
            }
        }

        if (Configuration::chirp_enabled) {
            // Chirp rate is an additional, fourth parameter of the local optimization.
            // Apart from the non-chirped atom from the dictionary, the optimization is also started
            // from the fast estimate of the locally optimal chirp atom (Gribonval 2001), if available,
            // as strongly chirped structures may be out of reach when starting from zero chirp rate.
            std::vector<BlockAtomParamsArray> starts{array};
            if (auto estimate = objective.estimate_chirp(params)) {
                // conversion back and forth forces the estimate into the allowed range of parameters
                starts.push_back(converter->arrayFromParams(
                        converter->paramsFromArray(converter->arrayFromParams(*estimate)).first
                ));
            }
            const std::array<double, 4> step{0.5, 0.5, 0.5, 0.5};
            double best_value = INFINITY;
            for (const auto &start : starts) {
                auto result = nelder_mead<double, 4>(
                        [&objective](const std::array<double, 4> &x) {
                            return objective(x);
                        },
                        start,
                        Configuration::optimization_target,
                        step,
                        1,
                        Configuration::optimization_max_iterations
                );
                if (result.ifault) {
                    ++failed_optimization_count;
                }
                ++total_optimization_count;
                const double value = objective(result.xmin);
                if (value < best_value) {
                    best_value = value;
                    array = result.xmin;
                }
            }
        } else {
            std::array<double, 3> step{0.5, 0.5, 0.5};
            auto result = nelder_mead<double, 3>(
                    [&objective](const std::array<double, 3> &x) {
                        return objective(BlockAtomParamsArray{x[0], x[1], x[2], 0.0});
                    },
                    {array[0], array[1], array[2]},
                    Configuration::optimization_target,
                    step,
                    1,
                    Configuration::optimization_max_iterations
            );
            if (result.ifault) {
                ++failed_optimization_count;
            }
            ++total_optimization_count;
            array = {result.xmin[0], result.xmin[1], result.xmin[2], 0.0};
        }
    }

    double norm;
    const int channel_count = data.height();
    Array1D<ExtraData> extra_data(channel_count);
    double fit_energy = objective.calculate_energy(array, &norm, extra_data.get());
    BlockAtomParams fit = converter->paramsFromArray(array).first;

    const double amplitude_factor = norm * family->value(0.0);
    for (int i = 0; i < channel_count; ++i) {
        extra_data[i].amplitude *= amplitude_factor;
    }

    auto result = std::make_shared<BlockExtendedAtom>(
            data, fit_energy, family,
            fit.frequency, fit.position, fit.scale,
            std::move(extra_data), fit.chirp
    );
    if (allow_optimization && cache_slot) {
        IndexRange range = family->compute_range(result->params.position, result->params.scale);
        cache_slot->set(range, result);
    }

    return std::static_pointer_cast<ExtendedAtom>(result);
}

/**
 * @return estimate for maximum possible energy of the atom that can be obtained by locally optimizing the coefficients
 */
[[nodiscard]] double BlockAtom::get_energy_upper_bound() const {
    return energy_upper_bound;
}

//////////////////////////////////////////////////////////////////////////////

BlockExtendedAtom::BlockExtendedAtom(Array2D<double> data, double energy, std::shared_ptr<Family> family, double frequency, double position,
                                     double scale,
                                     Array1D<ExtraData> extra, double chirp)
        : ExtendedAtom(std::move(data), energy), family(std::move(family)), params(frequency, position, scale, chirp), extra(std::move(extra)) {}

void BlockExtendedAtom::export_atom(std::list<ExportedAtom> *atoms) {
    for (index_t c = 0; c < extra.length(); ++c) {
        const ExtraData &e = extra[c];
        ExportedAtom atom(e.energy);
        atom.amplitude = e.amplitude;
        atom.envelope = family->name();
        atom.frequency = params.frequency;
        atom.scale = params.scale;
        atom.phase = e.phase;
        atom.position = params.position;
        if (Configuration::chirp_enabled) {
            atom.chirp = params.chirp;
        }
        atoms[c].push_back(std::move(atom));
    }
}

IndexRange BlockExtendedAtom::subtract_from_signal() const {
    index_t first_sample_offset;
    const index_t sample_count = family->size_for_values(params.position, params.scale, &first_sample_offset);
    index_t end_sample_offset = first_sample_offset + sample_count;
    if (end_sample_offset <= 0) {
        return {0, 0};
    }

    const index_t first_valid_sample_offset = std::max<index_t>(0, first_sample_offset);
    const index_t end_valid_sample_offset = std::min<index_t>(data.length(), end_sample_offset);

    Array1D<double> samples(sample_count);
    family->generate_values(params.position, params.scale, nullptr, samples.get(), false);
    double common_amplitude_factor = 1.0 / family->value(0.0);

    const int channel_count = data.height();
    const double omega = 2 * M_PI * params.frequency;
    const double half_chirp = M_PI * params.chirp;
    for (int c = 0; c < channel_count; ++c) {
        const double channel_amplitude_factor = extra[c].amplitude * common_amplitude_factor;
        for (index_t i = first_valid_sample_offset; i < end_valid_sample_offset; ++i) {
            const double t = static_cast<double>(i) - params.position;
            data[c][i] -= channel_amplitude_factor * samples[i - first_sample_offset] *
                           std::cos((omega + half_chirp * t) * t + extra[c].phase);
        }
    }

    return {first_valid_sample_offset, end_valid_sample_offset};
}
