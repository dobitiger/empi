/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "BlockHelper.h"

//////////////////////////////////////////////////////////////////////////////

Block BlockHelper::create_block(PinnedArray2D<double> data, std::shared_ptr<Family> family, double scale,
                         std::shared_ptr<BlockAtomParamsConverter> converter, double booster,
                         int window_length, int output_bins, int input_shift, double subsample_offset, Extractor extractor, SpectrumCalculator& calculator, bool allow_overstep,
                         double chirp, Array1D<double> booster_multipliers) {
    auto [envelope, envelope_center_offset] = generate_envelope(family.get(), scale, subsample_offset);
    if (chirp != 0.0) {
        auto [envelope_cos, envelope_sin] = generate_chirped_envelopes(envelope, envelope_center_offset, chirp);
        auto [correctors_plus, correctors_minus] = generate_chirped_correctors(envelope, envelope_center_offset, chirp, window_length, output_bins, calculator);
        return {std::move(data), std::move(family), scale, envelope_cos, correctors_plus, std::move(converter), booster, window_length, input_shift, envelope_center_offset, extractor, allow_overstep,
                chirp, envelope_sin, correctors_minus, std::move(booster_multipliers)};
    }
    auto correctors = generate_correctors(envelope, window_length, output_bins, calculator);
    return {std::move(data), std::move(family), scale, envelope, correctors, std::move(converter), booster, window_length, input_shift, envelope_center_offset, extractor, allow_overstep,
            0.0, PinnedArray1D<double>(), PinnedArray1D<Corrector>(), std::move(booster_multipliers)};
}

std::pair<PinnedArray1D<double>, PinnedArray1D<double>> BlockHelper::generate_chirped_envelopes(const Array1D<double>& envelope, double envelope_center_offset, double chirp) {
    const index_t envelope_length = envelope.length();
    PinnedArray1D<double> envelope_cos(envelope_length), envelope_sin(envelope_length);
    for (index_t i = 0; i < envelope_length; ++i) {
        const double t = static_cast<double>(i) - envelope_center_offset;
        const double phase = M_PI * chirp * t * t;
        envelope_cos[i] = envelope[i] * std::cos(phase);
        envelope_sin[i] = envelope[i] * std::sin(phase);
    }
    return { envelope_cos, envelope_sin };
}

std::pair<PinnedArray1D<Corrector>, PinnedArray1D<Corrector>> BlockHelper::generate_chirped_correctors(const Array1D<double>& envelope, double envelope_center_offset, double chirp,
                                                                                                      int window_length, int output_bins, SpectrumCalculator& calculator) {
    // Correctors need the sum of squares of the complex-valued atom, which is (for the positive chirp rate)
    // the spectrum of envelope² × exp(−2πi·chirp·t²), computed here as a combination of two real-input transforms.
    const index_t envelope_length = envelope.length();
    Array1D<double> squared_cos(envelope_length), squared_sin(envelope_length);
    for (index_t i = 0; i < envelope_length; ++i) {
        const double t = static_cast<double>(i) - envelope_center_offset;
        const double phase = 2 * M_PI * chirp * t * t;
        const double squared = envelope[i] * envelope[i];
        squared_cos[i] = squared * std::cos(phase);
        squared_sin[i] = squared * std::sin(phase);
    }

    // calculator returns a pointer to its internal buffer, so the first spectrum has to be copied
    const int spectrum_length = window_length / 2 + 1;
    const complex *spectrum = calculator.computeSpectrum(squared_cos, window_length);
    std::vector<complex> spectrum_cos(spectrum, spectrum + spectrum_length);
    const complex *spectrum_sin = calculator.computeSpectrum(squared_sin, window_length);

    PinnedArray1D<Corrector> correctors_plus(output_bins), correctors_minus(output_bins);
    const complex I(0.0, 1.0);
    for (int k = 0; k < output_bins; ++k) {
        int k_twice = 2 * k;
        complex value_cos, value_sin;
        if (2 * k_twice > window_length) {
            value_cos = std::conj(spectrum_cos[window_length - k_twice]);
            value_sin = std::conj(spectrum_sin[window_length - k_twice]);
        } else {
            value_cos = spectrum_cos[k_twice];
            value_sin = spectrum_sin[k_twice];
        }
        correctors_plus[k] = Corrector(value_cos - I * value_sin);
        correctors_minus[k] = Corrector(value_cos + I * value_sin);
    }
    return { correctors_plus, correctors_minus };
}

std::pair<PinnedArray1D<double>, double> BlockHelper::generate_envelope(const Family* family, double scale, double subsample_offset) {
    index_t first_sample_offset, envelope_length = family->size_for_values(subsample_offset, scale, &first_sample_offset);
    double envelope_center_offset = -first_sample_offset + subsample_offset;
    PinnedArray1D<double> envelope(envelope_length);
    family->generate_values(subsample_offset, scale, nullptr, envelope.get(), true);
    return { envelope, envelope_center_offset };
}

PinnedArray1D<Corrector> BlockHelper::generate_correctors(const Array1D<double>& envelope, int window_length, int output_bins, SpectrumCalculator& calculator) {
    const index_t envelope_length = envelope.length();
    Array1D<double> envelope_squared(envelope_length);
    for (index_t i = 0; i < envelope_length; ++i) {
        envelope_squared[i] = envelope[i] * envelope[i];
    }

    PinnedArray1D<Corrector> correctors(output_bins);
    const complex *spectrum = calculator.computeSpectrum(envelope_squared, window_length);
    for (int k = 0; k < output_bins; ++k) {
        int k_twice = 2 * k;
        if (2 * k_twice > window_length) {
            correctors[k] = Corrector(std::conj(spectrum[window_length - k_twice]));
        } else {
            correctors[k] = Corrector(spectrum[k_twice]);
        }
    }
    return correctors;
}

std::list<BlockStructure> BlockHelper::compute_block_structures(const Family* family, double scale_min, double scale_max, double log_scale_step, double df_scale, double dt_scale,
                                                                const ChirpGeometry& chirp_geometry) {
    const double log_scale_min = std::log(scale_min);
    const double log_scale_max = std::log(scale_max);

    std::list<BlockStructure> result;
    if (scale_min == scale_max) {
        add_block_structures_to_list(result, family, scale_min, df_scale, dt_scale);
    } else {
        int il_max = Types::ceil<int>((log_scale_max - log_scale_min) / log_scale_step); // TODO a może floor + 1 ?
        for (int il = 0; il <= il_max; ++il) {
            const double scale = std::exp(log_scale_min + (log_scale_max - log_scale_min) * il / il_max);
            add_block_structures_to_list(result, family, scale, df_scale, dt_scale);
        }
    }

    // layers of chirped atoms, each with its own (denser) set of scales
    const int layer_count = chirp_geometry.layer_count();
    for (int layer = 1; layer <= layer_count; ++layer) {
        const double beta = chirp_geometry.layer_beta(layer);
        const double log_scale_start = std::log(chirp_geometry.layer_scale_start(layer));
        const double log_scale_range = log_scale_max - log_scale_start;
        const int il_max = (log_scale_range > 0) ? Types::ceil<int>(log_scale_range / chirp_geometry.layer_log_scale_step(layer)) : 0;
        for (int il = 0; il <= il_max; ++il) {
            const double scale = il_max ? std::exp(log_scale_start + log_scale_range * il / il_max) : std::exp(log_scale_start);
            add_block_structures_to_list(result, family, scale, df_scale, dt_scale, beta / (scale * scale), layer);
        }
    }
    return result;
}

void BlockHelper::add_block_structures_to_list(std::list<BlockStructure>& result, const Family* family, double scale, double df_scale, double dt_scale,
                                               double chirp, int chirp_layer) {
    index_t envelope_length = family->size_for_values(0.0, scale, nullptr);
    if (envelope_length > static_cast<index_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("max atom scale is too large");
    }
    const double min_transform_size_as_float = std::max<double>(scale / df_scale, static_cast<double>(envelope_length));
    int transform_size = round_transform_size(min_transform_size_as_float);
    double input_shift = dt_scale * scale;
    if (input_shift >= 1) {
        input_shift = std::floor(input_shift + 1.0e-12);
    } else {
        input_shift = 1.0 / std::ceil(1.0 / input_shift - 1.0e-12);
    }
    result.push_back({scale, static_cast<int>(envelope_length), transform_size, input_shift, chirp, chirp_layer});
}

int BlockHelper::round_transform_size(double min_transform_size_as_float) {
    if (min_transform_size_as_float <= 0) {
        throw std::logic_error("invalid transform size passed to compute_window_length");
    }
    int transform_size_bits = std::max(0, Types::ceil<int>(log2(min_transform_size_as_float)));
    if (transform_size_bits >= sizeof(int) * CHAR_BIT - 1) {
        throw std::runtime_error("FFT transform size is too large");
    }
    return 1 << transform_size_bits;
}
