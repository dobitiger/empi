/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include "BlockAtomObjective.h"

//////////////////////////////////////////////////////////////////////////////

BlockAtomObjective::BlockAtomObjective(std::shared_ptr<Family> family, Array2D<double> data_, Extractor extractor,
                                       std::shared_ptr<BlockAtomParamsConverter> converter_) :
        family(std::move(family)),
        channel_count(data_.height()),
        data(std::move(data_)),
        products(channel_count, 1),
        extractor(extractor),
        converter(std::move(converter_)) {}

double BlockAtomObjective::calculate_energy(const BlockAtomParamsArray &array, double *out_norm, ExtraData *out_extra_data) {
    auto [params, penalty] = converter->paramsFromArray(array);
    return std::exp(-penalty) * calculate_energy(params, out_norm, out_extra_data);
}

double BlockAtomObjective::calculate_energy(const BlockAtomParams& params, double *out_norm, ExtraData *out_extra_data) {
    Corrector corrector(compute_products(params, out_norm));
    double tmp_for_extractor;
    return extractor(channel_count, 1, products.get(), &corrector, &tmp_for_extractor, out_extra_data).energy;
}

std::optional<BlockAtomParams> BlockAtomObjective::estimate_chirp(const BlockAtomParams &params) {
    // The second differences below are exact for a Gaussian chirp regardless of the frequency step,
    // but the second difference of phase is known only modulo 2π. Since |Φ″| ≤ π·scale², any step
    // smaller than 1/scale resolves this ambiguity; we use half of that.
    const double frequency_step = 0.5 / params.scale;
    if (params.frequency - frequency_step <= 0 || params.frequency + frequency_step >= 0.5) {
        return std::nullopt;
    }

    // z = Σ P(f−δ)·P(f+δ)·conj(P(f))² and the corresponding sums of energies, over all channels
    complex z = 0.0;
    double energies[3] = {0.0, 0.0, 0.0};
    std::vector<complex> central(channel_count), lower(channel_count);
    for (int k = 0; k < 3; ++k) {
        BlockAtomParams shifted(params.frequency + (k - 1) * frequency_step, params.position, params.scale, params.chirp);
        compute_products(shifted, nullptr);
        for (int c = 0; c < channel_count; ++c) {
            const complex product = products[c][0];
            energies[k] += std::norm(product);
            if (k == 0) {
                lower[c] = product;
            } else if (k == 1) {
                central[c] = product;
            } else {
                z += lower[c] * product * std::conj(central[c] * central[c]);
            }
        }
    }
    if (!(energies[0] > 0 && energies[1] > 0 && energies[2] > 0) || z == 0.0) {
        return std::nullopt;
    }

    // second derivatives (with respect to frequency) of the logarithm of modulus and of the phase
    const double step2 = frequency_step * frequency_step;
    const double d2_log_modulus = 0.5 * std::log(energies[0] * energies[2] / (energies[1] * energies[1])) / step2;
    const double d2_phase = std::arg(z) / step2;

    // For signal locally equal to exp(−πt²/σ²)·exp(2πi(f₀t + ½c₁t²)) and atom of scale s and chirp rate c₀,
    // (log A)″ + iΦ″ = −2π / (p − ic), where p = 1/σ² + 1/s² and c = c₁ − c₀
    const double denominator = d2_log_modulus * d2_log_modulus + d2_phase * d2_phase;
    if (!(denominator > 0)) {
        return std::nullopt;
    }
    const double chirp = -2 * M_PI * d2_phase / denominator;
    const double p = -2 * M_PI * d2_log_modulus / denominator;
    const double inv_scale2 = p - 1.0 / (params.scale * params.scale);
    if (!(inv_scale2 > 0) || !std::isfinite(chirp)) {
        // the model is not valid here: be conservative and do not propose any chirped atom
        return std::nullopt;
    }
    return BlockAtomParams(params.frequency, params.position, 1.0 / std::sqrt(inv_scale2), params.chirp + chirp);
}

complex BlockAtomObjective::compute_products(const BlockAtomParams& params, double *out_norm) {
    index_t envelope_length = family->size_for_values(params.position, params.scale, nullptr);
    index_t envelope_offset;
    envelope.resize(envelope_length);
    oscillating.resize(envelope_length);
    double norm = family->generate_values(params.position, params.scale, &envelope_offset, envelope.data(), true);
    if (out_norm) {
        *out_norm = norm;
    }

    const double omega = 2 * M_PI * params.frequency;
    const double half_chirp = M_PI * params.chirp;
    for (index_t i=0; i < envelope_length; ++i) {
        const double t = static_cast<double>(envelope_offset + i) - params.position;
        oscillating[i] = std::polar(envelope[i], -(omega + half_chirp * t) * t);
    }

    complex FT = 0.0;
    for (index_t i = 0; i < envelope_length; ++i) {
        FT += oscillating[i] * oscillating[i];
    }
    index_t first_sample_offset = std::max<index_t>(0, envelope_offset);
    index_t last_sample_offset = std::max<index_t>(0, envelope_offset + envelope_length - 1);

    first_sample_offset = std::min(data.length() - 1, first_sample_offset);
    last_sample_offset = std::min(data.length() - 1, last_sample_offset);

    products.fill(0.0);
    for (index_t i = first_sample_offset; i <= last_sample_offset; ++i) {
        complex z = oscillating[i - envelope_offset];
        for (int c = 0; c < channel_count; ++c) {
            products[c][0] += data[c][i] * z;
        }
    }

    return FT;
}
