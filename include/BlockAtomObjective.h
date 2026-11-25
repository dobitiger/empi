/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_BLOCK_ATOM_OBJECTIVE_H
#define EMPI_BLOCK_ATOM_OBJECTIVE_H

#include <array>
#include <memory>
#include <optional>
#include <vector>
#include "Array.h"
#include "BlockAtomBase.h"
#include "Extractor.h"
#include "Family.h"

/**
 * Objective function used by multidimensional optimization to find optimal atom parameters.
 */
class BlockAtomObjective {

    std::shared_ptr<Family> family;
    const int channel_count;
    Array2D<double> data;
    Array2D<complex> products;
    Extractor extractor;
    std::vector<double> envelope;
    std::vector<complex> oscillating;
    std::shared_ptr<BlockAtomParamsConverter> converter;

public:
    BlockAtomObjective(std::shared_ptr<Family> family, Array2D<double> data, Extractor extractor, std::shared_ptr<BlockAtomParamsConverter> converter);

    double calculate_energy(const BlockAtomParamsArray &array, double *out_norm, ExtraData *out_extra_data);

    double calculate_energy(const BlockAtomParams &params, double *out_norm, ExtraData *out_extra_data);

    double operator()(const BlockAtomParamsArray &array) {
        return -calculate_energy(array, nullptr, nullptr);
    }

    /**
     * Fast estimate of the locally optimal chirp rate and scale in the neighbourhood of a given atom,
     * following the "ridge pursuit" of Gribonval (2001), https://doi.org/10.1109/78.917803
     * The estimate is based on the local behaviour of the (complex) product between signal and atom
     * as a function of atom's frequency: if the signal is locally a Gaussian chirp,
     * both logarithm of modulus and phase of this product are second-order polynomials of frequency.
     *
     * @param params parameters of the atom (usually the best match from the dictionary)
     * @return parameters of the estimated chirped atom (with the same position and frequency),
     * or empty value if the signal does not locally resemble a Gaussian chirp
     */
    std::optional<BlockAtomParams> estimate_chirp(const BlockAtomParams &params);

private:
    /**
     * Compute products of the signal (in all channels) with complex-valued atom and store them in "products".
     *
     * @return sum of squares of the complex-valued atom, needed to initialize a Corrector instance
     */
    complex compute_products(const BlockAtomParams &params, double *out_norm);
};

#endif //EMPI_BLOCK_ATOM_OBJECTIVE_H
