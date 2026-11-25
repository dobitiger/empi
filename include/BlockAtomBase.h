/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_BLOCK_ATOM_BASE_H
#define EMPI_BLOCK_ATOM_BASE_H

#include <array>
#include <cmath>
#include <utility>

/**
 * Simple structure with four fields, used in both BlockAtom and ExtendedBlockAtom.
 *
 * The oscillation of the atom is cos(2π(frequency·t + ½·chirp·t²) + phase), where t is measured from the atom's position,
 * so the instantaneous frequency is equal to frequency + chirp·t.
 */
struct BlockAtomParams {
    double frequency; // between 0 and 0.5 (1 would be sampling frequency)
    double position; // in samples
    double scale; // in samples
    double chirp; // linear chirp rate in cycles per sample squared (0 for ordinary, non-chirped atoms)

    BlockAtomParams(double frequency, double position, double scale, double chirp = 0.0)
            : frequency(frequency), position(position), scale(scale), chirp(chirp) {}
};

/**
 * Atom parameters in units used internally by the minimizer: frequency, position, log-scale and chirp.
 */
using BlockAtomParamsArray = std::array<double, 4>;

/**
 * Auxiliary class converting atom parameters between actual values and scaled values used internally by the minimizer.
 */
class BlockAtomParamsConverter {
protected:
    double frequency_step;
    double position_step;
    double log_scale_step;

    double max_scaled_frequency;

    // step of the dimensionless chirp parameter β = chirp × scale²
    double chirp_step;

    // maximum absolute value of the chirp rate, in cycles per sample squared
    double chirp_max;

    // maximum absolute value of β = chirp × scale²
    double chirp_beta_max;

    // if positive, the instantaneous frequency is required to stay between 0 and the maximum frequency
    // for all |t| ≤ envelope_half_width × scale
    double envelope_half_width;

public:
    BlockAtomParamsConverter() : frequency_step(1.0), position_step(1.0), log_scale_step(1.0), max_scaled_frequency(0.5),
                                 chirp_step(1.0), chirp_max(INFINITY), chirp_beta_max(INFINITY), envelope_half_width(0.0) { }

    /**
     * @param frequency_step step in frequency (as a fraction of sampling frequency)
     * @param position_step step in position (in samples)
     * @param log_scale_step step in natural logarithm of scale
     * @param frequency_max maximum frequency (as a fraction of sampling frequency)
     * @param chirp_step step of the dimensionless chirp parameter β = chirp × scale²
     * @param chirp_max maximum absolute value of the chirp rate, in cycles per sample squared
     * @param envelope_half_width if positive, the instantaneous frequency of chirped atoms will be required
     * to stay between 0 and frequency_max for all |t| ≤ envelope_half_width × scale
     * @param chirp_beta_max maximum absolute value of β = chirp × scale²
     */
    BlockAtomParamsConverter(double frequency_step, double position_step, double log_scale_step, double frequency_max,
                             double chirp_step = 1.0, double chirp_max = INFINITY, double envelope_half_width = 0.0,
                             double chirp_beta_max = INFINITY);

    virtual ~BlockAtomParamsConverter() = default;

    /**
     * Convert from actual values to scaled values used internally by the minimizer.
     * @param params in actual units (@see BlockAtomParams)
     * @return array of 4 parameters in unit related to dictionary steps
     */
    [[nodiscard]] BlockAtomParamsArray arrayFromParams(const BlockAtomParams &params) const;

    /**
     * Convert from scaled values used internally by the minimizer to actual values.
     * @param array of 4 parameters in unit related to dictionary steps
     * @return params in actual units (@see BlockAtomParams) and a penalty for parameters outside of the allowed range
     */
    [[nodiscard]] std::pair<BlockAtomParams, double> paramsFromArray(const BlockAtomParamsArray &array) const;

private:
    double fix_frequency(double &scaled_frequency) const;

    double fix_chirp(double &scaled_chirp, double frequency, double scale) const;

protected:
    virtual double fix_log_scale(double &scaled_log_scale) const;

    /**
     * Fix the value given as a reference so it will be between min and max, inclusive.
     *
     * @return if given value has already been between min and max, 0;
     * otherwise, its distance from this interval
     */
    static double fix_scaled_argument(double &value, double min, double max);
};

/**
 * Auxiliary class converting atom parameters between actual values and scaled values used internally by the minimizer.
 */
class BlockAtomParamsConverterBounded : public BlockAtomParamsConverter {
    double min_scaled_log_scale;
    double max_scaled_log_scale;

public:
    BlockAtomParamsConverterBounded(double frequency_step, double position_step, double log_scale_step, double frequency_max, double min_scale, double max_scale,
                                    double chirp_step = 1.0, double chirp_max = INFINITY, double envelope_half_width = 0.0,
                                    double chirp_beta_max = INFINITY);

    double fix_log_scale(double &scaled_log_scale) const final;
};

#endif //EMPI_BLOCK_ATOM_BASE_H
