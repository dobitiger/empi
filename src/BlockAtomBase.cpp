/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <algorithm>
#include "BlockAtomBase.h"

//////////////////////////////////////////////////////////////////////////////

BlockAtomParamsConverter::BlockAtomParamsConverter(double frequency_step, double position_step, double log_scale_step, double frequency_max,
                                                   double chirp_step, double chirp_max, double envelope_half_width, double chirp_beta_max)
        : frequency_step(frequency_step), position_step(position_step), log_scale_step(log_scale_step),
          max_scaled_frequency(frequency_max / frequency_step),
          chirp_step(chirp_step), chirp_max(chirp_max), chirp_beta_max(chirp_beta_max), envelope_half_width(envelope_half_width) {}

BlockAtomParamsArray BlockAtomParamsConverter::arrayFromParams(const BlockAtomParams &params) const {
    return {
            params.frequency / frequency_step,
            params.position / position_step,
            std::log(params.scale) / log_scale_step,
            params.chirp * params.scale * params.scale / chirp_step
    };
}

std::pair<BlockAtomParams, double> BlockAtomParamsConverter::paramsFromArray(const BlockAtomParamsArray &array) const {
    double scaled_frequency = array[0];
    double scaled_position = array[1];
    double scaled_log_scale = array[2];
    double scaled_chirp = array[3];

    double penalty = fix_frequency(scaled_frequency) + fix_log_scale(scaled_log_scale);

    const double frequency = scaled_frequency * frequency_step;
    const double scale = std::exp(scaled_log_scale * log_scale_step);
    penalty += fix_chirp(scaled_chirp, frequency, scale);

    return std::make_pair(
            BlockAtomParams(
                    frequency,
                    scaled_position * position_step,
                    scale,
                    scaled_chirp * chirp_step / (scale * scale)
            ),
            penalty
    );
}

double BlockAtomParamsConverter::fix_chirp(double &scaled_chirp, double frequency, double scale) const {
    // the minimizer works with dimensionless β = chirp × scale², in which the dictionary metric is uniform
    double beta_max = std::min(chirp_beta_max, chirp_max * scale * scale);
    if (envelope_half_width > 0) {
        // instantaneous frequency must stay between 0 and maximum frequency within the atom's support
        const double frequency_max = max_scaled_frequency * frequency_step;
        const double frequency_margin = std::max(0.0, std::min(frequency, frequency_max - frequency));
        beta_max = std::min(beta_max, frequency_margin * scale / envelope_half_width);
    }
    if (std::isinf(beta_max)) {
        return 0.0;
    }
    const double max_scaled_chirp = beta_max / chirp_step;
    return fix_scaled_argument(scaled_chirp, -max_scaled_chirp, max_scaled_chirp);
}

double BlockAtomParamsConverter::fix_frequency(double &scaled_frequency) const {
    return fix_scaled_argument(scaled_frequency, 0.0, max_scaled_frequency);
}

double BlockAtomParamsConverter::fix_log_scale(double &scaled_log_scale) const {
    return 0.0;
}

double BlockAtomParamsConverter::fix_scaled_argument(double &value, double min, double max) {
    double penalty = 0.0;
    if (value < min) {
        penalty += min - value;
        value = min;
    } else if (value > max) {
        penalty += value - max;
        value = max;
    }
    return penalty;
}

//////////////////////////////////////////////////////////////////////////////

BlockAtomParamsConverterBounded::BlockAtomParamsConverterBounded(double frequency_step, double position_step, double log_scale_step,
                                                                 double frequency_max, double min_scale, double max_scale,
                                                                 double chirp_step, double chirp_max, double envelope_half_width, double chirp_beta_max)
        : BlockAtomParamsConverter(frequency_step, position_step, log_scale_step, frequency_max, chirp_step, chirp_max, envelope_half_width, chirp_beta_max),
          min_scaled_log_scale(std::log(min_scale) / log_scale_step),
          max_scaled_log_scale(std::log(max_scale) / log_scale_step) {}

double BlockAtomParamsConverterBounded::fix_log_scale(double &scaled_log_scale) const {
    return fix_scaled_argument(scaled_log_scale, min_scaled_log_scale, max_scaled_log_scale);
}
