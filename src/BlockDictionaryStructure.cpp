/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <algorithm>
#include "BlockDictionaryStructure.h"
#include "BlockHelper.h"

//////////////////////////////////////////////////////////////////////////////

BlockDictionaryStructure::BlockDictionaryStructure(std::shared_ptr<Family> family_, double energy_error, double scale_min, double scale_max, double frequency_max,
                                                   bool chirp_enabled, double chirp_max, double chirp_energy_error, double chirp_beta_max)
    : family(std::move(family_)), energy_error(energy_error), scale_min(scale_min), scale_max(scale_max), frequency_max(frequency_max),
    log_scale_step(family->inv_scale_integral(1 - energy_error)),
    dt_scale(family->inv_time_integral(1 - energy_error)),
    df_scale(family->inv_freq_integral(1 - energy_error)),
    chirp_enabled(chirp_enabled),
    chirp_max(chirp_max),
    chirp_beta_max(chirp_beta_max),
    chirp_geometry(chirp_enabled
        ? ChirpGeometry(chirp_energy_error, log_scale_step, chirp_max, frequency_max,
                        std::max(family->max_arg(), -family->min_arg()), scale_min, scale_max, chirp_beta_max)
        : ChirpGeometry()),
    block_structures(BlockHelper::compute_block_structures(family.get(), scale_min, scale_max, log_scale_step, df_scale, dt_scale, chirp_geometry))
{ }

[[nodiscard]] std::set<int> BlockDictionaryStructure::get_transform_sizes() const
{
    std::set<int> transform_sizes;
    for (const auto& bs : block_structures) {
        transform_sizes.insert(transform_sizes.end(), bs.transform_size);
    }
    return transform_sizes;
}
