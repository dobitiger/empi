/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_BLOCK_DICTIONARY_STRUCTURE_H
#define EMPI_BLOCK_DICTIONARY_STRUCTURE_H

#include <cmath>
#include <memory>
#include <list>
#include <set>
#include "BlockStructure.h"
#include "ChirpGeometry.h"
#include "Family.h"

struct BlockDictionaryStructure {

    const std::shared_ptr<Family> family;
    const double energy_error, scale_min, scale_max, frequency_max, log_scale_step, dt_scale, df_scale;
    const bool chirp_enabled;
    const double chirp_max;
    const double chirp_beta_max;
    const ChirpGeometry chirp_geometry;
    const std::list<BlockStructure> block_structures;

    /**
     * @param family Family object describing properties of the envelope function
     * @param energy_error ε² parameter of the optimal dictionary
     * @param scale_min minimum scale of atoms in samples
     * @param scale_max maximum scale of atoms in samples
     * @param frequency_max maximum frequency (as a fraction of sampling frequency)
     * @param chirp_enabled whether chirp rate will be fitted as an additional parameter of atoms
     * @param chirp_max maximum absolute chirp rate in cycles per sample squared (can be infinite)
     * @param chirp_energy_error analogue of ε² for chirp rate (see ChirpGeometry),
     * value of 1 results in a dictionary of non-chirped atoms only
     * @param chirp_beta_max maximum absolute value of dimensionless β = chirp × scale² (can be infinite)
     */
    BlockDictionaryStructure(std::shared_ptr<Family> family, double energy_error, double scale_min, double scale_max, double frequency_max,
                             bool chirp_enabled = false, double chirp_max = INFINITY, double chirp_energy_error = 1.0,
                             double chirp_beta_max = INFINITY);

    [[nodiscard]] std::set<int> get_transform_sizes() const;
};

#endif //EMPI_BLOCK_DICTIONARY_STRUCTURE_H
