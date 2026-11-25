/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_BLOCK_HELPER_H
#define EMPI_BLOCK_HELPER_H

#include <memory>
#include <utility>
#include "Array.h"
#include "Block.h"
#include "BlockAtomBase.h"
#include "BlockStructure.h"
#include "ChirpGeometry.h"
#include "Corrector.h"
#include "Family.h"
#include "PinnedArray.h"
#include "SpectrumCalculator.h"

class BlockHelper {
    static void add_block_structures_to_list(std::list<BlockStructure>& result, const Family* family, double scale, double df_scale, double dt_scale,
                                             double chirp = 0.0, int chirp_layer = 0);

public:
    static Block create_block(PinnedArray2D<double> data, std::shared_ptr<Family> family, double scale,
                             std::shared_ptr<BlockAtomParamsConverter> converter, double booster,
                             int window_length, int output_bins, int input_shift, double subsample_offset, Extractor extractor, SpectrumCalculator& calculator, bool allow_overstep = true,
                             double chirp = 0.0, Array1D<double> booster_multipliers = Array1D<double>());

    /**
     * Generate quadrature envelopes for a block of chirped atoms.
     *
     * @param envelope ordinary (non-chirped) envelope
     * @param envelope_center_offset position of the center of the envelope, relative to its first sample
     * @param chirp chirp rate in cycles per sample squared
     * @return envelope multiplied by, respectively, cosine and sine of the chirp phase
     */
    static std::pair<PinnedArray1D<double>, PinnedArray1D<double>> generate_chirped_envelopes(const Array1D<double>& envelope, double envelope_center_offset, double chirp);

    /**
     * Generate correctors for a block of chirped atoms.
     *
     * @return correctors for atoms with, respectively, positive and negative chirp rate
     */
    static std::pair<PinnedArray1D<Corrector>, PinnedArray1D<Corrector>> generate_chirped_correctors(const Array1D<double>& envelope, double envelope_center_offset, double chirp,
                                                                                                    int window_length, int output_bins, SpectrumCalculator& calculator);

    static std::pair<PinnedArray1D<double>, double> generate_envelope(const Family* family, double scale, double subsample_offset);

    static PinnedArray1D<Corrector> generate_correctors(const Array1D<double>& envelope, int window_length, int output_bins, SpectrumCalculator& calculator);

    static std::list<BlockStructure> compute_block_structures(const Family* family, double scale_min, double scale_max, double log_scale_step, double df_scale, double dt_scale,
                                                              const ChirpGeometry& chirp_geometry = ChirpGeometry());

    static int round_transform_size(double min_transform_size_as_float);
};

#endif //EMPI_BLOCK_HELPER_H
