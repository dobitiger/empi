/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <algorithm>
#include <cmath>
#include "BlockDictionary.h"
#include "BlockHelper.h"

static const int MAX_BLOCKS_PER_SCALE = 1000; // sanity check, quite arbitrary

//////////////////////////////////////////////////////////////////////////////

BlockDictionary::BlockDictionary(const BlockDictionaryStructure& structure, const PinnedArray2D<double>& data,
                                 Extractor extractor, SpectrumCalculator &calculator, bool allow_overstep) {
    const double booster = 1.0 / structure.family->optimality_factor_e2(structure.energy_error);

    // For Gaussian envelopes, the product of two atoms differing only in chirp rate is equal to (1 + Δβ²/4)^(−¼),
    // where β = chirp × scale². The step in β is therefore selected in the same way as for other parameters:
    // so the product between neighbouring atoms would be equal to 1−ε². It is only used to scale the chirp axis
    // for the local optimization, so it serves as a sensible unit for other envelopes as well.
    const double chirp_step = 2.0 * std::sqrt(std::pow(1 - structure.energy_error, -4) - 1);
    const double envelope_half_width = std::max(structure.family->max_arg(), -structure.family->min_arg());

    for (const auto& bs : structure.block_structures) {
        int output_bins = Types::floor<int>(bs.transform_size * structure.frequency_max) + 1;
        auto converter = std::make_shared<BlockAtomParamsConverterBounded>(
                1.0 / bs.transform_size,
                static_cast<double>(bs.input_shift),
                structure.log_scale_step,
                structure.frequency_max,
                structure.scale_min,
                structure.scale_max,
                chirp_step,
                structure.chirp_max,
                envelope_half_width,
                structure.chirp_beta_max
        );

        // additional multipliers for the upper bound of energy, accounting for all chirplets
        // which have this block's atoms as their nearest neighbours in the dictionary
        Array1D<double> booster_multipliers;
        if (structure.chirp_enabled) {
            booster_multipliers = Array1D<double>(output_bins);
            for (int k = 0; k < output_bins; ++k) {
                booster_multipliers[k] = structure.chirp_geometry.booster_multiplier(
                        bs.chirp_layer, bs.scale, static_cast<double>(k) / bs.transform_size, 1.0 / bs.transform_size
                );
            }
        }

        if (bs.input_shift < 1) {
            double number_of_blocks_as_float = 1.0 / bs.input_shift;
            if (number_of_blocks_as_float > MAX_BLOCKS_PER_SCALE) {
                throw std::runtime_error("requested minimum atom scale is too small");
            }
            int number_of_blocks = Types::round<int>(number_of_blocks_as_float);
            for (int i=0; i<number_of_blocks; ++i) {
                double subsample_offset = static_cast<double>(i) / static_cast<double>(number_of_blocks);
                blocks.push_back(
                        BlockHelper::create_block(data, structure.family, bs.scale, converter, booster, bs.transform_size,
                                output_bins, 1, subsample_offset, extractor, calculator, allow_overstep, bs.chirp, booster_multipliers)
                );
            }
        } else {
            int input_shift = Types::round<int>(bs.input_shift);
            blocks.push_back(
                BlockHelper::create_block(data, structure.family, bs.scale, converter, booster, bs.transform_size,
                    output_bins, input_shift, 0.0, extractor, calculator, allow_overstep, bs.chirp, booster_multipliers)
            );
        }
    }
}

BlockDictionary::BlockDictionary(Block&& block) {
    blocks.push_back(block);
}

size_t BlockDictionary::get_atom_count() {
    size_t result = 0;
    for (auto &block : blocks) {
        result += block.get_atom_count();
    }
    return result;
}

BasicAtomPointer BlockDictionary::get_best_match() {
    std::shared_ptr<BlockAtom> result;
    for (auto &block : blocks) {
        BlockAtom atom = block.get_best_match();
        if (!result || *result < atom) {
            result = std::make_shared<BlockAtom>(std::move(atom));
        }
    }
    if (!result) {
        return nullptr;
    }
    return result;
}

std::list<BasicAtomPointer> BlockDictionary::get_candidate_matches(double energy_to_exceed) {
    // TODO
    std::list<BlockAtom> atoms;
    for (auto &block : blocks) {
        atoms.splice(atoms.end(), block.get_candidate_matches(energy_to_exceed));
    }
    std::list<BasicAtomPointer> result;
    for (auto &atom : atoms) {
        result.push_back(std::make_shared<BlockAtom>(std::move(atom)));
    }
    return result;
}

void BlockDictionary::fetch_proto_requests(std::list<ProtoRequest> &requests) {
    for (auto &block: blocks) {
        requests.push_back(block.buildRequest());
    }
}

void BlockDictionary::fetch_requests(IndexRange signal_range, std::list<SpectrogramRequest> &requests) {
    for (auto &block: blocks) {
        SpectrogramRequest request = block.buildRequest(signal_range.first_index, signal_range.end_index);
        if (request.how_many > 0) {
            requests.push_back(request);
        }
    }
}
