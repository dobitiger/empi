/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_SPECTROGRAM_REQUEST_H
#define EMPI_SPECTROGRAM_REQUEST_H

#include <cassert>
#include <complex>
#include "BlockInterface.h"
#include "Corrector.h"
#include "Extractor.h"
#include "ProtoRequest.h"
#include "Types.h"

/**
 * Plain data object for a single request for spectrogram calculation.
 * Such request can be served by SpectrogramCalculator implementations by
 * 1) computing a windowed spectrogram of each channel in given multi-channel input data, and
 * 2) using a specified Extractor implementation to find the maximum value of each spectrum.
 */
struct SpectrogramRequest : public ProtoRequest {

    /**
     * pointer to the multi-channel data, of at least channel_count×channel_length samples
     * (in case of CUDA calculator, each channel MUST point to page-locked host memory)
     */
    const real *const *data;

    /**
     * length (in samples) of each channel (at least input_offset+envelope_length+(how_many-1)*input_shift)
     */
    index_t channel_length;

    /**
     * offset of the first sample of the input for the first transform
     */
    index_t input_offset;

    /**
     * envelope (window function) for calculating transforms (at least envelope_length items)
     * (in case of CUDA calculator, each channel MUST point to page-locked host memory)
     */
    const real *envelope;

    /**
     * if not null, this is a request for a block of chirped atoms:
     * "envelope" is the actual envelope multiplied by cosine of the chirp phase π·chirp·t²,
     * and this is the same envelope multiplied by sine of the chirp phase (at least envelope_length items).
     * Spectra S_cos and S_sin of the signal multiplied by both envelopes should be combined
     * as S_cos − i·S_sin (atoms with positive chirp rate) and S_cos + i·S_sin (atoms with negative chirp rate)
     */
    const real *envelope_sin = nullptr;

    /**
     * objects capable of transforming FFT output into normalized energy values (at least output_bins items)
     * (in case of CUDA calculator, each channel MUST point to page-locked host memory)
     */
    const Corrector *correctors;

    /**
     * same as correctors, but for the atoms with negative chirp rate (used only if envelope_sin is given)
     */
    const Corrector *correctors_minus = nullptr;

    /**
     * extractor implementation to reduce each spectrum to a single value
     */
    Extractor extractor;

    /**
     * buffer for extracted maximum values, one for each transform (space for at least how_many items)
     * (in case of CUDA calculator, this MUST point to page-locked host memory)
     */
    ExtractedMaximum *maxima;

    /**
     * same as maxima, but for the atoms with negative chirp rate (used only if envelope_sin is given)
     */
    ExtractedMaximum *maxima_minus = nullptr;

    /**
     * listener to be notified after the request is complete
     */
    BlockInterface *interface;

    void assertCorrectness() const {
        assert(data);
        assert(channel_length > 0);
        assert(channel_count > 0);
        assert(input_shift > 0);
        assert(how_many > 0);

        assert(envelope);
        assert(envelope_length > 0);
        assert(window_length > 0);
        assert(envelope_length <= window_length);

        assert(output_bins > 0);
        assert(output_bins <= window_length / 2 + 1);
        assert(correctors);
        assert(extractor);
        assert(maxima);
        assert(!envelope_sin || (correctors_minus && maxima_minus));
    }
};

#endif //EMPI_SPECTROGRAM_REQUEST_H
