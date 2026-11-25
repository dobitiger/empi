/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_CHIRP_GEOMETRY_H
#define EMPI_CHIRP_GEOMETRY_H

#include <cmath>

/**
 * Geometry of the dictionary of chirped atoms (chirplets).
 *
 * The shape of a Gaussian chirplet is described by scale s and chirp rate c, or by a single complex number z = c + i/s².
 * The squared product of two chirplets differing only in shape is equal to 1/cosh(d/2), where d is the hyperbolic
 * distance between their shapes on the upper half-plane. All ordinary (non-chirped) Gabor atoms lie on a single
 * geodesic c = 0, parametrized by λ = log s (the distance between two Gabor atoms being 2Δλ), and a chirplet with
 * dimensionless chirp β = c·s² lies at distance D = asinh(β) from that geodesic; the nearest Gabor atom ("foot point")
 * has a shorter scale: s/√cosh(D). In coordinates (λ, D), where exp(λ) is the scale of the foot point,
 *
 *   cosh d = cosh D₁ · cosh D₂ · cosh(2Δλ) − sinh D₁ · sinh D₂
 *
 * The dictionary consists of layers of constant β = ±sinh(k·Δd), for k = 0 (ordinary Gabor atoms), 1, 2, …
 * As the length of such a curve is cosh(D) times larger than the length of the corresponding part of the geodesic,
 * step in log-scale in k-th layer is cosh(k·Δd) times smaller than the one in the ordinary Gabor dictionary.
 *
 * Chirp rates are restricted so that
 *  1) |c| does not exceed the given maximum,
 *  2) |β| does not exceed the given maximum, and
 *  3) the instantaneous frequency stays between zero and the maximum frequency within the entire atom.
 * Atoms of each layer exist only for those scales for which at least some frequencies fulfill both conditions.
 */
class ChirpGeometry {
    bool enabled;
    int layers;
    double layer_step;
    double log_scale_step;
    double chirp_max;
    double beta_max;
    double frequency_max;
    double envelope_half_width;
    double scale_min;
    double scale_max;

public:
    /**
     * Create a geometry for the dictionary without any chirped atoms.
     */
    ChirpGeometry();

    /**
     * @param chirp_energy_error analogue of ε² for distance between layers: product between corresponding atoms
     * from neighbouring layers will be equal to 1−ε²; value of 1 (or more) results in ordinary Gabor atoms only
     * @param log_scale_step step in log-scale for ordinary (non-chirped) atoms
     * @param chirp_max maximum absolute chirp rate in cycles per sample squared (can be infinite)
     * @param frequency_max maximum frequency (as a fraction of sampling frequency)
     * @param envelope_half_width instantaneous frequency of chirped atoms will be required
     * to stay between 0 and frequency_max for all |t| ≤ envelope_half_width × scale
     * @param scale_min minimum scale of atoms in samples
     * @param scale_max maximum scale of atoms in samples
     * @param beta_max maximum absolute value of dimensionless β = chirp × scale² (can be infinite)
     */
    ChirpGeometry(double chirp_energy_error, double log_scale_step, double chirp_max, double frequency_max,
                  double envelope_half_width, double scale_min, double scale_max, double beta_max = INFINITY);

    /**
     * @return true if chirp rate is being fitted at all, false otherwise
     */
    [[nodiscard]] bool is_enabled() const {
        return enabled;
    }

    /**
     * @return number of layers of chirped atoms (not counting the layer of ordinary non-chirped atoms),
     * each consisting of atoms with both positive and negative chirp rates
     */
    [[nodiscard]] int layer_count() const;

    /**
     * @return hyperbolic distance between given layer and the layer of non-chirped atoms
     */
    [[nodiscard]] double layer_distance(int layer) const;

    /**
     * @return value of β = chirp × scale² for all atoms in given layer
     */
    [[nodiscard]] double layer_beta(int layer) const;

    /**
     * @return step in log-scale between consecutive scales of given layer
     */
    [[nodiscard]] double layer_log_scale_step(int layer) const;

    /**
     * @return the smallest scale (in samples) of atoms in given layer; may exceed scale_max if layer does not exist
     */
    [[nodiscard]] double layer_scale_start(int layer) const;

    /**
     * @return maximum allowed value of |β| = |chirp| × scale² for atoms of given scale,
     * with central frequency differing by frequency_margin from the nearest limit of the allowed frequency range
     */
    [[nodiscard]] double beta_allowed(double scale, double frequency_margin) const;

    /**
     * Each atom of the dictionary is responsible for all chirplets nearer to its layer than to any other layer.
     * This method calculates how much the upper bound for energy of the optimized atom
     * has to be increased (multiplied) compared to the ordinary Gabor dictionary.
     *
     * @param layer index of the layer, 0 for ordinary non-chirped atoms
     * @param scale scale of the atom in samples
     * @param frequency frequency of the atom (as a fraction of sampling frequency)
     * @param frequency_step step in frequency between neighbouring atoms
     * @return multiplier for the energy upper bound: 1 if chirp rate is not fitted,
     * 0 if no allowed chirplets are covered by this atom
     */
    [[nodiscard]] double booster_multiplier(int layer, double scale, double frequency, double frequency_step) const;

private:
    /**
     * @return maximum allowed distance from the geodesic of non-chirped atoms for chirplets
     * with given foot point on this geodesic (given as a logarithm of scale of the nearest non-chirped atom)
     */
    [[nodiscard]] double distance_allowed(double log_scale_foot, double frequency_margin) const;
};

#endif //EMPI_CHIRP_GEOMETRY_H
