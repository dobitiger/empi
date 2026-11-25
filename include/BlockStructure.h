/**********************************************************
 * Piotr T. Różański (c) 2015-2023                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_BLOCK_STRUCTURE_H
#define EMPI_BLOCK_STRUCTURE_H

struct BlockStructure {
	double scale;
	int envelope_length;
	int transform_size;
	double input_shift;

	// absolute value of chirp rate (in cycles per sample squared), the block consists of atoms with both signs of chirp rate
	double chirp = 0.0;

	// index of the layer of chirped atoms (see ChirpGeometry), 0 for ordinary non-chirped atoms
	int chirp_layer = 0;
};

#endif //EMPI_BLOCK_STRUCTURE_H
