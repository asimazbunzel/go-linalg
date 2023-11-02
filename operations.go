package linalg

// MatMul: returns the result of multiplying AxB
func Mul(a, b *Dense) (*Dense, error) {
	// get matrix shapes (m,n) -- rows,cols
	aShape := a.Shape()
	bShape := b.Shape()

	// check that cols in A == rows in B
	if aShape[1] != bShape[0] {
		return nil, ErrMulMismatchDimensions
	}

	// this works as follows: A(M,L), B(L,N) => C(M,N)
	//   - C[i,j] = C[i*N + j] = A[i,k] x B[k,j] = A[i*L + k] * B[k*N + j]
	res := make([]float64, aShape[0]*bShape[1])
	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < bShape[1]; j++ {
			for k := 0; k < aShape[1]; k++ {
				res[i*bShape[1]+j] += a.data[i*aShape[1]+k] * b.data[k*bShape[1]+j]
			}
		}
	}

	return NewDense(aShape[0], bShape[1], res), nil
}
