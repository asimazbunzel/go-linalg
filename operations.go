package linalg

// GetRow: returns a row from a dense matrix
func (m *Dense) GetRow(row int) (*Dense, error) {
	// get matrix shapes (m,n) -- rows,cols
	mShape := m.Shape()

	// check that row to retrieve is within rows of dense matrix
	if row >= mShape[0] {
		return nil, ErrMismatchDimensions
	}

	res := make([]float64, 1*mShape[1])
	for j := 0; j < mShape[1]; j++ {
		res[j] = m.Data[row*mShape[1]+j]
	}

	return NewDense(1, mShape[1], res), nil
}

// Slice: returns a slice of a dense matrix
func (m *Dense) Slice(rows, cols []int) (*Dense, error) {
	// dimensions of root dense matrix
	mShape := m.Shape()

	// check dimensions
	if rows[0] < 0 || rows[1] < 0 || cols[0] < 0 || cols[1] < 0 {
		return nil, ErrNegativeDimensions
	}
	if rows[1] > mShape[0] || cols[1] > mShape[1] {
		return nil, ErrBoundaryDimensions
	}

	nrows := rows[1] - rows[0]
	ncols := cols[1] - cols[0]
	res := []float64{}
	for i := rows[0]; i < rows[1]; i++ {
		for j := cols[0]; j < cols[1]; j++ {
			res = append(res, m.Data[i*mShape[1]+j])
		}
	}

	return NewDense(nrows, ncols, res), nil
}

// Sum: returns the result of substracting two dense matrices
func Sum(a, b *Dense) (*Dense, error) {
	// get matrix shapes (m,n) -- rows,cols
	aShape := a.Shape()
	bShape := b.Shape()

	// check that cols & rows in A == cols & rows in B
	if aShape[0] != bShape[0] || aShape[1] != bShape[1] {
		return nil, ErrMulMismatchDimensions
	}

	res := make([]float64, aShape[0]*bShape[1])
	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < aShape[1]; j++ {
			res[i*aShape[1]+j] = a.Data[i*aShape[1]+j] + b.Data[i*aShape[1]+j]
		}
	}

	return NewDense(aShape[0], aShape[1], res), nil
}

// Sub: returns the result of substracting two dense matrices
func Sub(a, b *Dense) (*Dense, error) {
	// get matrix shapes (m,n) -- rows,cols
	aShape := a.Shape()
	bShape := b.Shape()

	// check that cols & rows in A == cols & rows in B
	if aShape[0] != bShape[0] || aShape[1] != bShape[1] {
		return nil, ErrMulMismatchDimensions
	}

	res := make([]float64, aShape[0]*bShape[1])
	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < aShape[1]; j++ {
			res[i*aShape[1]+j] = a.Data[i*aShape[1]+j] - b.Data[i*aShape[1]+j]
		}
	}

	return NewDense(aShape[0], aShape[1], res), nil
}

// Scale: multiply a dense matrix with a scalar
func Scale(f float64, a *Dense) (*Dense, error) {
	// get matrix shapes (m,n) -- rows,cols
	aShape := a.Shape()

	res := make([]float64, aShape[0]*aShape[1])
	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < aShape[1]; j++ {
			res[i*aShape[1]+j] += f * a.Data[i*aShape[1]+j]
		}
	}

	return NewDense(aShape[0], aShape[1], res), nil
}

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
				res[i*bShape[1]+j] += a.Data[i*aShape[1]+k] * b.Data[k*bShape[1]+j]
			}
		}
	}

	return NewDense(aShape[0], bShape[1], res), nil
}

// MulElem
func MulElem(a, b *Dense) (*Dense, error) {
	// get matrix shapes (m,n) -- rows,cols
	aShape := a.Shape()
	bShape := b.Shape()

	// check that cols & rows in A == cols & rows in B
	if aShape[0] != bShape[0] || aShape[1] != bShape[1] {
		return nil, ErrMulMismatchDimensions
	}

	res := make([]float64, aShape[0]*aShape[1])
	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < aShape[1]; j++ {
			res[i*aShape[1]+j] = a.Data[i*aShape[1]+j] * b.Data[i*bShape[1]+j]
		}
	}

	return NewDense(aShape[0], aShape[1], res), nil
}

// DivlElem
func DivElem(a, b *Dense) (*Dense, error) {
	// get matrix shapes (m,n) -- rows,cols
	aShape := a.Shape()
	bShape := b.Shape()

	// check that cols & rows in A == cols & rows in B
	if aShape[0] != bShape[0] || aShape[1] != bShape[1] {
		return nil, ErrMulMismatchDimensions
	}

	res := make([]float64, aShape[0]*aShape[1])
	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < aShape[1]; j++ {
			res[i*aShape[1]+j] = a.Data[i*aShape[1]+j] / b.Data[i*bShape[1]+j]
		}
	}

	return NewDense(aShape[0], aShape[1], res), nil
}

// Apply
func Apply(m *Dense, fn func(v float64) float64) *Dense {
	// get matrix shape
	mShape := m.Shape()

	// apply function to each element of the matrix
	res := make([]float64, mShape[0]*mShape[1])
	for i := 0; i < mShape[0]; i++ {
		for j := 0; j < mShape[1]; j++ {
			res[i*mShape[1]+j] = fn(m.Data[i*mShape[1]+j])
		}
	}

	return NewDense(mShape[0], mShape[1], res)
}

// T
func T(m *Dense) *Dense {
	// get matrix shape
	mShape := m.Shape()

	res := make([]float64, mShape[0]*mShape[1])
	for i := 0; i < mShape[0]; i++ {
		for j := 0; j < mShape[1]; j++ {
			res[j*mShape[0]+i] = m.Data[i*mShape[1]+j]
		}
	}

	return NewDense(mShape[1], mShape[0], res)
}
