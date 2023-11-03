package linalg

// Dense: structure representing a dense matrix
type Dense struct {
	Nrows int
	Ncols int
	Data  []float64
}

// NewDense: returns a pointer to a dense matrix
func NewDense(r, c int, data []float64) *Dense {
	if r <= 0 || c <= 0 {
		if r == 0 || c == 0 {
			panic(ErrZeroLengthDimensions)
		}
		panic(ErrNegativeDimensions)
	}
	if data != nil && r*c != len(data) {
		panic(ErrMismatchDimensions)
	}
	if data == nil {
		data = make([]float64, r*c)
	}
	return &Dense{
		Nrows: r,
		Ncols: c,
		Data:  data,
	}
}

// NewRandDense: returns a dense matrix with random values
func NewRandDense(r, c int, min, max float64) *Dense {
	if r <= 0 || c <= 0 {
		if r == 0 || c == 0 {
			panic(ErrZeroLengthDimensions)
		}
		panic(ErrNegativeDimensions)
	}

	// alloc array of proper dimension & fill it with random floats
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = RandValue(min, max)
		}
	}

	return &Dense{
		Nrows: r,
		Ncols: c,
		Data:  data,
	}
}

// NewZeroDense: returns a dense matrix with zero values
func NewZeroDense(r, c int) *Dense {
	if r <= 0 || c <= 0 {
		if r == 0 || c == 0 {
			panic(ErrZeroLengthDimensions)
		}
		panic(ErrNegativeDimensions)
	}

	// alloc array of proper dimension & fill it with zeroes
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = 0.0
		}
	}

	return &Dense{
		Nrows: r,
		Ncols: c,
		Data:  data,
	}
}

// NewEye: returns an identity dense matrix
func NewEye(r int) *Dense {
	if r <= 0 {
		if r == 0 {
			panic(ErrZeroLengthDimensions)
		}
		panic(ErrNegativeDimensions)
	}

	data := make([]float64, r*r)
	for i := 0; i < r; i++ {
		data[i*r+i] = 1.0
	}

	return &Dense{
		Nrows: r,
		Ncols: r,
		Data:  data,
	}
}
