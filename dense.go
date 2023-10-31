package linalg

import (
	"math/rand"
)

// Dense: structure representing a dense matrix
type Dense struct {
	nrows int
	ncols int
	data  []float64
}

// Shape: struct containing dimensional information of matrix
type Shape []int

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
		nrows: r,
		ncols: c,
		data:  data,
	}
}

// Shape: returns number of rows and columns as an array of integers in the dense matrix
func (m *Dense) Shape() Shape {
	return Shape{m.nrows, m.ncols}
}

func randValue(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

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
			data[i*r+j] = randValue(min, max)
		}
	}

	return &Dense{
		nrows: r,
		ncols: c,
		data:  data,
	}
}
