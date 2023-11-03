package linalg

import (
	"fmt"
	"math/rand"
)

// Shape: struct containing dimensional information of matrix
type Shape []int

// Shape: returns number of rows and columns as an array of integers in the dense matrix
func (m *Dense) Shape() Shape {
	return Shape{m.nrows, m.ncols}
}

// RandValue: returns a random float between [min, max]
func RandValue(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func (m *Dense) Print() {
	// borders of message
	topLeft, topRight := "⎡", "⎤\n"
	botLeft, botRight := "⎣", "⎦\n"

	for i := 0; i < m.nrows; i++ {
		var leftMsg, rightMsg string
		if i == 0 {
			leftMsg = topLeft
			rightMsg = topRight
		} else if i == m.nrows-1 {
			leftMsg = botLeft
			rightMsg = botRight
		} else {
			leftMsg = "⎢"
			rightMsg = "⎢\n"
		}

		for j := 0; j < m.ncols; j++ {
			idx := i*m.ncols + j
			if j == m.ncols-1 {
				fmt.Printf("%.4f %s", m.data[idx], rightMsg)
			} else if j == 0 {
				fmt.Printf("%s %.4f ", leftMsg, m.data[idx])
			} else {
				fmt.Printf("%.4f ", m.data[idx])
			}
		}
	}
}
