package linalg

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
)

// Shape: struct containing dimensional information of matrix
type Shape []int

// Shape: returns number of rows and columns as an array of integers in the dense matrix
func (m *Dense) Shape() Shape {
	return Shape{m.Nrows, m.Ncols}
}

// RandValue: returns a random float between [min, max]
func RandValue(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

// Print: prints dense matrix to stdout
func (m *Dense) Print() {
	// borders of message
	topLeft, topRight := "⎡", "⎤\n"
	botLeft, botRight := "⎣", "⎦\n"

	for i := 0; i < m.Nrows; i++ {
		var leftMsg, rightMsg string
		if i == 0 {
			leftMsg = topLeft
			rightMsg = topRight
		} else if i == m.Nrows-1 {
			leftMsg = botLeft
			rightMsg = botRight
		} else {
			leftMsg = "⎢"
			rightMsg = "⎢\n"
		}

		for j := 0; j < m.Ncols; j++ {
			idx := i*m.Ncols + j
			if j == m.Ncols-1 {
				fmt.Printf("%.4f %s", m.Data[idx], rightMsg)
			} else if j == 0 {
				fmt.Printf("%s %.4f ", leftMsg, m.Data[idx])
			} else {
				fmt.Printf("%.4f ", m.Data[idx])
			}
		}
	}
}

// header: struct with information of dense matrix to store as binary
type header struct {
	Rows int64
	Cols int64
}

// marshalBinary
func (h *header) marshalBinary(w io.Writer) (int, error) {
	sizeHeader := binary.Size(header{})
	buf := bytes.NewBuffer(make([]byte, 0, sizeHeader))
	err := binary.Write(buf, binary.LittleEndian, h)
	if err != nil {
		return 0, err
	}
	return w.Write(buf.Bytes())
}

// unmarshalBinary
func (h *header) unmarshalBinary(buf []byte) error {
	err := binary.Read(bytes.NewReader(buf), binary.LittleEndian, h)
	if err != nil {
		return err
	}
	return nil
}

// MarshalBinary
func (m *Dense) MarshalBinary() ([]byte, error) {
	// header sizeof
	sizeHeader := binary.Size(header{})
	// sizeof float64 class
	sizeFloat64 := binary.Size(float64(0))

	// length & byte array
	bufLen := int64(sizeHeader) + int64(m.Nrows)*int64(m.Ncols)*int64(sizeFloat64)
	buf := make([]byte, bufLen)

	// create header of dense matrix
	h := header{
		Rows: int64(m.Nrows),
		Cols: int64(m.Ncols),
	}
	n, err := h.marshalBinary(bytes.NewBuffer(buf[:0]))
	if err != nil {
		return buf[:n], err
	}

	dims := m.Shape()
	p := sizeHeader
	for i := 0; i < dims[0]; i++ {
		for j := 0; j < dims[1]; j++ {
			binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(m.Data[i*dims[1]+j]))
			p += sizeFloat64
		}
	}

	return buf, nil
}

// UnmarshalBinary
func (m *Dense) UnmarshalBinary(data []byte) error {
	// header sizeof
	sizeHeader := binary.Size(header{})
	// sizeof float64 class
	sizeFloat64 := binary.Size(float64(0))

	// grab header data
	var h header
	err := h.unmarshalBinary(data[:sizeHeader])
	if err != nil {
		return err
	}

	// dims of matrix & data array
	r := int(h.Rows)
	c := int(h.Cols)
	m.Nrows = r
	m.Ncols = c
	m.Data = make([]float64, m.Nrows*m.Ncols)

	// fill matrix
	p := sizeHeader
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Data[i*c+j] = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
			p += sizeFloat64
		}
	}

	return nil
}
