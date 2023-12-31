package linalg

// Error: holds the error message
type Error struct{ string }

// Error: method for an error to print string
func (err Error) Error() string { return err.string }

// Different types of errors
var (
	ErrNegativeDimensions    = Error{"negative dimensions provided"}
	ErrZeroLengthDimensions  = Error{"zero length array"}
	ErrMismatchDimensions    = Error{"mismatch dimensions"}
	ErrMulMismatchDimensions = Error{"AxB error: number of cols in A != rows in B"}
	ErrBoundaryDimensions    = Error{"out of boundary limit reached"}
)
