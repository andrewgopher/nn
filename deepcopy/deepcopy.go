package deepcopy

import (
	"gonum.org/v1/gonum/mat"
)

func Vector(vec mat.MutableVector) mat.MutableVector {
	result := mat.NewVecDense(vec.Len(), make([]float64, vec.Len()))
	for i := 0; i < vec.Len(); i++ {
		result.SetVec(i, vec.AtVec(i))
	}
	return result
}

func VectorSlice1D(slice []mat.MutableVector) []mat.MutableVector {
	result := make([]mat.MutableVector, len(slice))
	for i := 0; i < len(slice); i++ {
		result[i] = Vector(slice[i])
	}
	return result
}

func VectorSlice2D(slice [][]mat.MutableVector) [][]mat.MutableVector {
	result := make([][]mat.MutableVector, len(slice))
	for i := 0; i < len(slice); i++ {
		result[i] = VectorSlice1D(slice[i])
	}
	return result
}

func PrimitiveSlice1D[T any](slice []T) []T { //"shallow" copy that's good enough for primitive types, including functions
	result := make([]T, len(slice))
	copy(result, slice)
	return result
}

func PrimitiveSlice2D[T any](slice [][]T) [][]T {
	result := make([][]T, len(slice))
	for i := 0; i < len(slice); i++ {
		result[i] = PrimitiveSlice1D(slice[i])
	}
	return result
}
