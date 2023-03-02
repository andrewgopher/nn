package mathext

import "math"

func RoundFloat64(val float64, precision int) float64 {
	ratio := math.Pow(10, float64(precision))
	return math.Round(val*ratio) / ratio
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}
