package random

import "math/rand"

func RandomFloat64(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}

func RandomInt(min, max int) int {
	return rand.Intn(max-min+1) + min
}
