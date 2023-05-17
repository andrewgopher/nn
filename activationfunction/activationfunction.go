package activationfunction

import "nn/mathext"

type ActivationFunction struct {
	Eval       func(float64) float64
	Derivative func(float64) float64
}

var Identity *ActivationFunction = &ActivationFunction{
	Eval: func(x float64) float64 {
		return x
	},
	Derivative: func(x float64) float64 {
		return 1
	},
}

var Sigmoid *ActivationFunction = &ActivationFunction{
	Eval: func(x float64) float64 {
		return mathext.Sigmoid(x)
	},
	Derivative: func(x float64) float64 {
		return mathext.Sigmoid(x) * (1 - mathext.Sigmoid(x))
	},
}

var IntToActivationFunction = map[int]*ActivationFunction{0: Identity, 1: Sigmoid}
var ActivationFunctionToInt = map[*ActivationFunction]int{Identity: 0, Sigmoid: 1}
