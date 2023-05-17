package feedforward

import (
	"nn/activationfunction"
	"nn/deepcopy"
	"nn/random"

	"gonum.org/v1/gonum/mat"
)

// type ActivationFunction struct {
// 	Eval       func(float64) float64
// 	Derivative func(float64) float64
// }

// var Identity *ActivationFunction = &ActivationFunction{
// 	Eval: func(x float64) float64 {
// 		return x
// 	},
// 	Derivative: func(x float64) float64 {
// 		return 1
// 	},
// }

// var Sigmoid *ActivationFunction = &ActivationFunction{
// 	Eval: func(x float64) float64 {
// 		return mathext.Sigmoid(x)
// 	},
// 	Derivative: func(x float64) float64 {
// 		return mathext.Sigmoid(x) * (1 - mathext.Sigmoid(x))
// 	},
// }

type Network struct {
	NumLayers           int
	LayerSizes          []int
	Weights             [][]*mat.VecDense
	Biases              [][]float64
	ActivationFunctions []*activationfunction.ActivationFunction //per layer
}

type JSONNetwork struct {
	NumLayers           int
	LayerSizes          []int
	Weights             [][][]float64
	Biases              [][]float64
	ActivationFunctions []int //per layer
}

func (network *Network) ToJSONNetwork() *JSONNetwork { //NOT DEEPCOPY!
	jsonNetwork := &JSONNetwork{}
	jsonNetwork.NumLayers = network.NumLayers
	jsonNetwork.Biases = network.Biases
	jsonNetwork.LayerSizes = network.LayerSizes
	jsonNetwork.ActivationFunctions = []int{}
	for i := 0; i < network.NumLayers-1; i++ {
		jsonNetwork.ActivationFunctions = append(jsonNetwork.ActivationFunctions, activationfunction.ActivationFunctionToInt[network.ActivationFunctions[i]])
	}
	jsonNetwork.Weights = make([][][]float64, network.NumLayers-1)
	for i := 0; i < network.NumLayers-1; i++ {
		jsonNetwork.Weights[i] = make([][]float64, network.LayerSizes[i+1])
		for j := 0; j < network.LayerSizes[i+1]; j++ {
			jsonNetwork.Weights[i][j] = make([]float64, network.LayerSizes[i])
			for k := 0; k < network.LayerSizes[i]; k++ {
				jsonNetwork.Weights[i][j][k] = network.Weights[i][j].AtVec(k)
			}
		}
	}
	return jsonNetwork
}

func (jsonNetwork *JSONNetwork) ToNetwork() *Network {
	network := &Network{}
	network.NumLayers = jsonNetwork.NumLayers
	network.Biases = jsonNetwork.Biases
	network.LayerSizes = jsonNetwork.LayerSizes
	network.ActivationFunctions = make([]*activationfunction.ActivationFunction, jsonNetwork.NumLayers)
	for i := 0; i < jsonNetwork.NumLayers-1; i++ {
		network.ActivationFunctions[i] = activationfunction.IntToActivationFunction[jsonNetwork.ActivationFunctions[i]]
	}
	network.Weights = make([][]*mat.VecDense, jsonNetwork.NumLayers-1)
	for i := 0; i < jsonNetwork.NumLayers-1; i++ {
		network.Weights[i] = make([]*mat.VecDense, jsonNetwork.LayerSizes[i+1])
		for j := 0; j < jsonNetwork.LayerSizes[i+1]; j++ {
			network.Weights[i][j] = mat.NewVecDense(jsonNetwork.LayerSizes[i], make([]float64, jsonNetwork.LayerSizes[i]))
			for k := 0; k < jsonNetwork.LayerSizes[i]; k++ {
				network.Weights[i][j].SetVec(k, jsonNetwork.Weights[i][j][k])
			}
		}
	}
	return network
}

func NewNetwork(layerSizes []int, activationFunctions []*activationfunction.ActivationFunction) *Network {
	network := &Network{}

	numLayers := len(layerSizes)
	network.NumLayers = numLayers

	network.LayerSizes = layerSizes

	network.Weights = make([][]*mat.VecDense, numLayers-1)
	for i := 0; i < numLayers-1; i++ {
		network.Weights[i] = make([]*mat.VecDense, layerSizes[i+1])
		for j := 0; j < layerSizes[i+1]; j++ {
			network.Weights[i][j] = mat.NewVecDense(layerSizes[i], make([]float64, layerSizes[i]))
		}
	}

	network.Biases = make([][]float64, numLayers-1)
	for i := 0; i < numLayers-1; i++ {
		network.Biases[i] = make([]float64, layerSizes[i+1])
	}
	network.ActivationFunctions = activationFunctions
	return network
}

func (network *Network) Randomize(minWeight, maxWeight, minBias, maxBias float64) {
	for i := 0; i < len(network.Weights); i++ {
		for j := 0; j < len(network.Weights[i]); j++ {
			for k := 0; k < network.Weights[i][j].Len(); k++ {
				network.Weights[i][j].SetVec(k, random.RandomFloat64(minWeight, maxWeight))
			}
		}
	}

	for i := 0; i < len(network.Weights); i++ {
		for j := 0; j < len(network.Weights[i]); j++ {
			network.Biases[i][j] = random.RandomFloat64(minBias, maxBias)
		}
	}
}

func (network *Network) Run(inputs *mat.VecDense, returnNonOutputStates, returnStatesBeforeActivationFunction bool) (*mat.VecDense, []*mat.VecDense, []*mat.VecDense) {
	prevLayer := inputs

	states := []*mat.VecDense{}
	if returnNonOutputStates {
		states = append(states, inputs)
	}

	statesBeforeActivationFunctions := []*mat.VecDense{}
	if returnStatesBeforeActivationFunction {
		statesBeforeActivationFunctions = append(statesBeforeActivationFunctions, inputs)
	}

	var nextLayer *mat.VecDense
	for i := 1; i < network.NumLayers; i++ {
		nextLayer = mat.NewVecDense(network.LayerSizes[i], make([]float64, network.LayerSizes[i]))
		for j := 0; j < network.LayerSizes[i]; j++ {
			nextLayer.SetVec(j, mat.Dot(prevLayer, network.Weights[i-1][j])+network.Biases[i-1][j])
		}
		if returnStatesBeforeActivationFunction {
			statesBeforeActivationFunctions = append(statesBeforeActivationFunctions, deepcopy.Vector(nextLayer))
		}
		for j := 0; j < network.LayerSizes[i]; j++ {
			nextLayer.SetVec(j, network.ActivationFunctions[i-1].Eval(nextLayer.AtVec(j)))
		}
		if returnNonOutputStates {
			states = append(states, nextLayer)
		}
		prevLayer = nextLayer
	}
	return nextLayer, states, statesBeforeActivationFunctions
}

func (network *Network) Vary(maxDiff float64) {
	for i := 0; i < len(network.Weights); i++ {
		for j := 0; j < len(network.Weights[i]); j++ {
			for k := 0; k < network.Weights[i][j].Len(); k++ {
				network.Weights[i][j].SetVec(k, network.Weights[i][j].AtVec(k)+random.RandomFloat64(-maxDiff, maxDiff))
			}
		}
	}

	for i := 0; i < len(network.Weights); i++ {
		for j := 0; j < len(network.Weights[i]); j++ {
			network.Biases[i][j] += random.RandomFloat64(-maxDiff, maxDiff)
		}
	}
}

func (network *Network) Copy() *Network {
	result := &Network{}
	result.NumLayers = network.NumLayers
	result.Biases = deepcopy.PrimitiveSlice2D(network.Biases)
	result.Weights = deepcopy.VectorSlice2D(network.Weights)
	result.LayerSizes = deepcopy.PrimitiveSlice1D(network.LayerSizes)
	result.ActivationFunctions = deepcopy.PrimitiveSlice1D(network.ActivationFunctions)
	return result
}

func (network *Network) Derivative(states []*mat.VecDense, statesBeforeActivationFunctions []*mat.VecDense, groundTruth *mat.VecDense) ([][]*mat.VecDense, [][]float64) {
	weightDerivatives := make([][]*mat.VecDense, network.NumLayers-1)
	for i := 0; i < network.NumLayers-1; i++ {
		weightDerivatives[i] = make([]*mat.VecDense, network.LayerSizes[i+1])
		for j := 0; j < network.LayerSizes[i+1]; j++ {
			weightDerivatives[i][j] = mat.NewVecDense(network.LayerSizes[i], make([]float64, network.LayerSizes[i]))
		}
	}
	biasDerivatives := make([][]float64, network.NumLayers-1)
	for i := 0; i < network.NumLayers-1; i++ {
		biasDerivatives[i] = make([]float64, network.LayerSizes[i+1])
	}

	currDerivatives := mat.NewVecDense(network.LayerSizes[network.NumLayers-1], make([]float64, network.LayerSizes[network.NumLayers-1]))
	for i := 0; i < network.LayerSizes[network.NumLayers-1]; i++ {
		currDerivatives.SetVec(i, 2*(states[network.NumLayers-1].AtVec(i)-groundTruth.AtVec(i)))
	}

	for i := network.NumLayers - 1; i >= 1; i-- {
		for j := 0; j < network.LayerSizes[i]; j++ {
			currDerivatives.SetVec(j, currDerivatives.AtVec(j)*network.ActivationFunctions[i-1].Derivative(statesBeforeActivationFunctions[i].AtVec(j)))
		}

		newDerivatives := mat.NewVecDense(network.LayerSizes[i-1], make([]float64, network.LayerSizes[i-1]))

		for j := 0; j < network.LayerSizes[i]; j++ {
			for k := 0; k < network.LayerSizes[i-1]; k++ {
				weightDerivatives[i-1][j].SetVec(k, states[i-1].AtVec(k)*currDerivatives.AtVec(j))
				newDerivatives.SetVec(k, newDerivatives.AtVec(k)+network.Weights[i-1][j].AtVec(k)*currDerivatives.AtVec(j))
			}
		}
		for j := 0; j < network.LayerSizes[i]; j++ {
			biasDerivatives[i-1][j] = currDerivatives.AtVec(j)
		}
		currDerivatives = newDerivatives
	}
	return weightDerivatives, biasDerivatives
}

func (network *Network) Learn(inputs *mat.VecDense, groundTruth *mat.VecDense, learnRate float64) (float64, mat.Vector) {
	output, states, statesBeforeActivationFunctions := network.Run(inputs, true, true)
	weightDerivatives, biasDerivatives := network.Derivative(states, statesBeforeActivationFunctions, groundTruth)

	cost := float64(0)
	for i := 0; i < network.LayerSizes[network.NumLayers-1]; i++ {
		cost += (output.AtVec(i) - groundTruth.AtVec(i)) * (output.AtVec(i) - groundTruth.AtVec(i))
	}

	for i := 0; i < network.NumLayers-1; i++ {
		for j := 0; j < network.LayerSizes[i+1]; j++ {
			for k := 0; k < network.LayerSizes[i]; k++ {
				network.Weights[i][j].SetVec(k, network.Weights[i][j].AtVec(k)-learnRate*weightDerivatives[i][j].AtVec(k))
			}
		}
	}
	for i := 0; i < network.NumLayers-1; i++ {
		for j := 0; j < network.LayerSizes[i+1]; j++ {
			network.Biases[i][j] -= biasDerivatives[i][j] * learnRate
		}
	}
	return cost, output
}
