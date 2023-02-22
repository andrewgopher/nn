package feedforward

import (
	"math"
	"nn/deepcopy"
	"nn/random"

	"gonum.org/v1/gonum/mat"
)

type ActivationFunction func(float64) float64

func Identity(x float64) float64 {
	return x
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

type Network struct {
	NumLayers           int
	LayerSizes          []int
	Weights             [][]mat.MutableVector
	Biases              [][]float64
	ActivationFunctions []ActivationFunction //per layer
}

func NewNetwork(layerSizes []int, activationFunctions []ActivationFunction) *Network {
	network := &Network{}

	numLayers := len(layerSizes)
	network.NumLayers = numLayers

	network.LayerSizes = layerSizes

	network.Weights = make([][]mat.MutableVector, numLayers-1)
	for i := 0; i < numLayers-1; i++ {
		network.Weights[i] = make([]mat.MutableVector, layerSizes[i+1])
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

func (network *Network) Run(inputs mat.MutableVector, returnNonOutputStates, returnStatesBeforeActivationFunction bool) (mat.MutableVector, []mat.MutableVector, []mat.MutableVector) {
	prevLayer := deepcopy.Vector(inputs)
	for i := 0; i < network.LayerSizes[0]; i++ {
		prevLayer.SetVec(i, network.ActivationFunctions[0](prevLayer.AtVec(i)))
	}

	states := []mat.MutableVector{}
	if returnNonOutputStates {
		states = append(states, inputs)
	}

	statesBeforeActivationFunctions := []mat.MutableVector{}
	if returnStatesBeforeActivationFunction {
		statesBeforeActivationFunctions = append(statesBeforeActivationFunctions, inputs)
	}

	var nextLayer mat.MutableVector
	for i := 1; i < network.NumLayers; i++ {
		nextLayer = mat.NewVecDense(network.LayerSizes[i], make([]float64, network.LayerSizes[i]))
		for j := 0; j < network.LayerSizes[i]; j++ {
			nextLayer.SetVec(j, mat.Dot(prevLayer, network.Weights[i-1][j])+network.Biases[i-1][j])
		}
		if returnStatesBeforeActivationFunction {
			statesBeforeActivationFunctions = append(statesBeforeActivationFunctions, deepcopy.Vector(nextLayer))
		}
		for j := 0; j < network.LayerSizes[i]; j++ {
			nextLayer.SetVec(j, network.ActivationFunctions[i](nextLayer.AtVec(j)))
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

func (network *Network) Learn(states []mat.MutableVector, statesBeforeActivationFunctions []mat.MutableVector, groundTruth mat.MutableVector, learnRate float64) {
	derivatives := mat.NewVecDense(network.LayerSizes[network.NumLayers-1], make([]float64, network.LayerSizes[network.NumLayers-1]))
	for i := 0; i < network.LayerSizes[network.NumLayers-1]; i++ {
		derivatives.SetVec(i, 2*(states[network.NumLayers-1].AtVec(i)-groundTruth.AtVec(i)))
	}

}
