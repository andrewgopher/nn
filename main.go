package main

import (
	"fmt"
	"math"
	"math/rand"
	"nn/feedforward"
	"nn/random"
	"nn/render"
	"sort"
	"time"

	"github.com/goccy/go-graphviz"
	"gonum.org/v1/gonum/mat"
)

func calcFitness(network *feedforward.Network, numSamples int) float64 {
	totalSamples := 0
	correctAnswers := 0
	for i := 0; i < numSamples; i++ {
		x, y := random.RandomFloat64(-10, 10), random.RandomFloat64(-10, 10)
		outputs, _, _ := network.Run(mat.NewVecDense(2, []float64{x, y}), false, false)

		output := outputs.AtVec(0) > outputs.AtVec(1)

		groundTruth := 5 <= math.Sqrt(x*x+y*y) && math.Sqrt(x*x+y*y) <= 8
		if groundTruth == output {
			correctAnswers++
		}
		totalSamples++
	}
	return float64(correctAnswers) / float64(totalSamples)
}

func main() {
	rand.Seed(time.Now().UnixMilli())

	poolSize := 100
	numSelected := 10
	numRounds := 1000
	numSamples := 100

	pool := []*feedforward.Network{}
	for i := 0; i < poolSize; i++ {
		network := feedforward.NewNetwork([]int{2, 3, 4, 3, 2}, []feedforward.ActivationFunction{feedforward.Identity, feedforward.Identity, feedforward.Identity, feedforward.Identity, feedforward.Identity})
		network.Randomize(-1, 1, -1, 1)
		pool = append(pool, network)
	}

	nextPool := []*feedforward.Network{}

	var bestNetwork *feedforward.Network
	bestFitness := -0.01
	for i := 0; i < numRounds; i++ {
		fitnessSum := float64(0)

		fitnesses := make([]float64, poolSize)
		indices := make([]int, poolSize)
		for j := 0; j < poolSize; j++ {
			indices[j] = j
			fitnesses[j] = calcFitness(pool[j], numSamples)
			fitnessSum += fitnesses[j]
			if fitnesses[j] > bestFitness {
				bestFitness = fitnesses[j]
				bestNetwork = pool[j]
			}
		}
		sort.Slice(indices, func(i, j int) bool {
			return fitnesses[i] > fitnesses[j]
		})
		for i := 0; i < numSelected; i++ {
			origCopiedNetwork := pool[indices[i]].Copy()
			nextPool = append(nextPool, origCopiedNetwork)
			for j := 0; j < poolSize/numSelected-1; j++ {
				copiedNetwork := pool[indices[i]].Copy()
				copiedNetwork.Vary(0.1)
				nextPool = append(nextPool, copiedNetwork)
			}
		}
		pool = nextPool
		nextPool = []*feedforward.Network{}
		fmt.Printf("Round %v, avg fitness %v, best fitness %v\n", i, fitnessSum/float64(poolSize), bestFitness)
	}

	render.RenderFeedForward(bestNetwork, mat.NewVecDense(2, make([]float64, 2)), 20, 20, graphviz.PNG, "plots/feedforward.png")
}
