package geneticalgorithm

import (
	"fmt"
	"math"
	"nn/activationfunction"
	"nn/codec"
	"nn/feedforward"
	"nn/render"
	"sort"

	"github.com/goccy/go-graphviz"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func calcCost(network *feedforward.Network, numSamples int, genInput func() (*mat.VecDense, *mat.VecDense)) float64 {
	totalCost := float64(0)
	for i := 0; i < numSamples; i++ {
		input, groundTruthOutput := genInput()
		output, _, _ := network.Run(input, false, false)

		currCost := float64(0)
		for j := 0; j < input.Len(); j++ {
			currCost += (output.AtVec(j) - groundTruthOutput.AtVec(j)) * (output.AtVec(j) - groundTruthOutput.AtVec(j))
		}
		totalCost += currCost
	}
	return totalCost / float64(numSamples)
}

func Run(poolSize, numSelected, numSteps, numSamples int, layerSizes []int, activationFunctions []*activationfunction.ActivationFunction, genInput func() (*mat.VecDense, *mat.VecDense)) {
	avgCostRange := 1

	avgCostPlot := plot.New()
	avgCostPlot.X.Label.Text = "step"
	avgCostPlot.Y.Label.Text = "avg cost (last 100 steps)"

	currCostSamples := []float64{}
	avgCosts := []float64{}

	pool := []*feedforward.Network{}
	for i := 0; i < poolSize; i++ {
		network := feedforward.NewNetwork(layerSizes, activationFunctions)
		network.Randomize(-1, 1, -1, 1)
		pool = append(pool, network)
	}

	nextPool := []*feedforward.Network{}

	var bestNetwork *feedforward.Network
	bestCost := math.MaxFloat64
	for i := 0; i < numSteps; i++ {
		costSum := float64(0)

		costs := make([]float64, poolSize)
		indices := make([]int, poolSize)
		for j := 0; j < poolSize; j++ {
			indices[j] = j
			costs[j] = calcCost(pool[j], numSamples, genInput)
			costSum += costs[j]
			if costs[j] < bestCost {
				bestCost = costs[j]
				bestNetwork = pool[j]
			}
		}
		sort.Slice(indices, func(i, j int) bool {
			return costs[i] < costs[j]
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
		fmt.Printf("Step %v | avg cost %v | best cost %v\n", i, costSum/float64(poolSize), bestCost)

		if len(currCostSamples) == avgCostRange {
			currCostSamples = currCostSamples[1:]
		}
		currCostSamples = append(currCostSamples, bestCost)

		currCostSampleSum := float64(0)
		for _, cost := range currCostSamples {
			currCostSampleSum += cost
		}
		avgCosts = append(avgCosts, currCostSampleSum/float64(len(currCostSamples)))
	}

	avgCostPlotPoints := make(plotter.XYs, numSteps)
	for i := 0; i < numSteps; i++ {
		avgCostPlotPoints[i].X = float64(i)
		avgCostPlotPoints[i].Y = avgCosts[i]
	}
	plotutil.AddLinePoints(avgCostPlot, "avg cost", avgCostPlotPoints)
	if err := avgCostPlot.Save(4*vg.Inch, 4*vg.Inch, "output/cost.png"); err != nil {
		panic(err)
	}

	render.RenderFeedForward(bestNetwork, mat.NewVecDense(2, make([]float64, 2)), 20, 20, graphviz.PNG, "output/feedforward.png")

	codec.EncodeNetwork(bestNetwork, "output/network.json")
}
