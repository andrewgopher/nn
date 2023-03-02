package main

import (
	"fmt"
	"math"
	"math/rand"
	"nn/feedforward"
	"nn/mathext"
	"nn/random"
	"nn/render"
	"os"
	"sort"
	"time"

	"github.com/goccy/go-graphviz"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func calcFitness(network *feedforward.Network, numSamples int) float64 {
	totalSamples := 0
	correctAnswers := 0
	for i := 0; i < numSamples; i++ {
		x, y := random.RandomFloat64(-10, 10), random.RandomFloat64(-10, 10)
		output, _, _ := network.Run(mat.NewVecDense(2, []float64{x, y}), false, false)

		verdict := output.AtVec(0) > output.AtVec(1)

		groundTruth := 5 <= math.Sqrt(x*x+y*y) && math.Sqrt(x*x+y*y) <= 8
		if groundTruth == verdict {
			correctAnswers++
		}
		totalSamples++
	}
	return float64(correctAnswers) / float64(totalSamples)
}

func geneticAlgorithm() {
	poolSize := 100
	numSelected := 10
	numRounds := 100
	numSamples := 100

	pool := []*feedforward.Network{}
	for i := 0; i < poolSize; i++ {
		network := feedforward.NewNetwork([]int{2, 3, 4, 3, 2}, []*feedforward.ActivationFunction{feedforward.Tanh, feedforward.Tanh, feedforward.Tanh, feedforward.Tanh})
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

func gradientDescent() {
	numSteps := 10000
	avgRange := 1000

	avgCostPlot := plot.New()
	avgCostPlot.X.Label.Text = "step"
	avgCostPlot.Y.Label.Text = "avg cost (last 100 steps)"

	currCostSamples := []float64{}

	avgCosts := []float64{}

	network := feedforward.NewNetwork([]int{2, 3, 4, 3, 2}, []*feedforward.ActivationFunction{feedforward.Sigmoid, feedforward.Sigmoid, feedforward.Sigmoid, feedforward.Sigmoid})
	network.Randomize(-1, 1, -1, 1)
	for i := 0; i < numSteps; i++ {
		x, y := random.RandomFloat64(-10, 10), random.RandomFloat64(-10, 10)
		groundTruth := mat.NewVecDense(2, []float64{0, 0})
		groundTruthBool := 5 <= math.Sqrt(x*x+y*y) && math.Sqrt(x*x+y*y) <= 8
		if groundTruthBool {
			groundTruth.SetVec(0, 1)
			groundTruth.SetVec(1, 0)
		} else {
			groundTruth.SetVec(0, 0)
			groundTruth.SetVec(1, 1)
		}

		currCost := mathext.RoundFloat64(network.Learn(mat.NewVecDense(2, []float64{x, y}), groundTruth, 0.01), 5)
		if len(currCostSamples) == avgRange {
			currCostSamples = currCostSamples[1:]
		}
		currCostSamples = append(currCostSamples, currCost)

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
	if err := avgCostPlot.Save(4*vg.Inch, 4*vg.Inch, "plots/cost.png"); err != nil {
		panic(err)
	}

	render.RenderFeedForward(network, mat.NewVecDense(2, make([]float64, 2)), 20, 20, graphviz.PNG, "plots/feedforward.png")
}

func main() {
	rand.Seed(time.Now().UnixMilli())
	demos := map[string]func(){"geneticAlgorithm": geneticAlgorithm, "gradientDescent": gradientDescent}
	if len(os.Args) == 1 {
		fmt.Println("please specify a demo to run:")
		for demoName, _ := range demos {
			fmt.Println(demoName)
		}
	} else {
		demo, ok := demos[os.Args[1]]
		if !ok {
			fmt.Println("please specify a valid demo:")
			for demoName, _ := range demos {
				fmt.Println(demoName)
			}
		} else {
			demo()
		}
	}
}
