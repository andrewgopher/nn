package gradientdescent

import (
	"fmt"
	"nn/activationfunction"
	"nn/codec"
	"nn/feedforward"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func Run(numSteps int, layerSizes []int, activationFunctions []*activationfunction.ActivationFunction, genInput func() (*mat.VecDense, *mat.VecDense)) {
	avgCostRange := 1000

	avgCostPlot := plot.New()
	avgCostPlot.X.Label.Text = "step"
	avgCostPlot.Y.Label.Text = "avg cost (last 100 steps)"

	currCostSamples := []float64{}
	avgCosts := []float64{}

	network := feedforward.NewNetwork(layerSizes, activationFunctions)
	network.Randomize(-1, 1, -1, 1)
	for i := 0; i < numSteps; i++ {
		input, groundTruthOutput := genInput()

		currCost, _ := network.Learn(input, groundTruthOutput, 0.02)
		if len(currCostSamples) == avgCostRange {
			currCostSamples = currCostSamples[1:]
		}
		currCostSamples = append(currCostSamples, currCost)

		currCostSampleSum := float64(0)
		for _, cost := range currCostSamples {
			currCostSampleSum += cost
		}
		avgCosts = append(avgCosts, currCostSampleSum/float64(len(currCostSamples)))
		fmt.Printf("Step %v | cost %v\n", i, currCostSampleSum/float64(len(currCostSamples)))
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

	// render.RenderFeedForward(network, mat.NewVecDense(network.LayerSizes[0], make([]float64, network.LayerSizes[0])), 20, 20, graphviz.PNG, "output/feedforward.png")
	codec.EncodeNetwork(network, "output/network.json")
}
