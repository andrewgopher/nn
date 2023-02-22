package render

import (
	"fmt"
	"nn/feedforward"
	"nn/mathext"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"gonum.org/v1/gonum/mat"
)

func RenderFeedForward(network *feedforward.Network, inputs mat.MutableVector, width, height float64, format graphviz.Format, filename string) {
	_, states, _ := network.Run(inputs, true, false)

	g := graphviz.New()
	g.SetLayout(graphviz.NEATO)
	graph, err := g.Graph()
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := graph.Close(); err != nil {
			panic(err)
		}
		g.Close()
	}()

	nodes := make([][]*cgraph.Node, network.NumLayers)
	for i := 0; i < network.NumLayers; i++ {
		nodes[i] = make([]*cgraph.Node, network.LayerSizes[i])
		for j := 0; j < network.LayerSizes[i]; j++ {
			currNode, err := graph.CreateNode(fmt.Sprint(i) + " " + fmt.Sprint(j))
			if err != nil {
				panic(err)
			}

			if i == 0 {
				currNode.SetLabel(fmt.Sprint(inputs.AtVec(j)))
			} else {
				currNode.SetLabel("bias " + fmt.Sprint(mathext.RoundFloat64(network.Biases[i-1][j], 2)) + " output " + fmt.Sprint(mathext.RoundFloat64(states[i].AtVec(j), 2)))
			}
			currNode.SetPos(width/float64(network.NumLayers+1)*float64(i+1), height/float64(network.LayerSizes[i]+1)*float64(j+1))
			currNode.SetPin(true)

			if i > 0 {
				for k := 0; k < network.LayerSizes[i-1]; k++ {
					currEdge, err := graph.CreateEdge(fmt.Sprint(i)+" "+fmt.Sprint(j)+" "+fmt.Sprint(k), nodes[i-1][k], currNode)
					if err != nil {
						panic(err)
					}

					currEdge.SetHeadLabel(fmt.Sprint(mathext.RoundFloat64(network.Weights[i-1][j].AtVec(k), 2)))
					currEdge.SetLabelDistance(3)
				}
			}
			nodes[i][j] = currNode
		}
	}

	if err := g.RenderFilename(graph, format, filename); err != nil {
		panic(err)
	}
}
