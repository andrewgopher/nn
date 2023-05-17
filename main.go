package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"nn/activationfunction"
	"nn/feedforward"
	"nn/geneticalgorithm"
	"nn/gradientdescent"
	"nn/random"
	"os"
	"os/exec"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

func readFile(name string) ([]byte, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}
	bytes := make([]byte, fileInfo.Size())
	_, err = file.Read(bytes)
	if err != nil {
		return nil, err
	}
	return bytes, nil
}

func genPoint() (*mat.VecDense, *mat.VecDense) {
	input := mat.NewVecDense(2, []float64{random.RandomFloat64(-10, 10), random.RandomFloat64(-10, 10)})
	output := mat.NewVecDense(2, []float64{0, 0})
	if -5 <= input.AtVec(0)+input.AtVec(1) && input.AtVec(0)+input.AtVec(1) <= 5 {
		output.SetVec(0, 1)
		output.SetVec(1, 0)
	} else {
		output.SetVec(0, 0)
		output.SetVec(1, 1)
	}
	return input, output
}

func classifyPointGeneticAlgorithm() {
	geneticalgorithm.Run(100, 50, 3000, 50, []int{2, 3, 4, 3, 2}, []*activationfunction.ActivationFunction{activationfunction.Sigmoid, activationfunction.Sigmoid, activationfunction.Sigmoid, activationfunction.Sigmoid}, genPoint)
}

func classifyPointGradientDescent() {
	gradientdescent.Run(100000, []int{2, 3, 4, 3, 2}, []*activationfunction.ActivationFunction{activationfunction.Sigmoid, activationfunction.Sigmoid, activationfunction.Sigmoid, activationfunction.Sigmoid}, genPoint)
}

func parseDigitDataset() ([][][]int, []int, error) {
	// digitImagesFile, _ := os.Open("datasets/digit_images.bin")
	// digitLabelsFile, _ := os.Open("datasets/digit_labels.bin")

	// digitImagesFileInfo, _ := digitImagesFile.Stat()
	// digitLabelsFileInfo, _ := digitImagesFile.Stat()

	// imagesFileBytes := make([]byte, digitImagesFileInfo.Size())
	// digitImagesFile.Read(imagesFileBytes)
	imagesFileBytes, err := readFile("datasets/digit_images.bin")
	if err != nil {
		return nil, nil, err
	}
	currInd := 0
	currImages := [][][]int{}
	for currInd < len(imagesFileBytes) {
		currImages = append(currImages, make([][]int, 28))
		for i := 0; i < 28; i++ {
			currImages[len(currImages)-1][i] = make([]int, 28)
			for j := 0; j < 28; j++ {
				currImages[len(currImages)-1][i][j] = int(imagesFileBytes[currInd])
				currInd++
			}
		}
	}
	labelsFileBytes, err := readFile("datasets/digit_labels.bin")
	if err != nil {
		return nil, nil, err
	}
	currLabels := []int{}
	for i := 0; i < len(labelsFileBytes); i++ {
		currLabels = append(currLabels, int(labelsFileBytes[i]))
	}
	return currImages, currLabels, nil
}

var digitImages [][][]int
var digitLabels []int

func genDigit() (*mat.VecDense, *mat.VecDense) {
	i := random.RandomInt(0, len(digitImages)-1)
	input := mat.NewVecDense(28*28, make([]float64, 28*28))
	for j := 0; j < 28; j++ {
		for k := 0; k < 28; k++ {
			input.SetVec(j*28+k, float64(digitImages[i][j][k]))
		}
	}
	output := mat.NewVecDense(10, make([]float64, 10))
	output.SetVec(digitLabels[i], 1)
	return input, output
}

func trainClassifyDigit() {
	fmt.Println("parsing digit dataset...")
	digitImages, digitLabels, _ = parseDigitDataset()
	fmt.Println("finished parsing digit datset")
	gradientdescent.Run(100000, []int{28 * 28, 384, 192, 91, 10}, []*activationfunction.ActivationFunction{activationfunction.Sigmoid, activationfunction.Sigmoid, activationfunction.Sigmoid, activationfunction.Sigmoid}, genDigit)
}

func runNeuralNetwork() {
	networkFileBytes, err := readFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
	}

	jsonNetwork := &feedforward.JSONNetwork{}
	json.Unmarshal(networkFileBytes, jsonNetwork)
	network := jsonNetwork.ToNetwork()

	inputsSlice := []float64{}
	json.Unmarshal([]byte(os.Args[3]), &inputsSlice)

	numInputs := len(inputsSlice)
	inputs := mat.NewVecDense(numInputs, inputsSlice)
	outputs, _, _ := network.Run(inputs, false, false)
	fmt.Print("[")
	for i := 0; i < network.LayerSizes[network.NumLayers-1]; i++ {
		fmt.Print(outputs.AtVec(i))
		if i != network.LayerSizes[network.NumLayers-1]-1 {
			fmt.Print(",")
		}
	}
	fmt.Println("]")
}

func queryDigitDataset() {
	parseDigitDataset()
	imageIndex64, _ := strconv.ParseInt(os.Args[2], 10, 0)
	imageIndex := int(imageIndex64) - 1
	digitImages, digitLabels, _ = parseDigitDataset()
	output := "["
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			output += strconv.Itoa(digitImages[imageIndex][i][j])
			if j != 27 || i != 27 {
				output += ","
			}
		}
	}
	output += "]"
	fmt.Print("{\"Image\":" + output + ",")
	fmt.Println("\"Label\":" + strconv.Itoa(digitLabels[imageIndex]) + "}")
}

func classifyDigitInDataset() { //TODO: clean this up
	digitJSON, _ := exec.Command("./nn", "queryDigitDataset", os.Args[2]).Output()
	digit := struct {
		Image []int
		Label int
	}{}
	json.Unmarshal(digitJSON, &digit)

	imageJSON := "["
	for i := 0; i < 784; i++ {
		imageJSON += strconv.Itoa(digit.Image[i])
		if i != 783 {
			imageJSON += ","
		}
	}
	imageJSON += "]"
	outputJSON, _ := exec.Command("./nn", "runNeuralNetwork", os.Args[3], imageJSON).Output()
	output := []float64{}
	json.Unmarshal(outputJSON, &output)

	maxProbabilityInd := 0
	maxProbability := output[0]
	for i := 0; i < len(output); i++ {
		if output[i] > maxProbability {
			maxProbability = output[i]
			maxProbabilityInd = i
		}
	}

	fmt.Printf("{\"Output\":%v,\"GroundTruth\":%v}\n", maxProbabilityInd, digit.Label)
}

func getRoot(w http.ResponseWriter, r *http.Request) {
	fmt.Println(r.Method, r.URL)
	image, ok := r.URL.Query()["image"]
	if !ok {
		digitWebpageBytes, err := readFile("webpages/digit.html")
		if err != nil {
			io.WriteString(w, err.Error())
		} else {
			io.WriteString(w, string(digitWebpageBytes))
		}
	} else {
		output, err := exec.Command("./nn", "runNeuralNetwork", os.Args[2], image[0]).Output()
		if err != nil {
			io.WriteString(w, err.Error())
		} else {
			io.WriteString(w, string(output))
		}
	}
}

func getRandom(w http.ResponseWriter, r *http.Request) {
	fmt.Println(r.Method, r.URL)
	randomDigit, err := exec.Command("./nn", "randomDigitDataset").Output()
	if err != nil {
		io.WriteString(w, err.Error())
	} else {
		io.WriteString(w, string(randomDigit))
	}
}

func classifyDigitWebserver() {
	http.HandleFunc("/", getRoot)
	http.HandleFunc("/random", getRandom)

	http.ListenAndServe(":8080", nil)
}

func randomDigitDataset() {
	digitImages, digitLabels, _ := parseDigitDataset()
	imageIndex := rand.Intn(len(digitImages))
	digitImages, digitLabels, _ = parseDigitDataset()
	output := "["
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			output += strconv.Itoa(digitImages[imageIndex][i][j])
			if j != 27 || i != 27 {
				output += ","
			}
		}
	}
	output += "]"
	fmt.Print("{\"Image\":" + output + ",")
	fmt.Println("\"Label\":" + strconv.Itoa(digitLabels[imageIndex]) + "}")
}

func main() {
	rand.Seed(time.Now().UnixMilli())
	demos := map[string]struct {
		runFunc    func()
		descripton string
	}{"classifyPointGeneticAlgorithm": {classifyPointGeneticAlgorithm, "checks if the sum of x and y values is >= -5 and <= 5 using the genetic algorithm"}, "classifyPointGradientDescent": {classifyPointGradientDescent, "checks if the sum of x and y values is >= -5 and <= 5 using gradient descent"}, "trainClassifyDigit": {trainClassifyDigit, "train classifying digits using nn"}, "runNeuralNetwork": {runNeuralNetwork, "run neural network"}, "queryDigitDataset": {queryDigitDataset, "output the kth image in a 1D JSON list"}, "classifyDigitInDataset": {classifyDigitInDataset, "classify kth digit in dataset"}, "classifyDigitWebserver": {classifyDigitWebserver, "start digit classification web interface"}, "randomDigitDataset": {randomDigitDataset, "random digit in dataset"}}
	if len(os.Args) == 1 {
		fmt.Println("please specify a demo to run:")
		for demoName, demo := range demos {
			fmt.Printf("%v: %v\n", demoName, demo.descripton)
		}
	} else {
		if _, err := os.Stat("output/"); err != nil {
			os.Mkdir("output/", 0775)
		}
		demo, ok := demos[os.Args[1]]
		if !ok {
			fmt.Println("please specify a valid demo:")
			for demoName := range demos {
				fmt.Println(demoName)
			}
		} else {
			demo.runFunc()
		}
	}
}
