package codec

import (
	"encoding/json"
	"nn/feedforward"
	"os"
)

func EncodeNetwork(network *feedforward.Network, filename string) {
	file, _ := os.Create(filename)
	defer file.Close()

	// "men" - Rojeel Sharma, 2023
	encodedNetwork, _ := json.Marshal(network.ToJSONNetwork())
	file.Write(encodedNetwork)
}
