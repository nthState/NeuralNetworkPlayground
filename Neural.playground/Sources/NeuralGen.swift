

import Foundation

let LEARNING_CONSTANT:Float = 0.5

func sigmoid(x:Float) -> Float
{
    return 1 / (1 + exp(-x))
}


//class Connection {}

func == (left: Neuron, right: Neuron) -> Bool {
    return (left.id == right.id)
}

class Neuron
{
    var connections:[Connection] = [Connection]()
    var output:Float = 0.0
    var isBias:Bool = false
    var id:String!
    
    init()
    {
        id = NSUUID().uuidString
        isBias = false
    }
    
    convenience init(bias:Bool)
    {
        self.init()
        isBias = true
    }
    
    func addConnection(connection:Connection)
    {
        connections.append(connection)
    }
    
    func calculate()
    {
        if isBias == true
        {
            //print("skip")
            return
        }
        
        var sum:Float = 0;
        var bias:Float = 0;
        for i in 0..<connections.count
        {
            let c = connections[i]
            let from = c.from!
            let to = c.to!
            if (to == self) {
                if from.isBias
                {
                    bias += from.output*c.weight
                } else {
                    sum += from.output*c.weight
                }
            }
        }
        // Output is result of sigmoid function
        output = sigmoid(x: bias+sum)
    }
    
    func debugPrint()
    {
        print("Neuron: \(id!)")
        for c in connections
        {
            c.debugPrint()
        }
    }
}

class Connection
{
    var from:Neuron!
    var to:Neuron!
    var weight:Float = 0.0
    
    init(from:Neuron, to:Neuron) {
        self.from = from
        self.to = to
        let we = Float(arc4random()) /  Float(UInt32.max)
        self.weight = we
    }
    
    func adjustWeight(deltaWeight:Float)
    {
        weight += deltaWeight
    }
    
    func debugPrint()
    {
        print("Connection: \(from!.id!), \(to!.id!), weight: \(weight)")
    }
}

public class Network
{
    var inputs:[Neuron]!
    var hidden:[Neuron]!
    var output:Neuron!
    
    public init()
    {
        inputs = [Neuron]()
        hidden = [Neuron]()
        
        for _ in 1...2 {
            inputs.append(Neuron())
        }
        
        for _ in 1...4 {
            hidden.append(Neuron())
        }
        
        //print("inputs length: \(inputs.count)")
        
        inputs[inputs.count-1] = Neuron(bias: true)
        hidden[hidden.count-1] = Neuron(bias: true)
        
        output = Neuron()
        
        setupInputHidden()
        setupHiddenOutput()
    }
    
    func setupInputHidden()
    {
        for i in 0..<inputs.count
        {
            for j in 0..<hidden.count-1
            {
                let connection = Connection(from: inputs[i], to: hidden[j])
                inputs[i].addConnection(connection: connection)
                hidden[j].addConnection(connection: connection)
            }
        }
    }
    
    func setupHiddenOutput()
    {
        for i in 0..<hidden.count
        {
            let connection = Connection(from: hidden[i], to: output)
            hidden[i].addConnection(connection: connection)
            output.addConnection(connection: connection)
        }
    }
    
    public func feedForward(inputValues:[Float]) -> Float
    {
        for i in 0..<inputValues.count
        {
            //print("input: \(inputValues[i])")
            inputs[i].output = inputValues[i]
        }
        
        for i in 0..<hidden.count-1
        {
            hidden[i].calculate()
            //print("output \(hidden[i].output)")
        }
        
        output.calculate()
        
        //print("feed complete: \(output.output)")
        
        return output.output
    }
    
    
    func train(inputs:[Float], answer:Float) -> Float
    {
        let result = feedForward(inputValues: inputs)
        //print("result: \(result)")
        
        // This is where the error correction all starts
        // Derivative of sigmoid output function * diff between known and guess
        let deltaOutput = result*(1-result) * (answer-result)
        //print("deltaOutput: \(deltaOutput)")
        
        
        // BACKPROPOGATION
        // This is easier b/c we just have one output
        // Apply Delta to connections between hidden and output
        var connections = output.connections
        for i in 0..<connections.count
        {
            let c = connections[i]
            let neuron = c.from
            let output:Float = neuron!.output
            let deltaWeight:Float = output*deltaOutput
            c.adjustWeight(deltaWeight: LEARNING_CONSTANT*deltaWeight)
            //print("adjustWeight: \(deltaWeight)")
        }
        
        // ADJUST HIDDEN WEIGHTS
        for i in 0..<hidden.count
        {
            connections = hidden[i].connections
            var sum:Float  = 0
            // Sum output delta * hidden layer connections (just one output)
            for j in 0..<connections.count
            {
                let c = connections[j]
                // Is this a connection from hidden layer to next layer (output)?
                if (c.from == hidden[i]) {
                    sum += c.weight*deltaOutput
                }
            }
            // Then adjust the weights coming in based:
            // Above sum * derivative of sigmoid output function for hidden neurons
            for j in 0..<connections.count
            {
                let c = connections[j]
                // Is this a connection from previous layer (input) to hidden layer?
                if (c.to == hidden[i]) {
                    let output = hidden[i].output
                    var deltaHidden = output * (1 - output)  // Derivative of sigmoid(x)
                    deltaHidden *= sum;   // Would sum for all outputs if more than one output
                    let neuron = c.from;
                    let deltaWeight = neuron!.output*deltaHidden
                    c.adjustWeight(deltaWeight: LEARNING_CONSTANT*deltaWeight);
                }
            }
        }
        
        //print("train complete")
        
        return result;
    }
    
    public func multitrain()
    {
        for _ in 0..<10000
        {
            train(inputs: [0,0], answer: 0)
            train(inputs: [0,1], answer: 1)
            train(inputs: [1,0], answer: 1)
            train(inputs: [1,1], answer: 0)
            train(inputs: [0.3,0], answer: 0)
        }

    }
    
    public func debugPrint()
    {
        print("Inputs")
        for i in inputs
        {
            i.debugPrint()
        }
        
        print("Hidden")
        for h in hidden
        {
            h.debugPrint()
        }
    }
}

